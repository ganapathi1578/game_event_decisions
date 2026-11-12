from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from badminton_decision.constants import DEFAULT_MODEL_ID, TASK_TYPES
from badminton_decision.decision.layout import build_choice_layout, to_additive_mask
from badminton_decision.models.choice_head import ChoiceHead


def _transformers_imports():
    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise RuntimeError(
            "SmolVLM2 requires a recent Transformers release. Install the project dependencies with: "
            "pip install -e ."
        ) from exc
    return AutoModelForImageTextToText, AutoProcessor


def lora_target_regex(num_layers: int, last_k_layers: int, targets: list[str]) -> str:
    if last_k_layers < 1 or last_k_layers > num_layers:
        raise ValueError(f"last_k_layers must be between 1 and {num_layers}")
    ids = "|".join(str(i) for i in range(num_layers - last_k_layers, num_layers))
    target_group = "|".join(re.escape(name) for name in targets)
    return rf".*layers\.({ids})\.self_attn\.({target_group})"


def _group_legacy_visual_features(
    image_features: torch.Tensor,
    pixel_values: torch.Tensor,
    hidden_dim: int,
) -> list[torch.Tensor]:
    """Restore per-example visual features from the Transformers 4.57 tensor API.

    Transformers 4.57's ``SmolVLMModel.get_image_features`` returns a tensor shaped
    ``[num_real_images, visual_tokens, hidden_dim]``. Newer releases return a
    ``BaseModelOutputWithPooling`` whose ``pooler_output`` is already grouped by
    example. The legacy API flattens the batch and image axes, so reconstruct the
    groups using the same all-zero padded-image convention as Transformers.
    """
    if image_features.ndim != 3:
        raise RuntimeError(
            "Unsupported legacy SmolVLM visual feature shape "
            f"{tuple(image_features.shape)}; expected [images, tokens, hidden]."
        )
    if image_features.shape[-1] != hidden_dim:
        raise RuntimeError(
            f"Visual feature width {image_features.shape[-1]} does not match decoder width {hidden_dim}."
        )
    if pixel_values.ndim != 5:
        raise RuntimeError(
            "Legacy SmolVLM feature grouping expects pixel_values shaped "
            f"[batch, images, channels, height, width], got {tuple(pixel_values.shape)}."
        )

    batch_size, num_images = pixel_values.shape[:2]
    values_per_image = pixel_values.shape[2:].numel()
    valid_images = (pixel_values == 0).sum(dim=(-1, -2, -3)) != values_per_image
    # Match Transformers 4.57's fallback for batches containing only padded images.
    if not bool(valid_images.any()):
        valid_images.reshape(-1)[0] = True
    image_counts = [int(count) for count in valid_images.sum(dim=1).tolist()]
    expected_images = sum(image_counts)
    if expected_images != image_features.shape[0]:
        raise RuntimeError(
            "Cannot group legacy SmolVLM visual features: input contains "
            f"{expected_images} real images across {batch_size} examples, but the encoder "
            f"returned {image_features.shape[0]} image feature groups."
        )

    grouped = image_features.split(image_counts, dim=0)
    return [features.reshape(-1, features.shape[-1]) for features in grouped]


def merge_visual_features_into_context(
    context_embeds: torch.Tensor, context_ids: torch.Tensor,
    visual_features: torch.Tensor, image_token_id: int,
) -> torch.Tensor:
    """Replace native SmolVLM `<image>` placeholder embeddings with connector features."""
    image_mask = context_ids == int(image_token_id)
    expected = int(image_mask.sum().item())
    if expected != int(visual_features.shape[0]):
        raise RuntimeError(
            f"SmolVLM image placeholder/feature mismatch: {expected} <image> tokens "
            f"for {visual_features.shape[0]} visual features."
        )
    if visual_features.shape[-1] != context_embeds.shape[-1]:
        raise RuntimeError(
            f"Visual feature width {visual_features.shape[-1]} does not match "
            f"decoder width {context_embeds.shape[-1]}."
        )
    merged = context_embeds.clone()
    merged[image_mask] = visual_features.to(device=merged.device, dtype=merged.dtype)
    return merged


class SmolVLMChoiceModel(nn.Module):
    """SmolVLM2 visual features + BEV/Jev-style option blocks and pointer head.

    The prompt contains native SmolVLM frame/image placeholders. The pretrained vision
    tower and connector replace those embeddings in-place before the custom text decoder
    mask and shared option position IDs are applied.
    """
    def __init__(self, model_id: str = DEFAULT_MODEL_ID, dtype: str = "float16",
                 attn_implementation: str = "eager", lora_cfg: dict[str, Any] | None = None,
                 head_cfg: dict[str, Any] | None = None, gradient_checkpointing: bool = True,
                 local_files_only: bool = False):
        super().__init__()
        AutoModelForImageTextToText, AutoProcessor = _transformers_imports()
        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id, local_files_only=local_files_only)
        torch_dtype = getattr(torch, dtype)
        self.vlm = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            local_files_only=local_files_only,
        )
        # The reference requires a precomputed 4D additive mask. Eager attention consumes it directly.
        self.vlm.config.text_config._attn_implementation = "eager"
        self.vlm.model.text_model.config._attn_implementation = "eager"
        self.vlm.config.use_cache = False
        self.vlm.config.text_config.use_cache = False
        self.hidden_dim = int(self.vlm.config.text_config.hidden_size)
        self.option_open_ids = self.processor.tokenizer.encode("<option>", add_special_tokens=False)
        self.option_close_ids = self.processor.tokenizer.encode("</option>", add_special_tokens=False)
        self.answer_prompt_ids = self.processor.tokenizer.encode("The answer is:", add_special_tokens=False)
        self.image_token_id = int(self.processor.image_token_id)
        if not self.option_open_ids or not self.option_close_ids or not self.answer_prompt_ids:
            raise RuntimeError("Tokenizer produced empty option markers or answer prompt")

        # Freeze all VLM parameters first. LoRA adapters are inserted only in the selected text layers.
        for parameter in self.vlm.parameters():
            parameter.requires_grad_(False)
        self.vlm.model.vision_model.eval()
        self.vlm.model.connector.eval()

        lora_cfg = lora_cfg or {"enabled": True, "r": 8, "alpha": 16, "dropout": 0.0,
                                "last_k_layers": 12, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]}
        self.lora_enabled = bool(lora_cfg.get("enabled", True))
        self.lora_cfg = dict(lora_cfg)
        if self.lora_enabled:
            try:
                from peft import LoraConfig, get_peft_model
            except ImportError as exc:
                raise RuntimeError("PEFT is required for LoRA. Install with: pip install peft") from exc
            text_model = self.vlm.model.text_model
            n_layers = int(text_model.config.num_hidden_layers)
            target = lora_target_regex(n_layers, int(lora_cfg.get("last_k_layers", 12)),
                                       list(lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])))
            peft_config = LoraConfig(
                r=int(lora_cfg.get("r", 8)),
                lora_alpha=int(lora_cfg.get("alpha", 16)),
                lora_dropout=float(lora_cfg.get("dropout", 0.0)),
                target_modules=target,
                bias="none",
                task_type="FEATURE_EXTRACTION",
            )
            self.vlm.model.text_model = get_peft_model(text_model, peft_config)
            if gradient_checkpointing:
                try:
                    self.vlm.model.text_model.gradient_checkpointing_enable()
                except Exception:
                    pass
                try:
                    self.vlm.model.text_model.enable_input_require_grads()
                except Exception:
                    pass
        else:
            # Head-only mode: leave all pretrained VLM weights frozen.
            self.vlm.model.text_model.requires_grad_(False)

        head_cfg = head_cfg or {"hidden_dim": 512, "num_layers": 2, "num_task_types": len(TASK_TYPES)}
        self.head = ChoiceHead(
            hidden_dim=self.hidden_dim,
            head_dim=int(head_cfg.get("hidden_dim", 512)),
            num_layers=int(head_cfg.get("num_layers", 2)),
            num_task_types=int(head_cfg.get("num_task_types", len(TASK_TYPES))),
        )
        self.head.float()

    @property
    def tokenizer(self):
        return self.processor.tokenizer

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep frozen visual path deterministic and in eval mode.
        self.vlm.model.vision_model.eval()
        self.vlm.model.connector.eval()
        return self

    def _encode_visual(self, pixel_values: torch.Tensor, pixel_attention_mask: torch.Tensor | None):
        base_model = self.vlm.model
        dtype = next(base_model.vision_model.parameters()).dtype
        pixel_values = pixel_values.to(device=next(base_model.vision_model.parameters()).device, dtype=dtype)
        if pixel_attention_mask is not None:
            pixel_attention_mask = pixel_attention_mask.to(device=pixel_values.device, dtype=torch.bool)
        with torch.no_grad():
            # Transformers 4.57 exposes get_image_features(pixel_values, pixel_attention_mask)
            # and returns a flattened Tensor. Newer releases accept additional kwargs and
            # return BaseModelOutputWithPooling. Omitting return_dict=True is compatible
            # with both APIs; the model config defaults to structured outputs on newer versions.
            image_outputs = base_model.get_image_features(
                pixel_values=pixel_values,
                pixel_attention_mask=pixel_attention_mask,
            )

        # Transformers 4.57 returns [real_images, visual_tokens, hidden] directly.
        # Newer releases return an output object with pooler_output grouped by example.
        if isinstance(image_outputs, torch.Tensor):
            return _group_legacy_visual_features(image_outputs, pixel_values, self.hidden_dim)
        if isinstance(image_outputs, tuple):
            # A return-dict-disabled BaseModelOutputWithPooling is tuple-ordered as
            # (last_hidden_state, pooler_output, ...); prefer the grouped pooler output.
            if len(image_outputs) > 1:
                image_outputs = image_outputs[1]
            elif image_outputs:
                image_outputs = image_outputs[0]
            else:
                raise RuntimeError("SmolVLM returned an empty tuple for visual features.")
        features = getattr(image_outputs, "pooler_output", image_outputs)
        if isinstance(features, torch.Tensor):
            if features.ndim == 3:
                return [features[i] for i in range(features.shape[0])]
            if features.ndim == 2:
                return [features]
        features = list(features)
        if len(features) != pixel_values.shape[0]:
            raise RuntimeError(
                f"SmolVLM returned {len(features)} visual feature groups for batch size "
                f"{pixel_values.shape[0]}. Check the processor's [B, N, C, H, W] layout."
            )
        for index, item in enumerate(features):
            if item.ndim != 2 or item.shape[-1] != self.hidden_dim:
                raise RuntimeError(
                    f"Visual features for sample {index} have shape {tuple(item.shape)}; "
                    f"expected [tokens, {self.hidden_dim}]."
                )
        return features

    def _build_decoder_sequence(
        self,
        visual_features: torch.Tensor,
        context_ids: torch.Tensor,
        option_ids: list[torch.Tensor],
    ):
        """Build one context/options/answer sequence and its independent attention layout."""
        text_model = self.vlm.model.text_model
        device = next(text_model.parameters()).device
        embedding = text_model.get_input_embeddings()
        context_ids = context_ids.to(device=device, dtype=torch.long)
        context_embeds = embedding(context_ids)
        visual_features = visual_features.to(device=device, dtype=context_embeds.dtype)
        context_embeds = merge_visual_features_into_context(
            context_embeds, context_ids, visual_features, self.image_token_id
        )

        option_token_lists: list[list[int]] = []
        chunks = [context_embeds]
        for tokens in option_ids:
            token_list = tokens.detach().to(device="cpu", dtype=torch.long).tolist()
            if not token_list:
                raise ValueError("An option tokenized to an empty sequence")
            option_token_lists.append(token_list)
            full_ids = self.option_open_ids + token_list + self.option_close_ids
            chunks.append(embedding(torch.tensor(full_ids, dtype=torch.long, device=device)))

        answer_ids = torch.tensor(self.answer_prompt_ids, dtype=torch.long, device=device)
        chunks.append(embedding(answer_ids))
        layout = build_choice_layout(
            context_length=context_embeds.shape[0],
            option_token_ids=option_token_lists,
            option_open_ids=self.option_open_ids,
            option_close_ids=self.option_close_ids,
            answer_token_ids=self.answer_prompt_ids,
        )
        return torch.cat(chunks, dim=0), layout

    def _batched_text_forward(
        self,
        visual_features: list[torch.Tensor],
        context_ids: list[torch.Tensor],
        option_ids: list[list[torch.Tensor]],
        task_type_ids: list[str] | list[int],
    ) -> torch.Tensor:
        """Run one padded decoder call with an independent option mask per example."""
        batch_lengths = {len(visual_features), len(context_ids), len(option_ids), len(task_type_ids)}
        if len(batch_lengths) != 1:
            raise ValueError("Visual features, contexts, options and task IDs must share a batch size")
        if not visual_features:
            raise ValueError("Cannot run an empty decision batch")

        sequences: list[torch.Tensor] = []
        layouts = []
        for visual, context, options in zip(visual_features, context_ids, option_ids):
            sequence, layout = self._build_decoder_sequence(visual, context, options)
            sequences.append(sequence)
            layouts.append(layout)

        batch_size = len(sequences)
        max_length = max(int(sequence.shape[0]) for sequence in sequences)
        hidden_dim = int(sequences[0].shape[-1])
        device, dtype = sequences[0].device, sequences[0].dtype
        inputs_embeds = torch.zeros((batch_size, max_length, hidden_dim), device=device, dtype=dtype)
        position_ids = torch.zeros((batch_size, max_length), device=device, dtype=torch.long)
        allowed_attention = torch.zeros(
            (batch_size, max_length, max_length), device=device, dtype=torch.bool
        )

        for index, (sequence, layout) in enumerate(zip(sequences, layouts)):
            length = int(sequence.shape[0])
            inputs_embeds[index, :length] = sequence
            position_ids[index, :length] = layout.position_ids.to(device=device)
            allowed_attention[index, :length, :length] = layout.allowed_attention.to(device=device)
            # Padded query rows attend only to themselves; valid tokens cannot attend to padding.
            if length < max_length:
                padding = torch.arange(length, max_length, device=device)
                allowed_attention[index, padding, padding] = True

        additive_mask = to_additive_mask(allowed_attention, dtype=inputs_embeds.dtype)
        decoder_outputs = self.vlm.model.text_model(
            inputs_embeds=inputs_embeds,
            attention_mask=additive_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )
        hidden = decoder_outputs.last_hidden_state

        max_choices = max(len(layout.choice_read_indices) for layout in layouts)
        choice_hidden = hidden.new_zeros((batch_size, max_choices, hidden_dim))
        choice_mask = torch.zeros((batch_size, max_choices), dtype=torch.bool, device=device)
        answer_hidden = hidden.new_zeros((batch_size, hidden_dim))
        for index, layout in enumerate(layouts):
            count = len(layout.choice_read_indices)
            read_indices = layout.choice_read_indices.to(device=device)
            choice_hidden[index, :count] = hidden[index].index_select(0, read_indices)
            choice_mask[index, :count] = True
            answer_hidden[index] = hidden[index, layout.answer_end_index]

        task_ids = [TASK_TYPES[item] if isinstance(item, str) else int(item) for item in task_type_ids]
        task_tensor = torch.tensor(task_ids, device=device, dtype=torch.long)
        # Keep the choice head in FP32 when the decoder is running under FP16 autocast.
        with torch.autocast(device_type=device.type, enabled=False):
            return self.head(
                choice_hidden.float(), answer_hidden.float(), choice_mask, task_tensor
            )

    def _one_forward(
        self,
        visual_features: torch.Tensor,
        context_ids: torch.Tensor,
        option_ids: list[torch.Tensor],
        task_type_id: int,
    ) -> torch.Tensor:
        """Backward-compatible single-example helper."""
        return self._batched_text_forward(
            [visual_features], [context_ids], [option_ids], [task_type_id]
        )[0, :len(option_ids)]

    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        pixel_attention_mask: torch.Tensor | None = None,
        context_ids: list[torch.Tensor] | None = None,
        option_ids: list[list[torch.Tensor]] | None = None,
        task_type_ids: list[str] | list[int] | None = None,
        visual_features: torch.Tensor | list[torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if context_ids is None or option_ids is None or task_type_ids is None:
            raise ValueError("context_ids, option_ids and task_type_ids are required")
        if visual_features is None:
            if pixel_values is None:
                raise ValueError("Supply either cached visual_features or raw pixel_values")
            grouped_features = self._encode_visual(pixel_values, pixel_attention_mask)
        elif isinstance(visual_features, torch.Tensor):
            if visual_features.ndim != 3:
                raise ValueError(
                    "Batched cached visual_features must have shape [batch, visual_tokens, hidden_dim]"
                )
            grouped_features = [visual_features[index] for index in range(visual_features.shape[0])]
        else:
            grouped_features = list(visual_features)
        return self._batched_text_forward(grouped_features, context_ids, option_ids, task_type_ids)

    def trainable_parameter_counts(self) -> dict[str, int]:
        lora_params = 0
        for name, param in self.vlm.model.text_model.named_parameters():
            if param.requires_grad and "lora_" in name:
                lora_params += param.numel()
        head_params = sum(p.numel() for p in self.head.parameters() if p.requires_grad)
        total = lora_params + head_params
        return {"lora": lora_params, "head": head_params, "total_trainable": total}

    def save_checkpoint(self, checkpoint_dir: str | Path, metadata: dict[str, Any]):
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        adapter_dir = checkpoint_dir / "lora_adapter"
        if self.lora_enabled:
            self.vlm.model.text_model.save_pretrained(adapter_dir)
        torch.save(self.head.state_dict(), checkpoint_dir / "choice_head.pt")
        import json
        (checkpoint_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_decision_model(checkpoint_dir: str | Path, device: str = "cuda", local_files_only: bool = False):
    import json
    checkpoint_dir = Path(checkpoint_dir)
    metadata = json.loads((checkpoint_dir / "metadata.json").read_text(encoding="utf-8"))
    model_cfg = metadata["model_config"]
    model = SmolVLMChoiceModel(
        model_id=model_cfg["name"], dtype=model_cfg["dtype"],
        attn_implementation="eager", lora_cfg=model_cfg["lora"],
        head_cfg=model_cfg["head"], gradient_checkpointing=False,
        local_files_only=local_files_only,
    )
    if model.lora_enabled:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("PEFT is required to load this LoRA checkpoint: pip install peft") from exc
        base_text_model = model.vlm.model.text_model.get_base_model() if hasattr(model.vlm.model.text_model, "get_base_model") else model.vlm.model.text_model
        model.vlm.model.text_model = PeftModel.from_pretrained(
            base_text_model, checkpoint_dir / "lora_adapter", is_trainable=False
        )
    state = torch.load(checkpoint_dir / "choice_head.pt", map_location="cpu", weights_only=True)
    model.head.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
