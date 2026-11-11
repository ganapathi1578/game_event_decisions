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

    def _one_forward(self, visual_features: torch.Tensor, context_ids: torch.Tensor,
                     option_ids: list[torch.Tensor], task_type_id: int) -> torch.Tensor:
        device = visual_features.device
        text_model = self.vlm.model.text_model
        embedding = text_model.get_input_embeddings()
        context_ids = context_ids.to(device)
        context_embeds = embedding(context_ids)
        # Match SmolVLM's native multimodal merge while preserving the image markers.
        context_embeds = merge_visual_features_into_context(
            context_embeds, context_ids, visual_features, self.image_token_id
        )
        option_embeddings = []
        option_tokens_list: list[list[int]] = []
        for tokens in option_ids:
            tokens = tokens.to(device)
            option_tokens_list.append(tokens.tolist())
        layout = build_choice_layout(
            context_length=context_embeds.shape[0],
            option_token_ids=option_tokens_list,
            option_open_ids=self.option_open_ids,
            option_close_ids=self.option_close_ids,
            answer_token_ids=self.answer_prompt_ids,
        )
        chunks = [context_embeds]
        for option_tokens in option_ids:
            option_full = self.option_open_ids + option_tokens.tolist() + self.option_close_ids
            ids = torch.tensor(option_full, dtype=torch.long, device=device)
            option_embeddings.append(embedding(ids))
            chunks.append(option_embeddings[-1])
        answer_ids = torch.tensor(self.answer_prompt_ids, dtype=torch.long, device=device)
        chunks.append(embedding(answer_ids))
        inputs_embeds = torch.cat(chunks, dim=0).unsqueeze(0)
        position_ids = layout.position_ids.to(device).unsqueeze(0)
        additive_mask = to_additive_mask(layout.allowed_attention.to(device), dtype=inputs_embeds.dtype)
        hidden = text_model(
            inputs_embeds=inputs_embeds,
            attention_mask=additive_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state.float()
        choice_indices = layout.choice_read_indices.to(device)
        choice_embeddings = hidden[:, choice_indices, :]
        answer_embedding = hidden[:, layout.answer_end_index, :]
        task_tensor = torch.tensor([task_type_id], device=device, dtype=torch.long)
        choice_mask = torch.ones((1, len(option_ids)), device=device, dtype=torch.bool)
        with torch.autocast(device_type=device.type, enabled=False):
            logits = self.head(choice_embeddings.float(), answer_embedding.float(), choice_mask, task_tensor)
        return logits[0]

    def forward(self, pixel_values: torch.Tensor, pixel_attention_mask: torch.Tensor | None,
                context_ids: list[torch.Tensor], option_ids: list[list[torch.Tensor]],
                task_type_ids: list[str] | list[int], **kwargs) -> torch.Tensor:
        visual_features = self._encode_visual(pixel_values, pixel_attention_mask)
        logits_list = []
        for index, visual in enumerate(visual_features):
            task = task_type_ids[index]
            task_id = TASK_TYPES[task] if isinstance(task, str) else int(task)
            logits_list.append(self._one_forward(visual, context_ids[index], option_ids[index], task_id))
        max_choices = max(logit.numel() for logit in logits_list)
        output = torch.full((len(logits_list), max_choices), -torch.inf, device=logits_list[0].device)
        for i, logits in enumerate(logits_list):
            output[i, :logits.numel()] = logits
        return output

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
