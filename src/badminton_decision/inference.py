from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from badminton_decision.data.dataset import DecisionCollator
from badminton_decision.data.video import VideoFrameReader
from badminton_decision.models.smolvlm_decision import load_decision_model


def predict_video(checkpoint_dir: str | Path, video_path: str | Path, question: str, options: list[str],
                  start_seconds: float | None = None, end_seconds: float | None = None,
                  num_frames: int | None = None, device: str | None = None, task_type: str = "hit_type"):
    checkpoint_dir = Path(checkpoint_dir)
    metadata = json.loads((checkpoint_dir / "metadata.json").read_text(encoding="utf-8"))
    vcfg, ccfg = metadata["video_config"], metadata["choice_config"]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_decision_model(checkpoint_dir, device=device)
    video_path = Path(video_path)
    cap_fps = 30.0
    # Read FPS through OpenCV; the dataset annotation FPS is not available at inference time.
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if cap.isOpened():
        found_fps = cap.get(cv2.CAP_PROP_FPS)
        if found_fps and found_fps > 0:
            cap_fps = float(found_fps)
    cap.release()
    duration = float(vcfg["duration_seconds"])
    if start_seconds is not None and end_seconds is not None:
        center_seconds = (float(start_seconds) + float(end_seconds)) / 2.0
    elif start_seconds is not None:
        center_seconds = float(start_seconds) + duration / 2.0
    else:
        center_seconds = 0.0
    frame_count = int(num_frames or vcfg["num_frames"])
    reader = VideoFrameReader(num_frames=frame_count, duration_seconds=duration, cache_size=1)
    frames = reader.read(video_path, center_seconds * cap_fps, cap_fps, training=False)
    item = {
        "sample_id": "interactive", "match_id": video_path.stem, "frames": frames,
        "question": question, "options": options, "correct_index": 0,
        "answer": options[0], "task_type": task_type, "metadata": {},
    }
    collator = DecisionCollator(model.processor, ccfg["context_max_tokens"], ccfg["option_max_tokens"], ccfg["task_prompt"])
    batch = collator([item])
    with torch.no_grad():
        logits = model(
            pixel_values=batch["pixel_values"].to(device),
            pixel_attention_mask=batch["pixel_attention_mask"].to(device),
            context_ids=batch["context_ids"], option_ids=batch["option_ids"],
            task_type_ids=batch["task_type_names"],
        )[0]
    probabilities = torch.softmax(logits, dim=-1).cpu().tolist()
    ranked = sorted(zip(options, probabilities), key=lambda pair: pair[1], reverse=True)
    return {"question": question, "prediction": ranked[0][0], "confidence": ranked[0][1],
            "ranked_options": [{"option": name, "probability": prob} for name, prob in ranked]}


def main():
    parser = argparse.ArgumentParser(description="Ask a multiple-choice question about a badminton video clip.")
    parser.add_argument("checkpoint")
    parser.add_argument("--video", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--options", nargs="+", required=True, help="Candidate options, each as one shell argument.")
    parser.add_argument("--start", type=float, default=None, help="Clip start time in seconds.")
    parser.add_argument("--end", type=float, default=None, help="Clip end time in seconds.")
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--task-type", choices=["hit_type", "subtype", "hitter", "ball_area"], default="hit_type")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    result = predict_video(args.checkpoint, args.video, args.question, args.options,
                           args.start, args.end, args.frames, args.device, args.task_type)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
