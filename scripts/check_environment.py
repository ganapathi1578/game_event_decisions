import importlib.util
import platform
import sys

print("Python:", sys.version)
print("Platform:", platform.platform())
for package in ("torch", "transformers", "peft", "accelerate", "cv2", "PIL", "huggingface_hub"):
    spec = importlib.util.find_spec(package)
    print(f"{package}: {'installed' if spec else 'MISSING'}")
if importlib.util.find_spec("torch"):
    import torch
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {p.name}, {p.total_memory / 1024**3:.1f} GiB")
