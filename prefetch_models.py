"""Pre-download IndexTTS-2 model weights to /app/checkpoints so rp_handler can load them."""
import os
from huggingface_hub import snapshot_download

MODEL_ID = "IndexTeam/IndexTTS-2"
LOCAL_DIR = "/app/checkpoints"

os.makedirs(LOCAL_DIR, exist_ok=True)
print(f"Prefetching {MODEL_ID} into {LOCAL_DIR}...")

snapshot_download(
    repo_id=MODEL_ID,
    local_dir=LOCAL_DIR,
)
print("IndexTTS-2 model prefetch complete")
