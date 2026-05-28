"""Pre-download IndexTTS-2 model weights to a cached Docker layer."""
from huggingface_hub import snapshot_download

# Download IndexTTS-2 model checkpoints
MODEL_ID = "IndexTeam/IndexTTS-2"
print(f"Prefetching {MODEL_ID} from HuggingFace...")

try:
    # Download model files to cache
    snapshot_download(
        repo_id=MODEL_ID,
        cache_dir="/tmp/huggingface_cache",
        local_files_only=False
    )
    print("IndexTTS-2 model prefetch complete")
except Exception as e:
    print(f"Warning: Failed to prefetch model: {e}")
    print("Will download on first run")
