import os
from huggingface_hub import snapshot_download
WORKDIR = os.environ["WORKDIR"]
SAVED_DIR=f"{WORKDIR}/hf_checkpoint"

# Download HF checkpoints
snapshot_download(repo_id="mistralai/Mixtral-8x22B-v0.1", ignore_patterns=["*.pt"], local_dir=SAVED_DIR, local_dir_use_symlinks=False)