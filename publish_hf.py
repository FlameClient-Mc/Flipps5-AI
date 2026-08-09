"""Upload the trained Flipps V0.1 LoRA adapter to Hugging Face so the public can use it.

Usage:
    1. Create a free account at https://huggingface.co and a token with Write
       access at https://huggingface.co/settings/tokens
    2. Set the token, then run:
         set HF_TOKEN=hf_xxxxxxxxxxxxxxxx     (Command Prompt)
         $env:HF_TOKEN="hf_xxxxxxxxxxxxxxxx"  (PowerShell)
         python publish_hf.py
    3. The adapter appears at https://huggingface.co/FlameFlipps/flipps-v0.1-lora

    Optional:  set HF_REPO=YourName/flipps-v0.1-lora   to publish under your account.
"""
import os

from huggingface_hub import HfApi

REPO = os.environ.get("HF_REPO", "FlameFlipps/flipps-v0.1-lora")
FOLDER = "flipps-v0.1-lora"


def main():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("No HF token found. Create one at https://huggingface.co/settings/tokens,")
        print("then set it:  set HF_TOKEN=hf_xxxxxxxx  (Command Prompt)")
        print("or           $env:HF_TOKEN=\"hf_xxxxxxxx\" (PowerShell)")
        return 1
    if not os.path.isdir(FOLDER):
        print(f"Folder '{FOLDER}' not found — train the adapter first: python train.py --data data/train.jsonl")
        return 1
    api = HfApi(token=token)
    api.create_repo(repo_id=REPO, repo_type="model", exist_ok=True, private=False)
    print(f"Uploading {FOLDER} -> {REPO} (public)...")
    api.upload_folder(folder_path=FOLDER, repo_id=REPO, repo_type="model")
    print(f"Done! The public can now load it: https://huggingface.co/{REPO}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
