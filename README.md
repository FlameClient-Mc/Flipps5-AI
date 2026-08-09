# 🔥 Flipps V0.1 AI

**Flipps V0.1** — a personal AI assistant created by and owned by **FlameFlipps**.

Flipps V0.1 runs **completely on your own machine** (no cloud, no API fees, no data leaves your PC). It's built on a small coding-specialized open-source model (`Qwen2.5-Coder-0.5B-Instruct`) that knows **all programming languages**, and it can be **fine-tuned on your own data** with the included training script.

## What's in here

| File | What it does |
|---|---|
| `app.py` | Chat with Flipps V0.1 (CLI) |
| `train.py` | Fine-tune Flipps V0.1 on your own Q&A data (LoRA, CPU-friendly) |
| `persona.py` | Flipps V0.1's personality — **edit this to change who it is** |
| `data/train.jsonl` | Sample training data (replace with your own) |
| `requirements.txt` | Python dependencies |

## Quick start (chat)

1. Install Python dependencies (in a virtual environment):
   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt
   ```
   > The `--index-url .../whl/cpu` line installs the small CPU-only version of PyTorch. This machine has no NVIDIA GPU, so skip this and it'll download a huge CUDA build for nothing.

2. Start chatting:
   ```bash
   python app.py
   ```
   The first run downloads the base model (~1 GB) automatically, then you're talking to Flipps V0.1. Type `exit` to quit.

## Train Flipps V0.1 on your data

1. **Add your data** to `data/train.jsonl` — one JSON object per line. The simplest format:
   ```json
   {"instruction": "Who are you?", "response": "I'm Flipps V0.1, your personal AI assistant!"}
   ```
   Or use full chat format (recommended, lets you include the persona per example):
   ```json
   {"messages": [
     {"role": "system", "content": "You are Flipps V0.1, an AI assistant created by FlameFlipps."},
     {"role": "user", "content": "Who are you?"},
     {"role": "assistant", "content": "I'm Flipps V0.1!"}
   ]}
   ```
   Even **50–100 good Q&A pairs** about your topic is enough to start. Replace the sample file with your own.

2. **Train**:
   ```bash
   python train.py --data data/train.jsonl
   ```

3. **Chat with your trained version**:
   ```bash
   python app.py --adapter flipps-v0.1-lora
   ```
   The app auto-detects `flipps-v0.1-lora/`, so plain `python app.py` also works once you've trained.

### Useful training options
```bash
python train.py --epochs 5              # train longer (stronger, slower)
python train.py --rank 16               # bigger LoRA (more capacity to learn)
python train.py --merge                 # also save a standalone merged model
```

## Make it yours

- **Personality** — edit `persona.py`. That's the voice Flipps V0.1 uses in chat *and* for training examples without a system message.
- **Bigger/smarter** — swap the base model: `python app.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct` (much stronger coding, needs more RAM/CPU — best trained in a Codespace) or `python train.py --model <model-name>`.

## How it works (the honest version)

This is a **fine-tuned open-source model**, which is how individuals realistically build "their own AI" — training a model from scratch needs data-center GPUs. What's *yours*: the fine-tune (your data, your knowledge baked in), the Flipps V0.1 persona, the app, and the repo. The model weights themselves remain under their original open license.

## Hardware notes

Runs on this machine: 4-core CPU, 16 GB RAM, no GPU. Generation speed is ~5–20 tokens/second — slower than ChatGPT but fully private and free. Fine-tuning is slow on CPU (a few hours for a small dataset) — for faster training, run `train.py` in a GitHub Codespace instead.
