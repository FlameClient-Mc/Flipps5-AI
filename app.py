"""FlameFlipps — local chat assistant.

Usage:
    python app.py                              # chat with the base model
    python app.py --adapter flameflipps-lora   # chat with your fine-tuned model
"""
import argparse
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import tools
from persona import SYSTEM_PROMPT

DEFAULT_MODEL = os.environ.get("FLAMEFLIPPS_MODEL", "Qwen/Qwen2.5-Coder-0.5B-Instruct")
ADAPTER_DIR = "flipps-v0.1-lora"
MAX_HISTORY_TURNS = 10


def load_model(model_name, adapter_dir):
    print(f"[Flipps V0.1] Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    if adapter_dir and os.path.isdir(adapter_dir):
        from peft import PeftModel

        print(f"[Flipps V0.1] Loading fine-tuned adapter: {adapter_dir}")
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return tokenizer, model


def run_tool(user_input):
    """Dispatch tool commands. Returns (label, result_text) or None for normal chat."""
    s = user_input.strip()
    lower = s.lower()
    if lower.startswith("!help") or lower.startswith("help:") or s == "help":
        return ("help", tools.TOOL_HELP)
    if lower.startswith("search:") or lower.startswith("search ") or lower.startswith("!search"):
        q = s.split(":", 1)[1].strip() if ":" in s else s.split(None, 1)[1].strip()
        return ("search", str(tools.web_search(q)))
    if lower.startswith("research:") or lower.startswith("!research"):
        q = s.split(":", 1)[1].strip() if ":" in s else s.split(None, 1)[1].strip()
        return ("research", tools.research(q))
    if lower.startswith("youtube:") or lower.startswith("!youtube"):
        q = s.split(":", 1)[1].strip() if ":" in s else s.split(None, 1)[1].strip()
        return ("youtube", str(tools.youtube_search(q)))
    if lower.startswith("github:") or lower.startswith("!github"):
        q = s.split(":", 1)[1].strip() if ":" in s else s.split(None, 1)[1].strip()
        return ("github", str(tools.github_search(q)))
    if lower.startswith("repo:") or lower.startswith("!repo"):
        q = s.split(":", 1)[1].strip() if ":" in s else s.split(None, 1)[1].strip()
        return ("repo", tools.github_repo(q))
    if lower.startswith("fetch:") or lower.startswith("read:") or lower.startswith("!fetch"):
        q = s.split(":", 1)[1].strip() if ":" in s else s.split(None, 1)[1].strip()
        return ("fetch", tools.read_url(q))
    if lower.startswith("telegram:") or lower.startswith("!telegram"):
        rest = s.split(":", 1)[1].strip() if ":" in s else s.split(None, 1)[1].strip()
        parts = rest.split(None, 1)
        if len(parts) < 2:
            return ("telegram", "Usage: telegram: <chat_id> <message>")
        return ("telegram", tools.telegram_send(parts[0], parts[1]))
    if lower.startswith("twitter:") or lower.startswith("!twitter"):
        q = s.split(":", 1)[1].strip() if ":" in s else s.split(None, 1)[1].strip()
        return ("twitter", str(tools.social_search("twitter", q)))
    if lower.startswith("instagram:") or lower.startswith("!instagram"):
        q = s.split(":", 1)[1].strip() if ":" in s else s.split(None, 1)[1].strip()
        return ("instagram", str(tools.social_search("instagram", q)))
    return None


def chat_loop(tokenizer, model, max_tokens, temperature):
    print()
    print("  Flipps V0.1 is ready. Type 'exit' or 'quit' to leave.")
    print("  " + "-" * 60)
    history = []
    while True:
        try:
            user_input = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break
        tool = run_tool(user_input)
        if tool:
            label, result = tool
            print(f"[tool] {label}: done")
            history.append({"role": "user", "content": user_input})
            history.append({"role": "system", "content":
                f"Tool results from '{label}':\n{result}\n\nUse these results to answer the user's request concisely."})
        else:
            history.append({"role": "user", "content": user_input})
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history[-MAX_HISTORY_TURNS:]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        reply = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        history.append({"role": "assistant", "content": reply})
        print(f"Flipps V0.1> {reply}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Flipps V0.1 — your local AI assistant")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Base model name on Hugging Face")
    parser.add_argument("--adapter", default=ADAPTER_DIR, help="Path to a fine-tuned LoRA adapter (auto-detected)")
    parser.add_argument("--no-adapter", action="store_true", help="Ignore any fine-tuned adapter")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    adapter = None if args.no_adapter else args.adapter
    tokenizer, model = load_model(args.model, adapter)
    chat_loop(tokenizer, model, args.max_tokens, args.temperature)
    return 0


if __name__ == "__main__":
    sys.exit(main())
