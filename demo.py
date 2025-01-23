import argparse
import random
import sys

from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
import torch

parser = argparse.ArgumentParser()
parser.add_argument("question", type=str)
parser.add_argument(
    "-m", "--model-name", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
)
parser.add_argument(
    "-r", "--replacements", nargs="+", default=["\nWait, but", "\nHmm", "\nSo"]
)
parser.add_argument("-t", "--min-thinking-tokens", type=int, default=128)
parser.add_argument("-p", "--prefill", default="")
args = parser.parse_args()

print(f"checking device")
device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
print(f"Using tokenizer: {tokenizer}")
model = AutoModelForCausalLM.from_pretrained(
    args.model_name, torch_dtype=torch.bfloat16, device_map=device
)
print(f"Using model: {model}")

_, _start_think_token, end_think_token = tokenizer.encode("<think></think>")


@torch.inference_mode
def reasoning_effort(question: str, min_thinking_tokens: int):
    tokens = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": "<think>\n" + args.prefill},
        ],
        continue_final_message=True,
        return_tensors="pt",
    )
    tokens = tokens.to(model.device)
    kv = DynamicCache()
    n_thinking_tokens = 0

    yield tokenizer.decode(list(tokens[0]))
    while True:
        out = model(input_ids=tokens, past_key_values=kv, use_cache=True)
        next_token = torch.multinomial(
            torch.softmax(out.logits[0, -1, :], dim=-1), 1
        ).item()
        kv = out.past_key_values

        if (
            next_token in (end_think_token, model.config.eos_token_id)
            and n_thinking_tokens < min_thinking_tokens
        ):
            replacement = random.choice(args.replacements)
            yield replacement
            replacement_tokens = tokenizer.encode(replacement)
            n_thinking_tokens += len(replacement_tokens)
            tokens = torch.tensor([replacement_tokens]).to(tokens.device)
        elif next_token == model.config.eos_token_id:
            break
        else:
            yield tokenizer.decode([next_token])
            n_thinking_tokens += 1
            tokens = torch.tensor([[next_token]]).to(tokens.device)


for chunk in reasoning_effort(args.question, args.min_thinking_tokens):
    print(chunk, end="", flush=True)