from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import wat
import json
import torch
from dataclasses import dataclass
from repeng import ControlModel, ControlVector, DatasetEntry

def chat_template_unparse(messages: list[tuple[str, str]]) -> str:
    # Convert chat template (role, content) into a string
    template = []
    for role, content in messages:
        template.append(
            f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
        )
    if messages[-1][0] != "assistant":
        # prefill assistant prefix
        template.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    return "".join(template)


def chat_template_parse(resp: str) -> list[tuple[str, str]]:
    # Parse chat template response into list of (role, content) tuples
    resp = resp.strip().removeprefix("<|begin_of_text|>")
    messages = []
    for part in resp.split("<|start_header_id|>"):
        role_and_content = part.split("<|end_header_id|>")
        if len(role_and_content) == 1:
            role, content = role_and_content[0], ""
        else:
            role, content = role_and_content
        content = content.split("<|eot_id|>")[0]
        messages.append((role.strip(), content.strip()))
    return messages

def generate_dataset(prompts, tokenizer:AutoTokenizer, dataset_path="data/truncated_vision.json"):
    with open(dataset_path) as f:
        output_suffixes = json.load(f)
        truncated_output_suffixes = [
            tokenizer.convert_tokens_to_string(tokens[:i])
            for tokens in (tokenizer.tokenize(s) for s in output_suffixes)
            for i in range(1, len(tokens))
        ]
    dataset = make_dataset(
        chat_template_unparse([("user", "{}{persona}")]),
        prompts.pos,
        prompts.neg,
        truncated_output_suffixes,
    )
    return dataset

# "role": "user",
# "content": [
#     {"type": "image", "image": image_path },
#     {"type": "text", "text": "What do you see"},
# ],

def make_dataset(template: str, positive_personas: list[str],
    negative_personas: list[str], suffix_list: list[str],):
    dataset = []
    for suffix in suffix_list:
        for positive_persona, negative_persona in zip(positive_personas, negative_personas):
            positive_template = template.format(persona=positive_persona)
            negative_template = template.format(persona=negative_persona)
            dataset.append(
                DatasetEntry(
                    positive=f"{positive_template}{suffix}",
                    negative=f"{negative_template}{suffix}",
                )
            )
    return dataset

device = "mps" if torch.backends.mps.is_available() else "cuda"
print("Using device:", device)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")

# # Print model layers
# print("\nModel layers:")
# for name, _ in model.named_modules():
#     print(name)

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

model = ControlModel(model, layer_ids=list(range(5, 22)))

model.reset()

@dataclass
class Prompts:
    pos: list[str]
    neg: list[str]

prompts = Prompts(
    pos=["Pretend to be a very friendly person"],
    neg=["Pretend to be a very mean person"],
)

dataset = generate_dataset(prompts, tokenizer)

vector = ControlVector.train(
    model, tokenizer, dataset, 
    batch_size=32, method='pca_center'
)

vector.export_gguf("vectors/qwenv_cvec.gguf")


exit()

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "/Users/isaac/Desktop/IMG_2752.JPG",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
print("processing vision info")
image_inputs, video_inputs = process_vision_info(messages)
print("processing inputs")
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to(device)
print("generating ids")
# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)