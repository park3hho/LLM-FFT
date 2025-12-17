###################
### Model Import ##
###################

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "kakaocorp/kanana-nano-2.1b-base"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    trust_remote_code=True,
).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token # <|end_of_text|> 128001

###################
#### DATA SET #####
###################

print("")
print("##################################")
print("####         DATA SET        #####")
print("##################################")

from pathlib import Path

qna_list = []

text = Path("../data/dataset.txt").read_text(encoding="utf-8")

blocks = text.split("<|question|>")

print(repr(text[:50]))
print(text.count("<|question|>"), text.count("<|answer|>"))

for block in blocks:
    block = block.strip()
    if not block:
        continue

    if "<|answer|>" not in block:
        continue

    q_part, a_part = block.split("<|answer|>", 1)  # ⭐ 핵심

    question = q_part.strip()
    answer = a_part.strip()

    input_str = (
        "<|question|>\n"
        + question
        + "\n<|answer|>\n"
        + answer
    )

    item = {
        "q": question,
        "a": answer,
        "input": input_str,
        "q_ids": tokenizer.encode(
            "<|question|>\n" + question + "\n<|answer|>\n",
            add_special_tokens=False,
        ),
        "input_ids": tokenizer.encode(
            input_str,
            add_special_tokens=False,
        ),
    }

    qna_list.append(item)

print("QNA count:", len(qna_list))
max_length = max(len(item["input_ids"]) for item in qna_list)

##################################
#### Check Answer before FFT #####
##################################

print("")
print("##################################")
print("#### Check Answer before FFT #####")
print("##################################")

questions = [ qna['q'] for qna in qna_list]
questions.append("너에 대해서 설명해봐.")
questions.append("파이썬 리스트와 튜플의 차이는 뭐야? ")
questions.append("파이썬에서 리스트와 튜플의 차이는 무엇인가요?")
questions.append("방구나 먹어라 마 ㅋㅋ")
questions.append("뭐하냐? ㅋㅋ")

input_ids = tokenizer(
    questions,
    padding=True,
    return_tensors="pt",
)["input_ids"].to("cuda")

# print(type(model))

model.eval()
with torch.no_grad():
    output = model.generate(
        input_ids,
        max_new_tokens=32,
        do_sample=False,
    )

output_list = output.tolist()

print("")

for i, output in enumerate(output_list):
    decoded = tokenizer.decode(output, skip_special_tokens=True)
    question = questions[i]

    answer = decoded[len(question):].strip()

    print(f"Q{i}", question)
    print("A:", answer)
