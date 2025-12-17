# llm_fft
1. DataSet
2. BaseModel 
3. Q/A Prompt Construction & Tokenization (Instruction Dataset Preprocessing)
4. Check Logic Answer before FFT


## a. ref 

## Dataset
Let LLM make.

(1) 정규식 사용 (ex " | ")  
(2) JSON  
(3) 커스텀 토큰  

## BaseModel 
```Model Import 
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
```

## Q/A Prompt Processing
```Q/A Prompt Processing 
#########################
# Q/A Prompt Processing #
#########################

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
```

## Check Logic Answer before FFT


---
### Strengthen 
### (1) base, instruct Model.
https://github.com/HongLabInc/HongLabLLM/blob/main/02_fullfinetuning1_base.ipynb