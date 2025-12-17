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

## Q/A Prompt Construction
```

```

## Check Logic Answer before FFT


---
### Strengthen 
### (1) base, instruct Model.
https://github.com/HongLabInc/HongLabLLM/blob/main/02_fullfinetuning1_base.ipynb