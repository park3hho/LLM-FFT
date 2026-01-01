# Base Model

## Dataset
Let LLM make Datasets.

Structure of Datasets
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
```Check Answer before FFT
##################################
#### Check Answer before FFT #####
##################################

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
```

Results
```commandline
##################################
#### Check Answer before FFT #####
##################################
The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.

Q0 파이썬에서 리스트와 튜플의 차이는 무엇인가요?
A: 리스트와 튜플은 파이썬에서 사용되는 두 가지 데이터 구조입니다. 리스트는 순서가 있는 데이터의 모음이며,
Q1 REST API란 무엇인가요?
A: 1. REST API란? REST API는 Representational State Transfer의 약자로, 웹 서비스를 위한 아키텍처 스타
Q2 파이썬에서 가상환경을 사용하는 이유는 무엇인가요?
A: 가상환경은 파이썬에서 프로젝트를 독립적으로 관리하고, 다른 프로젝트와의 충돌을 방지하기 위해 사용
Q3 HTTP 상태 코드 404는 무엇을 의미하나요?
A: 404는 "Not Found"의 약자로, 요청한 리소스가 서버에 존재하지 않음을 나타냅니다. 이는 주로 웹
Q4 클래스와 객체의 차이는 무엇인가요?
A: 클래스는 객체를 정의하는 틀입니다. 객체는 클래스를 이용해서 만들어진 인스턴스입니다. 클래스는 객체를 생성하기 위한
Q5 파이썬에서 예외 처리는 어떻게 하나요?
A: 예외 처리는 파이썬에서 프로그램의 실행 중 발생할 수 있는 오류를 처리하는 방법입니다. 예외 처리는 try-ex
Q6 JWT란 무엇인가요?
A: 1. 1. 2. 3. 4. 5. 6. 7. 8. 9. 10
Q7 데이터베이스에서 인덱스의 역할은 무엇인가요?
A: 인덱스는 데이터베이스에서 검색 속도를 향상시키기 위해 사용되는 자료구조입니다. 인덱스를 사용하면 데이터를 빠
Q8 파이썬에서 딕셔너리를 사용하는 이유는 무엇인가요?
A: 딕셔너리는 키와 값의 쌍으로 이루어진 데이터 구조입니다. 파이썬에서 딕셔너리를 사용하는
Q9 OSI 7계층 중 전송 계층의 역할은 무엇인가요?
A: OSI 7계층 중 전송 계층은 데이터를 목적지까지 전달하는 역할을 합니다. 전송 계층은 데이터를
Q10 Flask와 Django의 차이점은 무엇인가요?
A: Flask는 Python을 위한 마이크로 프레임워크로, 간단하고 빠른 웹 애플리케이션을 개발하기 위한 목
Q11 파이썬에서 리스트 컴프리헨션이란 무엇인가요?
A: 리스트 컴프리헨션은 파이썬에서 리스트를 생성하는 데 사용되는 간결한 방법입니다. 리스트 컴프리�
Q12 CORS 에러는 왜 발생하나요?
A: CORS 에러는 웹 브라우저가 서버로부터 리소스를 요청할 때, 서버가 해당 리소스에 대한 접근을 허용하지
Q13 CPU와 GPU의 차이점은 무엇인가요?
A: CPU는 중앙처리장치로, 컴퓨터의 두뇌 역할을 합니다. CPU는 명령어를 해석하고 실행하는 역
Q14 파이썬에서 None은 어떤 의미인가요?
A: None은 파이썬에서 사용되는 특별한 값으로, 변수나 함수의 반환값으로 사용됩니다. None은 값이 없거나,
Q15 데이터 직렬화란 무엇인가요?
A: 1. 직렬화란? 직렬화는 객체를 데이터로 변환하는 과정입니다. 직렬화된 데이터는 파일, 네트
Q16 GET과 POST 요청의 차이는 무엇인가요?
A: GET은 데이터를 읽어오는 요청이고, POST는 데이터를 전송하는 요청입니다. GET은 주소창에 URL이 노출되지만,
Q17 파이썬에서 lambda 함수는 언제 사용하나요?
A: 파이썬에서 lambda 함수는 함수를 간단하게 정의하고 싶을 때 사용합니다. 예를 들어, 리스트의 요소를 제
Q18 캐시(Cache)의 목적은 무엇인가요?
A: 캐시(Cache)는 데이터를 저장하는 공간으로, 데이터를 빠르게 접근할 수 있도록 도와줍니다. 캐시는 데이터를 저장
Q19 트랜잭션이란 무엇인가요?
A: 트랜잭션은 데이터베이스의 상태를 변화시키기 위해 수행하는 작업의 단위입니다. 트랜잭션은 데이터베
Q20 너에 대해서 설명해봐.
A: 1. 2. 3. 4. 5. 6. 7. 8. 9. 10. 11
Q21 파이썬 리스트와 튜플의 차이는 뭐야? 
A: 파이썬에서 리스트와 튜플은 모두 데이터를 저장하는 데 사용되는 자료형이지만, 몇 가지 중요한 차이점이
Q22 파이썬에서 리스트와 튜플의 차이는 무엇인가요?
A: 리스트와 튜플은 파이썬에서 사용되는 두 가지 데이터 구조입니다. 리스트는 순서가 있는 데이터의 모음이며,
Q23 방구나 먹어라 마 ㅋㅋ
A: ㅋ
안녕하세요. 오늘은 5월 5일 어린이날입니다. 어린이날은 어린이의 인격을 존중하고,
Q24 뭐하냐? ㅋㅋ
A: ㅋ
안녕하세요. 오늘은 제가 좋아하는 과일인 바나나에 대해 알아보겠습니다. 바나나는 열대지방에서 자라는
```

## Question-masked SFT
```Question-masked SFT
import torch
from torch.utils.data import Dataset, DataLoader

EOT = 128001 # instruct 모델과 다름

class MyDataset(Dataset):
    def __init__(self, qna_list, max_length):
        self.input_ids = []
        self.target_ids = []

        for qa in qna_list:
            token_ids = qa['input_ids']
            input_chunk = token_ids
            target_chunk = token_ids[1:]
            input_chunk += [EOT]* (max_length - len(input_chunk))
            target_chunk +=  [EOT]* (max_length - len(target_chunk))
            len_ignore = len(qa['q_ids']) - 1 # target은 한 글자가 짧기 때문
            target_chunk[:len_ignore] = [-100] * len_ignore

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

dataset = MyDataset(qna_list, max_length=max_length)

train_loader = DataLoader(dataset, batch_size=2, shuffle=True, drop_last=False)

i = iter(train_loader)

x, y = next(i)

y_temp = y[0].tolist()
y_temp = [x for x in y_temp if x != -100] # -100은 제외하고 디코딩

print(tokenizer.decode(x[0].tolist()))
print(tokenizer.decode(y_temp))
```

Results
```
##################################
####     Q/A masking SFT     #####
##################################
<|question|>
클래스와 객체의 차이는 무엇인가요?
<|answer|>
클래스는 객체를 생성하기 위한 설계도이고, 객체는 클래스를 기반으로 생성된 실제 인스턴스입니다.<|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|>
클래스는 객체를 생성하기 위한 설계도이고, 객체는 클래스를 기반으로 생성된 실제 인스턴스입니다.<|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|><|end_of_text|>
```

## FFT Results
``` FFT Results
0 Tokens seen: 132
1 Tokens seen: 264
2 Tokens seen: 396
3 Tokens seen: 528
4 Tokens seen: 660
^@5 Tokens seen: 792
6 Tokens seen: 924
7 Tokens seen: 1056
8 Tokens seen: 1188
9 Tokens seen: 1320
Epoch: 0, Loss: 1.452734375
10 Tokens seen: 1452
11 Tokens seen: 1584
12 Tokens seen: 1716
13 Tokens seen: 1848
14 Tokens seen: 1980
15 Tokens seen: 2112
16 Tokens seen: 2244
17 Tokens seen: 2376
18 Tokens seen: 2508
19 Tokens seen: 2640
Epoch: 1, Loss: 0.1677734375
20 Tokens seen: 2772
21 Tokens seen: 2904
22 Tokens seen: 3036
23 Tokens seen: 3168
24 Tokens seen: 3300
25 Tokens seen: 3432
26 Tokens seen: 3564
27 Tokens seen: 3696
28 Tokens seen: 3828
29 Tokens seen: 3960
Epoch: 2, Loss: 0.0400390625
30 Tokens seen: 4092
31 Tokens seen: 4224
32 Tokens seen: 4356
33 Tokens seen: 4488
34 Tokens seen: 4620
35 Tokens seen: 4752
36 Tokens seen: 4884
37 Tokens seen: 5016
38 Tokens seen: 5148
39 Tokens seen: 5280
Epoch: 3, Loss: 0.0127197265625
40 Tokens seen: 5412
41 Tokens seen: 5544
42 Tokens seen: 5676
43 Tokens seen: 5808
44 Tokens seen: 5940
45 Tokens seen: 6072
46 Tokens seen: 6204
47 Tokens seen: 6336
48 Tokens seen: 6468
49 Tokens seen: 6600
Epoch: 4, Loss: 0.00618743896484375
50 Tokens seen: 6732
51 Tokens seen: 6864
52 Tokens seen: 6996
53 Tokens seen: 7128
54 Tokens seen: 7260
55 Tokens seen: 7392
56 Tokens seen: 7524
57 Tokens seen: 7656
58 Tokens seen: 7788
59 Tokens seen: 7920
Epoch: 5, Loss: 0.00465240478515625
60 Tokens seen: 8052
61 Tokens seen: 8184
62 Tokens seen: 8316
63 Tokens seen: 8448
64 Tokens seen: 8580
65 Tokens seen: 8712
66 Tokens seen: 8844
67 Tokens seen: 8976
68 Tokens seen: 9108
69 Tokens seen: 9240
Epoch: 6, Loss: 0.00398406982421875
70 Tokens seen: 9372
71 Tokens seen: 9504
72 Tokens seen: 9636
73 Tokens seen: 9768
74 Tokens seen: 9900
75 Tokens seen: 10032
76 Tokens seen: 10164
77 Tokens seen: 10296
78 Tokens seen: 10428
79 Tokens seen: 10560
Epoch: 7, Loss: 0.003619384765625
80 Tokens seen: 10692
81 Tokens seen: 10824
82 Tokens seen: 10956
83 Tokens seen: 11088
84 Tokens seen: 11220
85 Tokens seen: 11352
86 Tokens seen: 11484
87 Tokens seen: 11616
88 Tokens seen: 11748
89 Tokens seen: 11880
Epoch: 8, Loss: 0.003287506103515625
90 Tokens seen: 12012
91 Tokens seen: 12144
92 Tokens seen: 12276
93 Tokens seen: 12408
94 Tokens seen: 12540
95 Tokens seen: 12672
96 Tokens seen: 12804
97 Tokens seen: 12936
98 Tokens seen: 13068
99 Tokens seen: 13200
Epoch: 9, Loss: 0.0030487060546875
```

---

# Catastrophic Forgetting

목적: BaseModel에 Catstrophic Forgetting 유도하기
