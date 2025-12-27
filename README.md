# llm_fft
1. DataSet
2. BaseModel   
   1. Q/A Prompt Construction & Tokenization (Instruction Dataset Preprocessing)
   2. Check Logic Answer before FFT
3. InstructModel
   1. Q/A Prompt Construction & Tokenization (Instruction Dataset Preprocessing)
   2. Check Logic Answer before FFT

## Feedback.
Instruct Model이 Base Model보다 훨씬 훈련이 잘됨.

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

## Review Hardware
25.12.23
이전에 pretrain 할 때는 하지 않았던 트레인 도중 하드웨어가 어떻게 동작하고 있는지 살펴보았다.
현재 토큰을 임베딩하고 체크할 때는 GPU와 메모리가 맛있게 돌아간다.
GPU는 100퍼센트, 메모리는 83퍼센트까지 차지하면서 돌아간다.

RTX 5060Ti가 대역폭이 개박살나서 별로라고 했는데(본디 성능에 비해서), 일단 돌아가는 상황에서
메모리와 GPU의 비율은 맛있게 잘 돌아가는 것 같다.
GPU - Memory의 대역폭이 막혀버려 메모리랑 GPU 모두 최대치로 사용하는거 일수도 있겠다만,
그 상황이라면 Memory가 80퍼센트에서 노는게 아니라 98퍼센트 이런식으로 먹어야 하지 않을까?
메모리 설정이 어떻게 되어있는지 잘 몰라서 어떨진 모르겠다만 아무튼, 일단 겉만 보자면 아직 
큰 문제는 없어보인다. 음, 나중에 Hardware 단계에서 어떻게 돌아가는 지 더 공부할 거라 지금은
아직 시기상조라 생각이 들긴해. AI 트레이닝 모델들은 메모리가 못따라와서 프로세서 성능이 말리는
거로 알고 있는데 GPU 빵빵하게 돌아가면서 훈련되는거면 긍정적으로 판단은 된다.

Pretrain 때는 하드웨어 지식이 부족하기도 했고, 회사에서 배우면서 흥미를 쌓다보니 fft 돌릴 때
갑자기 봐야겠단 아이디어가 떠올라서,,,

> 관찰하다 보니 GPU가 노는 구간이 있었다. 그 구간에 대해 GPT한테 역전파 때문이냐는 질문하니, 
> 역전파는 오히려 GPU가 하드워킹 개빡세게 하는 구간이라카고, (1) Dataload, (2) Validation loop,
> (3) Checkpoint/logging 때문이라고 한다. 그리고 그땐 CPU 연산을 많이 쓴다는데 관찰해보니 CPU Spike가 
> 몇번씩 터지긴 하더라.

## Results of Base Model
Loss가 3일 때 가장 대답이 좋다. (Epoch 0일 때, 1 이하로 내려가면 바로 이상해짐)
- 모델이 작아서 Loss가 낮으면 낮을수록 고장나버림 기본적인 성능을 못내버림.

**Loss 3일 때,**
```
#############################
##    Loss = 3 | Answer    ##
#############################

Q0: 파이썬에서 리스트와 튜플의 차이는 무엇인가요? 리스트는 가변 데이터 타입으로, 요소의 추가, 삭제, 수 정이 가능합니다. 반면 튜플은 불변 데이터 타입
Q1: REST API란 무엇인가요? REST API는 웹 서비스를 위한 아키텍처 스타일로, HTTP 프로토콜을 기반으로 자원을 URI로 표현하고
Q2: 파이썬에서 가상환경을 사용하는 이유는 무엇인가요? 가상환경을 사용하는 이유는 프로젝트 간의 의존성  충돌을 방지하고, 각 프로젝트마다 독립적인 환경을 유지
Q3: HTTP 상태 코드 404는 무엇을 의미하나요? 404 상태 코드는 서버에서 요청한 리소스가 존재하지 않을 때  반환되는 응답 코드입니다.
Q4: 클래스와 객체의 차이는 무엇인가요? 클래스는 객체를 생성하기 위한 설계도이고, 객체는 클래스를 기반으로 생성된 인스턴스입니다.
Q5: 파이썬에서 예외 처리는 어떻게 하나요?
Q6: JWT란 무엇인가요? JWT는 JSON Web Token의 약자로, 사용자 인증 정보를 JSON 형태로 안전하게 전달하기  위한 토큰 기반 인증 방
Q7: 데이터베이스에서 인덱스의 역할은 무엇인가요? 인덱스는 데이터 검색 속도를 향상시키기 위한 자료구조로, 데이터의 위치를 빠르게 찾을 수 있도록 도와줍
Q8: 파이썬에서 딕셔너리를 사용하는 이유는 무엇인가요? 딕셔너리는 키-값 쌍으로 데이터를 저장하는 자료형 으로, 빠른 검색과 수정이 가능합니다.
Q9: OSI 7계층 중 전송 계층의 역할은 무엇인가요? 1. OSI 7계층 중 전송 계층의 역할은 무엇인가요? 2. TCP와 UDP의 차이점은 무엇
Q10: Flask와 Django의 차이점은 무엇인가요? Flask는 경량 웹 프레임워크로, 간단한 웹 애플리케이션을 빠르 게 개발할 수 있습니다. 반면
Q11: 파이썬에서 리스트 컴프리헨션이란 무엇인가요? 리스트 컴프리헨션은 반복문과 조건문을 한 줄로 표현할 수 있는 파이썬의 문법입니다.
Q12: CORS 에러는 왜 발생하나요? CORS 에러는 웹 브라우저가 서버로부터 리소스를 요청할 때, 서버가 해당 요청을 허용하지 않을 때 발생합니다. 이는
Q13: CPU와 GPU의 차이점은 무엇인가요? CPU는 중앙 처리 장치로, 주로 연산과 제어를 담당합니다. 반면에 GPU는 그래픽 처리 장치로,
Q14: 파이썬에서 None은 어떤 의미인가요? None은 파이썬에서 사용되는 특별한 값으로, 변수가 아직 초기화되 지 않았거나, 값이 없을 때 사용됩니다.
Q15: 데이터 직렬화란 무엇인가요? 데이터 직렬화는 객체를 저장하거나 전송할 수 있는 형태로 변환하는 과정 입니다. 이를 통해 객체의 상태를 일관
Q16: GET과 POST 요청의 차이는 무엇인가요? GET은 데이터를 조회할 때 사용하고, POST는 데이터를 생성할 때 사용합니다.
Q17: 파이썬에서 lambda 함수는 언제 사용하나요?
Q18: 캐시(Cache)의 목적은 무엇인가요? 캐시(Cache)는 자주 사용되는 데이터를 임시로 저장하여 빠른 접근을 가능하게 하는 메모리 영역입니다. 주
Q19: 트랜잭션이란 무엇인가요? 트랜잭션은 데이터베이스에서 하나의 논리적 작업 단위로, 모두 성공하거나 모두 실패해야 하는 작업 단위
Q20: 파이썬에서 가상환경을 사용하는 이유는 무엇인가요? 가상환경을 사용하는 이유는 프로젝트 간의 의존성 충돌을 방지하고, 각 프로젝트마다 독립적인 환경을 유지
Q21: 방구나 먹으시고 ㅋㅋ;
Q22: 박찬호는 은퇴 후 어떤 활동을 하고 있나요? 박찬호는 은퇴 후에도 다양한 활동을 하고 있습니다. 그는  미국에서 투수 코치로 활동하며 후배 선수들을 지도하고 있습니다.
Q23: 파이썬에서 리스트와 튜플의 차이는 무엇인가요? 리스트는 가변 데이터 타입으로, 요소의 추가, 삭제, 수정이 가능합니다. 반면 튜플은 불변 데이터 타입
Q24: 박찬호는 왜 야구를 시작했나요? 박찬호는 어릴 때부터 야구를 좋아했습니다. 그는 야구를 통해 자신의  꿈을 이루고 싶었고, 야구를 통해
```

**Loss 1 미만일 때**
```yaml
#############################
##    Loss < 1 | Answer    ##
#############################

Q0: 파이썬에서 리스트와 튜플의 차이는 무엇인가요?
Q1: REST API란 무엇인가요? REST API는 HTTP 프로토콜을 기반으로 자원을 URI로 표현하고, HTTP 메서드(GET, POST, PUT, DELETE)를 통해 자원을
Q2: 파이썬에서 가상환경을 사용하는 이유는 무엇인가요?
Q3: HTTP 상태 코드 404는 무엇을 의미하나요?
Q4: 클래스와 객체의 차이는 무엇인가요? 클래스는 객체를 생성하기 위한 설계도이고, 객체는 클래스를 기반으로 생성된 실제 인스턴스입니다.
Q5: 파이썬에서 예외 처리는 어떻게 하나요?
Q6: JWT란 무엇인가요? JWT는 JSON Web Token의 약자로, 사용자 인증 정보를 JSON 형태로 안전하게 전달하기  위한 토큰 기반 인증 방
Q7: 데이터베이스에서 인덱스의 역할은 무엇인가요? 인덱스는 데이터 검색 속도를 향상시키기 위한 자료구조로, 조회 성능을 높이는 대신 저장 공간과 쓰기 성
Q8: 파이썬에서 딕셔너리를 사용하는 이유는 무엇인가요?
Q9: OSI 7계층 중 전송 계층의 역할은 무엇인가요? 전송 계층은 데이터의 신뢰성 있는 전달을 담당하며, TCP와 UDP 프로토콜이 이 계층에 속합니다.
Q10: Flask와 Django의 차이점은 무엇인가요? Flask는 경량 웹 프레임워크로, 유연성이 높고 확장성이 뛰어나 지만, 초기 설정이 복잡할 수 있습니다.
Q11: 파이썬에서 리스트 컴프리헨션이란 무엇인가요?
Q12: CORS 에러는 왜 발생하나요?
Q13: CPU와 GPU의 차이점은 무엇인가요? CPU는 범용 연산에 최적화되어 있고, GPU는 대규모 병렬 연산에 특화 되어 있습니다.
Q14: 파이썬에서 None은 어떤 의미인가요?
Q15: 데이터 직렬화란 무엇인가요? 데이터 직렬화는 객체를 저장하거나 전송할 수 있는 형태로 변환하는 과정 입니다.
Q16: GET과 POST 요청의 차이는 무엇인가요? GET은 데이터를 조회할 때 사용하고, POST는 데이터를 생성하거나 서버 상태를 변경할 때 사용합니다.
Q17: 파이썬에서 lambda 함수는 언제 사용하나요?
Q18: 캐시(Cache)의 목적은 무엇인가요?
Q19: 트랜잭션이란 무엇인가요? 트랜잭션은 데이터베이스에서 하나의 논리적 작업 단위로, 모두 성공하거나 모두 실패해야 합니다.
Q20: 파이썬에서 가상환경을 사용하는 이유는 무엇인가요?
Q21: 방구나 먹으시고 ㅋㅋ;
Q22: 박찬호는 은퇴 후 어떤 활동을 하고 있나요?
Q23: 파이썬에서 리스트와 튜플의 차이는 무엇인가요?
Q24: 박찬호는 왜 야구를 시작했나요?
```

> Loss 조절 방식
> 1. Learning Rate(Optimizer)
> 2. Batch Size()

1epoch에 걸린 시간
2epoch에 약 3분 걸린다. 10epoch 돌리니 15분 예상 됨.

![Training Loss](/main/base/epoch_loss.png)

Interpretation by ChatGPT
1. Convergence Speed
- Epoch 0 -> 1 Loss `1.45 -> 0.16`
  - 데이터가 너무 작거나, 패턴이 단순함. 강한 오버피팅 신호.
- Epoch 2(0.04)부터 사실상 학습 완료 수준.

2. After Epoch 3
- `0.012 -> 0.003`으로 아주 미세한 감소
- 의미있는 학습은 없음.
- 사실상 3~4 Epoch이면 충분

***결론***
> 즉, 모델이 데이터를 거의 외워버렸다. 학습은 성공했지만, 일반화는 모른다.

```FFT Question Results
REST API는 무엇인가요
Q24: ? REST API는 HTTP 프로토콜을 기반으로 자원을 URI로 표현하고
GET, POST, PUT, DELETE 등의 메서드로 자원을 처리하는
```

관찰 포인트
- (1) 핵심적인 내용은 맞음
- (2) 형식이 깨짐
- (3) ? 나오면서 나의 짋문을 이어서 완성함.
- (4) 모르는 질문은 그냥 넘어감.

(1) 학습시키려는 내용은 제대로 학습이 되었다.

(2) 형식이 깨지는 이유는 이전의 답변의 형식을 데이터가 학습하여 그렇게 착각하였기 때문임, 또한 `token` 최대 개수는 32개로 인해 중간에 말이 끊기는 것은 자연스러운 상황임.

(3) `?`가 나오면서 질문을 이어서 완성하는 것은, 언어 모델로서 다음에 올 가장 높은 확률의 토큰이기 `?`이기 때문임. **작은 모델**에서는 문장 구조 보정을 의미생성보다 먼저 해버리기 때문임. 오히려 언어적 감각이 살아있다는 신호

(4) 모르는 질문은 그냥 넘어가는 것은 모델이 지금 '답해야 한다'는 규칙이 없기 때문임. 다음에 올 확률이 높은 토큰을 예측하는 것이지, 질문에 반드시 답하거나 모르면 모른다고 답한다는 규칙을 한번도 학습하지 못했음.


> 기대 범위 안의 결과, 작은 모델에서 할 수 있는 만큼은 하였다. 

---
### Catastrophic Forgetting
Pre-datset: ver1
Now-datset: ver2

새로운 질문, `박찬호`에 대해서 학습시키려고 했음, 하지만, 이것을 그냥하게 되면 Catastrophic Forgetting 현상이 발생함.
| 화이트보드를 전체 지우고 칠하는 것이 아니라,
| 같은 부분 위에 다시 덧칠하는 느낌.

```Results
0 Tokens seen: 148
1 Tokens seen: 296
2 Tokens seen: 444
3 Tokens seen: 592
4 Tokens seen: 740
Epoch: 0, Loss: 2.8671875
5 Tokens seen: 888
6 Tokens seen: 1036
7 Tokens seen: 1184
8 Tokens seen: 1332
9 Tokens seen: 1480
Epoch: 1, Loss: 0.410546875
10 Tokens seen: 1628
11 Tokens seen: 1776
12 Tokens seen: 1924
13 Tokens seen: 2072
14 Tokens seen: 2220
Epoch: 2, Loss: 0.11416015625
15 Tokens seen: 2368
16 Tokens seen: 2516
17 Tokens seen: 2664
18 Tokens seen: 2812
19 Tokens seen: 2960
Epoch: 3, Loss: 0.036328125
20 Tokens seen: 3108
21 Tokens seen: 3256
22 Tokens seen: 3404
23 Tokens seen: 3552
24 Tokens seen: 3700
Epoch: 4, Loss: 0.0124267578125
25 Tokens seen: 3848
26 Tokens seen: 3996
27 Tokens seen: 4144
28 Tokens seen: 4292
29 Tokens seen: 4440
Epoch: 5, Loss: 0.00638427734375
30 Tokens seen: 4588
31 Tokens seen: 4736
32 Tokens seen: 4884
33 Tokens seen: 5032
34 Tokens seen: 5180
Epoch: 6, Loss: 0.0043792724609375
35 Tokens seen: 5328
36 Tokens seen: 5476
37 Tokens seen: 5624
38 Tokens seen: 5772
39 Tokens seen: 5920
Epoch: 7, Loss: 0.0035064697265625
40 Tokens seen: 6068
41 Tokens seen: 6216
42 Tokens seen: 6364
43 Tokens seen: 6512
44 Tokens seen: 6660
Epoch: 8, Loss: 0.00306549072265625
45 Tokens seen: 6808
46 Tokens seen: 6956
47 Tokens seen: 7104
48 Tokens seen: 7252
49 Tokens seen: 7400
Epoch: 9, Loss: 0.00269775390625

# 약 12분 소요됨.
```

```Catastrophic ver2 Results
#############################
##     After FFT Answer    ##
#############################

Q0: 박찬호는 왜 야구를 시작했나요? 박찬호는 어릴 때부터 공을 던지는 것을 좋아했고, 우연히 학교 야구부에 들어가면서 본격적으로 야
Q1: 박찬호가 메이저리그에 진출할 수 있었던 이유는 무엇인가요?
Q2: 박찬호는 경기 전에 어떤 루틴을 가지고 있었나요? 경기 전에는 항상 스트레칭과 가벼운 러닝을 하며, 마 음을 가다듬기 위해 조용히 집중
Q3: 박찬호가 가장 중요하게 생각한 운동 습관은 무엇인가요?  부상을 방지하기 위한 꾸준한 몸 관리와 기본적인 체력 훈련을 가장 중요하게 여겼습니다.
Q4: 박찬호는 은퇴 후 어떤 활동을 하고 있나요? 은퇴 후에는 야구 해설, 강연, 그리고 후배 선수들을 위한 멘토 활동을 하고 있습니다.
Q5: 박찬호가 힘든 시기를 극복한 방법은 무엇이었나요?
Q6: 박찬호는 팀워크를 어떻게 생각했나요?
Q7: 박찬호가 후배 선수들에게 자주 하는 조언은 무엇인가요?
Q8: 박찬호는 야구 외에 관심 있던 분야가 있었나요?
Q9: 박찬호에게 성공이란 무엇을 의미하나요? 성공이란 단순한 기록이 아니라, 최선을 다한 후 스스로 만족할 수 있는 상태라고 말합니다.
Q10: 파이썬에서 가상환경을 사용하는 이유는 무엇인가요?
Q11: 방구나 먹으시고 ㅋㅋ;
Q12: 박찬호는 은퇴 후 어떤 활동을 하고 있나요? 은퇴 후에는 야구 해설, 강연, 그리고 후배 선수들을 위한  멘토 활동을 하고 있습니다.
Q13: 파이썬에서 리스트와 튜플의 차이는 무엇인가요?
Q14: 박찬호는 왜 야구를 시작했나요? 박찬호는 어릴 때부터 공을 던지는 것을 좋아했고, 우연히 학교 야구부 에 들어가면서 본격적으로 야
클래스와 객체의 차이는 무엇인가요?
Q14: 클래스와 객체의 차이는 무엇인가요? 클래스는 객체를 생성하기 위한 설계도이고, 객체는 클래스를 기반 으로 생성된 실제 인스턴스입니다.
```

질문:
> 갑자기 든 생각 GPT에게 질문하기: "흠 근데 이게 질문 개수가 2:1로 차이나는 것도 학습에 영향이 가나, 왜냐하면 하나의 질문엔 하나의 고정된 답이니깐, 학습에 큰 영향이 없다고 생각했는데, 토큰 개수가 차이가 나기도 하니깐"

답변: LM은 질문 단위로 학습하지 않고 토큰 단위로 하기 때문에 학습 데이터가 부족한거로 인식한다.

**심지어**
CS QA : 박찬호 QA 가 2:1 비율 만이 아니라 평균 답변 길이(40:15)도 달라서 학습되는 토큰의 양도 다른다.
답변의 길이가 길수록 더 중요한 데이터로 인식한다. 질문의 길이가 같아도 답변 토큰이 많을수록 그 유형의 패턴을 강하게 학습함.

**추가적인 질문**: 그러면 토큰이 길수록 학습을 잘된다는 의미인가? 질문으로 답변이 나뉘어져도 서로 다른 답변 간의 관계성이 확인이 되면서 학습이 된다는 의미인가?

(1) 단순히 더 잘 준다는 것은 아님, `더 많은 영향을 준다`가 맞는 표현.
> 답변 20 tokens -> Gradient 20번 반영
> 답변 60 tokens -> Gradient 60번 반영

(2) 또한, 직접적인 관계성을 학습하는 것은 아니지만 질문-답변 패턴을 통해 암묵적인 표현 공유는 학습됨.
> 각 답변간의 내용적 관계를 이해하는 것이 아니라, 표현 공간(embedding space)에서 비슷해진다는 의미이다.
- 표현하는 법을 배운다고 생각하면 될 듯?

(3) 주의점  
> 이러한 특징으로 인해, 특정 주제에 대한 답변만 길고 다른 주제에 대한 길이가 짧다면, 긴 답변을 "정답의 표준"으로 인식하여 짧은 질문에도 불필요하게 길게 답하거나 답변의 퀄리티가 떨어질 수도 있음.
- 데이터의 불균형으로 인해 ver1(처음 학습시킨 내용)의 망각 작용이 잘 일어나지 않고, 새로운 학습 데이터에 대한 학습은 군데군데 적용됨.

### Continuous Learning
Research Model: A -> B -> A+B (CL)
Normal Model: A+B

1. Replay
- 새로운 데이터 + 이전 데이터 일부를 섞어 만듬.
- 안전함 / 원인 분석 가능

2. Parameter-efficient
- 본체 고정
- 새 태스크는 작은 모듈만 학습
CL의 실질 해법

3. Routing / MoE
- 입력에 따라
  - 어떤 head / Expert를 쓸지 분기함.
  - 새 태스크는 새로운 expert를 추가함
- GPT-4 Mixtral 계열임.

A. 현업 흐름
초기 모델 -> distil -> base -> 새 데이터 학습


### Strengthen 
### (1) base, instruct Model.
https://github.com/HongLabInc/HongLabLLM/blob/main/02_fullfinetuning1_base.ipynb
https://github.com/HongLabInc/HongLabLLM/blob/main/03_fullfinetuning2_instruct.ipynb 