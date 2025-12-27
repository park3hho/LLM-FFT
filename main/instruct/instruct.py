print("")
print("#############################")
print("##         Round 0         ##")
print("#############################")
print("")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "kakaocorp/kanana-nano-2.1b-instruct" # "-instruct" 지시에 따르도록 파인튜닝(사후훈련)이 된 모델
# model_name = "kakaocorp/kanana-nano-2.1b-base" # base 모델로도 지시 훈련됨.
# model_name = "microsoft/Phi-4-mini-instruct" # MIT 라이센스라서 상업적 사용 가능, 아래에서 epoch 50번 정도면 훈련됨

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    # torch_dtype="auto", # Phi-4-mini 모델
    trust_remote_code=True,
).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token # <|eot_id|> 128009

print("")
print("#############################")
print("##         Round 1         ##")
print("#############################")
print("")

messages = [
    {"role": "system", "content": "You are a helpful AI assistant developed by Kakao."},
    {"role": "user", "content": "홍정모가 좋아하는 게임은?"},
    {"role": "assistant", "content":"홍정모는 헬다이버즈2를 좋아해서 자주합니다."}
]

tokens = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

print(tokens)
#print(tokenizer.encode(tokens, add_special_tokens=False))

print("")
print("#############################")
print("##         Round 2         ##")
print("#############################")
print("")

from pathlib import Path

qna_list = []

# 데이터 읽기
text = Path("../data/dataset3.txt").read_text(encoding="utf-8")

# <|question|> 단위로 나누기
blocks = text.split("<|question|>")

for block in blocks:
    block = block.strip()
    if not block:
        continue

    if "<|answer|>" not in block:
        continue

    # 질문과 답변 분리
    q_part, a_part = block.split("<|answer|>", 1)
    question = q_part.strip()
    answer = a_part.strip()

    # Kakao chat template 메시지 구성
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant developed by Kakao."},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]

    # tokenizer를 이용해 prompt 생성
    # 질문만
    q_prompt = tokenizer.apply_chat_template(messages[:2], tokenize=False, add_generation_prompt=True)
    # 질문 + 답변
    full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # 마지막 generation prompt 제거 (Kakao tokenizer 특성)
    if full_prompt.endswith('start_header_id\>assistant<|end_header_id|>'):
        full_prompt = full_prompt[:-len('start_header_id\>assistant<|end_header_id|>')]

    # 토큰화
    q_ids = tokenizer.encode(q_prompt, add_special_tokens=False)
    input_ids = tokenizer.encode(full_prompt, add_special_tokens=False)

    # 리스트에 저장
    qna_list.append({
        "q": q_prompt,
        "a": answer,
        "input": full_prompt,
        "q_ids": q_ids,
        "input_ids": input_ids
    })

# 최대 토큰 길이 계산
max_length = max(len(i['input_ids']) for i in qna_list)

print(f"QNA count: {len(qna_list)}")
print(f"Max token length: {max_length}")
print("샘플:", qna_list[0])

print("")
print("#############################")
print("##         Round 3         ##")
print("#############################")
print("")

from torch.utils.data import Dataset, DataLoader

EOT = 128009

class MyDataset(Dataset):
    def __init__(self, qna_list, max_length):
        self.input_ids = []
        self.target_ids = []

        # token_ids = tokenizer.encode("<|endoftext|>" + txt, allowed_special={"<|endoftext|>"})
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
print("----------")
print(tokenizer.decode(y_temp))

print("")
print("#############################")
print("##         Round 4         ##")
print("#############################")
print("")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
#device = "cpu"
torch.manual_seed(123)
model.to(device)

print("")
print("#############################")
print("##         Round 5         ##")
print("#############################")
print("")

# 파인튜닝 전에 어떻게 대답하는지 확인
questions = [ qna['q_ids'] for qna in qna_list]

for i, q_ids in enumerate(questions):

    model.eval()
    with torch.no_grad():
        output = model.generate(
            torch.tensor([q_ids]).to("cuda"),
            max_new_tokens=32,
            #attention_mask = (input_ids != 0).long(),
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            # temperature=1.2,
            # top_k=5
        )

    output_list = output.tolist()
    print("")
    print(f"==================Q{i}==================")
    print(f"{tokenizer.decode(output[0], skip_special_tokens=True)}")