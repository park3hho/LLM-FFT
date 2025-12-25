from base import model, tokenizer, qna_list
import torch

print("")
print("")
print("#############################")
print("##     After FFT Answer    ##")
print("#############################")
print("")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 파인튜닝 후에 어떻게 응답하는지 확인
model.load_state_dict(torch.load("model_Replay_009.pth", map_location=device, weights_only=True))
model.eval()

questions = [ qna['q'] for qna in qna_list]
questions.append("파이썬에서 가상환경을 사용하는 이유는 무엇인가요?")
questions.append("방구나 먹으시고 ㅋㅋ;")
questions.append("박찬호는 은퇴 후 어떤 활동을 하고 있나요?")
questions.append("파이썬에서 리스트와 튜플의 차이는 무엇인가요?")
questions.append("박찬호는 왜 야구를 시작했나요?")

for i, q in enumerate(questions):

    input_ids = tokenizer(
        q,
        padding=True,
        return_tensors="pt",
    )["input_ids"].to("cuda")

    # print(type(model))

    model.eval()
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=32,
            attention_mask = (input_ids != 0).long(),
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            # temperature=1.2,
            # top_k=5
        )

    output_list = output.tolist()

    print(f"Q{i}: {tokenizer.decode(output[0], skip_special_tokens=True)}")


input_ids = tokenizer(
    input(),
    padding=True,
    return_tensors="pt",
)["input_ids"].to("cuda")

# print(type(model))

model.eval()
with torch.no_grad():
    output = model.generate(
        input_ids,
        max_new_tokens=32,
        attention_mask = (input_ids != 0).long(),
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,
        # temperature=1.2,
        # top_k=5
    )

output_list = output.tolist()

print(f"Q{i}: {tokenizer.decode(output[0], skip_special_tokens=True)}")