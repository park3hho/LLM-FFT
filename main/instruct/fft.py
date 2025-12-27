from instruct import model, tokenizer, device, qna_list
import torch

# 파인튜닝 후에 어떻게 응답하는지 확인
model.load_state_dict(torch.load("model_ff000.pth", map_location=device, weights_only=True))
model.eval()

# 파인튜닝 후에 어떻게 대답하는지 확인
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
    print(f"Q{i}: {tokenizer.decode(output[0], skip_special_tokens=True)}")