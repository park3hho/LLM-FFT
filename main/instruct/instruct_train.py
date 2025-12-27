print("")
print("#############################")
print("##         Round -1        ##")
print("#############################")
print("")

import torch
from instruct import model, train_loader, device

print("")
print("#############################")
print("##         Round a         ##")
print("#############################")
print("")

tokens_seen, global_step = 0, -1

losses = []

optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001, weight_decay=0.01)

for epoch in range(10):
    model.train()  # Set model to training mode

    epoch_loss = 0
    for input_batch, target_batch in train_loader:
        optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)

        logits = model(input_batch).logits  # 뒤에 .logits를 붙여서 tensor만 가져옴

        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
        epoch_loss += loss.item()
        loss.backward()  # Calculate loss gradients
        optimizer.step()  # Update model weights using loss gradients
        tokens_seen += input_batch.numel()
        global_step += 1

        print(f"{global_step} Tokens seen: {tokens_seen}")

        # if global_step % 1000 == 0:
        #     print(f"Tokens seen: {tokens_seen}")
        # Optional evaluation step

    avg_loss = epoch_loss / len(train_loader)
    losses.append(avg_loss)
    print(f"Epoch: {epoch}, Loss: {avg_loss}")
    torch.save(model.state_dict(), "model_ff" + str(epoch).zfill(3) + ".pth")

    # num_batches = 8 에서 4분 22초 (결과가 안좋음)
    # num_batches = 4 에서 6분 30초 (결과가 안좋음)
    # num_batches = 2 에서 4분 37초 (결과 정확)



