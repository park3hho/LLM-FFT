import matplotlib.pyplot as plt

losses = []

with open("epoch.txt", "r", encoding="utf-8") as f:
    for line in f:
        if "Loss:" in line:
            # "Epoch: 0, Loss: 1.452734375"
            loss = float(line.split("Loss:")[1].strip())
            losses.append(loss)

# Epoch은 자동으로 0,1,2,... 로 생성
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.savefig("epoch_loss.png", dpi=300, bbox_inches="tight")

# (선택) 화면에도 같이 보고 싶으면
# plt.show()

plt.close()
