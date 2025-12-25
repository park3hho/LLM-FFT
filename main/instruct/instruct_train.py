print("")
print("#############################")
print("##         Round 0         ##")
print("#############################")
print("")

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
#device = "cpu"
torch.manual_seed(123)
model.to(device)

print("")
print("#############################")
print("##         Round 1         ##")
print("#############################")
print("")