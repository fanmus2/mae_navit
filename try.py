import torch

if torch.cuda.is_available():
    print("GPU可用！")
else:
    print("GPU不可用，将使用CPU进行计算。")
    
num_devices = torch.cuda.device_count()
print(f"可用的GPU数量: {num_devices}")

print(torch.version.cuda)