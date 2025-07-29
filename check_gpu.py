import torch
import os

# Tạo thư mục nếu chưa tồn tại
save_dir = "out/check_gpu"
os.makedirs(save_dir, exist_ok=True)

# Đường dẫn file log
log_file_path =  os.path.join(save_dir, f"info.log")

# Thu thập thông tin
output_lines = []
output_lines.append(f"CUDA available: {torch.cuda.is_available()}")
output_lines.append(f"Number of GPUs: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    device_index = torch.cuda.current_device()
    output_lines.append(f"Current device index: {device_index}")
    output_lines.append(f"Device name: {torch.cuda.get_device_name(device_index)}")
    output_lines.append(f"CUDA version: {torch.version.cuda}")
    output_lines.append(f"cuDNN version: {torch.backends.cudnn.version()}")
    output_lines.append(f"Device capability: {torch.cuda.get_device_capability(device_index)}")
else:
    output_lines.append("CUDA is not available.")

# Ghi vào file
with open(log_file_path, "w") as f:
    for line in output_lines:
        print(line)
        f.write(line + "\n")
