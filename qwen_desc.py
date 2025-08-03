import dataset

import os
import preprocess
import numpy as np

import pydicom
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

import torch
torch.manual_seed(1234)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map=device, trust_remote_code=True).eval()
model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

query = tokenizer.from_list_format([
    {'image': '/root/letractien/Mammo-VLM/out/detect_qwen/0f0551f4edb5494b0d8765c23fe421ae/a37e508fc994c1c7a846ec23edfb400f.png'},
    {'text': '输出图像中小圆点的检测框'}
])

response, history = model.chat(tokenizer, query=query, history=None)
print("Response:", response)
print("History:", history)

save_dir = "out/detect_qwen"
folder = "0f0551f4edb5494b0d8765c23fe421ae"
basename = "a37e508fc994c1c7a846ec23edfb400f"
image = tokenizer.draw_bbox_on_latest_picture(response, history)
if image:
    image.save(os.path.join(save_dir, folder, f"{basename}_bbox_desc.png"))
else:
    print("No bbox")
