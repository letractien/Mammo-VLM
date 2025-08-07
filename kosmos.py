import dataset

import os
import preprocess
import numpy as np

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForImageTextToText

import torch
torch.manual_seed(1234)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE_DIR = "/root/letractien/Mammo-VLM/.cache"

model = AutoModelForImageTextToText.from_pretrained("microsoft/kosmos-2-patch14-224", device_map=device, cache_dir=CACHE_DIR)
processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224", use_fast=True, cache_dir=CACHE_DIR)

image_annotation_tuples = dataset.load_image_annotation_tuples()
img_path, annotation = image_annotation_tuples[0]

image = Image.open(img_path)
image_arr = np.array(image)

ymin = int(annotation["ymin"])
ymax = int(annotation["ymax"])
xmin = int(annotation["xmin"])
xmax = int(annotation["xmax"])

bbox_arr = np.zeros(image_arr.shape[:2], dtype=np.uint8)
bbox_arr[ymin:ymax+1, xmin:xmax+1] = 1

mammogram, breast_mask, mass_bbox = preprocess.crop(image_arr, bbox_arr)
minmax_normalized = preprocess.minmax_normalization(mammogram)
trunc_normalized = preprocess.truncation_normalization(mammogram, breast_mask)
cl2 = preprocess.clahe(trunc_normalized, 0.01)

# Chuyển đổi ảnh nếu cần
minmax_disp = preprocess.normalize_for_display(minmax_normalized)
trunc_disp  = preprocess.normalize_for_display(trunc_normalized)
cl2_disp    = preprocess.normalize_for_display(cl2)

# Stack 3 ảnh grayscale thành 1 ảnh 3 kênh
mammogram_stack = np.stack([minmax_normalized, trunc_normalized, cl2], axis=2)
mammogram_stack_disp = preprocess.normalize_for_display(mammogram_stack)

image_stack = Image.fromarray(mammogram_stack_disp)

labels = annotation["finding_categories"]
height = annotation["height"]
width = annotation["width"]
xmin_norm = annotation["xmin"] / width
xmax_norm = annotation["xmax"] / width
ymin_norm = annotation["ymin"] / height
ymax_norm = annotation["xmax"] / height

prompt = (
    f"<grounding> {labels} at (xmin={xmin_norm:.2f}, ymin={ymin_norm:.2f}, "
    f"xmax={xmax_norm:.2f}, ymax={ymax_norm:.2f})."
)

inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
generated_ids = model.generate(
    pixel_values=inputs["pixel_values"],
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    image_embeds=None,
    image_embeds_position_mask=inputs["image_embeds_position_mask"],
    use_cache=True,
    max_new_tokens=64,
)

# Generate
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
caption, entities = processor.post_process_generation(generated_text, cleanup_and_extract=True)

print(caption) # ['Mass'] at (xmin=0.84, ymin=0.49, xmax=0.89, ymax=0.71). The image is a black and white photograph of a breast with a large, red, and white mass in the center of the breast.
print(entities) # ('The mass', (60, 68), [(0.015625, 0.015625, 0.984375, 0.984375)])]

# Lấy kích thước ảnh
width, height = image.size

# Tạo bản sao ảnh để không vẽ đè lên ảnh gốc
image_draw = image.copy()
draw = ImageDraw.Draw(image_draw)

# Duyệt qua các thực thể và vẽ bounding box
for label, _, boxes in entities:
    for box in boxes:
        x_min = int(box[0] * width)
        y_min = int(box[1] * height)
        x_max = int(box[2] * width)
        y_max = int(box[3] * height)

        draw.rectangle([(x_min, y_min), (x_max, y_max)], outline="red", width=1)
        draw.text((x_min, max(y_min - 10, 0)), label, fill="red")

save_dir = "out/detect_kosmos"
os.makedirs(save_dir, exist_ok=True)

basename = os.path.splitext(os.path.basename(img_path))[0]
image_draw.save(os.path.join(save_dir, f"{basename}_bbox.png"))
