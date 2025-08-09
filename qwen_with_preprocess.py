import prompt
import dataset

import os
import preprocess
import numpy as np

import pydicom
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from skimage.filters import threshold_otsu, gaussian, median, unsharp_mask
from skimage.measure import label, regionprops


import torch
torch.manual_seed(1234)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE_DIR = "/root/letractien/Mammo-VLM/.cache"

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True, cache_dir=CACHE_DIR)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map=device, trust_remote_code=True, cache_dir=CACHE_DIR).eval()
model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True, cache_dir=CACHE_DIR)

image_annotation_tuples = dataset.load_image_annotation_tuples()
save_dir = "out/detect_qwen_with_preprocess"
os.makedirs(save_dir, exist_ok=True)
log_path = os.path.join(save_dir, "log.txt")

for idx, (img_path, annotation) in enumerate(image_annotation_tuples):

    # folder = annotation['study_id']
    # os.makedirs(os.path.join(save_dir, folder), exist_ok=True)

    basename = annotation['image_id']
    # img_png_path = os.path.join(save_dir, folder, f"{basename}.png")

    ds = pydicom.dcmread(img_path)
    # plt.imsave(img_png_path, ds.pixel_array, cmap="gray")

    img_arr = ds.pixel_array.astype(np.float32)
    # img_with_bbox = preprocess.draw_bbox_grayscale(img_arr, annotation, color=255, thickness=5)

    x, m, new_annotation = preprocess.crop(img_arr, annotation=annotation)
    norm = preprocess.truncation_normalization(x, m)

    step1 = preprocess.median_denoise(norm, disk_radius=3)
    step2 = preprocess.unsharp_enhance(step1, radius=1.0, amount=1.5)
    step3 = preprocess.morphological_tophat(step2, selem_radius=15)
    step4 = preprocess.non_local_means_denoise(step3, patch_size=5, patch_distance=6, h_factor=0.8)
    step5 = preprocess.wavelet_enhancement(step4, wavelet='db8', level=1)
    final = preprocess.clahe(step5, clip_limit=0.02)
    disp = preprocess.normalize_for_display(final)
    disp = np.nan_to_num(disp)
    disp = preprocess.draw_bbox_grayscale(disp.copy(), new_annotation, color=255, thickness=5)

    # img_png_path_pre = os.path.join(save_dir, folder, f"{basename}_preprocessed.png")
    img_png_path_pre = os.path.join(save_dir, f"{basename}_preprocessed.png")
    Image.fromarray(disp).save(img_png_path_pre)

    history = [(
        f'Picture 1: <img>{img_png_path_pre}</img>\n这是什么?', 
        prompt.generate_mammogram_description(
            laterality=annotation['laterality'],
            view_position=annotation['view_position'],
            breast_density=annotation['breast_density'],
            breast_birads=annotation['breast_birads'],
            finding_categories=annotation['finding_categories'],
            finding_birads=annotation['finding_birads'],
            width=new_annotation['width'],
            height=new_annotation['height'],
            xmin=new_annotation['xmin'],
            ymin=new_annotation['ymin'],
            xmax=new_annotation['xmax'],
            ymax=new_annotation['ymax'],
        )
    )]

    query = tokenizer.from_list_format([
        {'image': img_png_path_pre},
        {'text': prompt.generate_request_description}
    ])

    response, history = model.chat(tokenizer, query=query, history=history)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"Response {idx}: {response}\n")
        f.write(f"History {idx}: {history}\n")
        f.write("\n")

    image = tokenizer.draw_bbox_on_latest_picture(response, history)
    if image:
        # image.save(os.path.join(save_dir, folder, f"{basename}_{idx}_bbox.png"))
        image.save(os.path.join(save_dir, f"{basename}_{idx}_bbox.png"))
    else:
        print("No bbox")
