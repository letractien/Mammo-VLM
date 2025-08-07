import os
import dataset
import preprocess
import pydicom
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


save_dir = "out/check_preprocess"
os.makedirs(save_dir, exist_ok=True)

image_annotation_tuples = dataset.load_image_annotation_tuples()
for idx, (img_path, annotation) in enumerate(image_annotation_tuples):
    folder = annotation.get('study_id', f"case_{idx}")
    os.makedirs(os.path.join(save_dir, folder), exist_ok=True)
    basename = os.path.splitext(os.path.basename(img_path))[0]

    ds = pydicom.dcmread(img_path)    
    plt.imsave(os.path.join(save_dir, folder, f"{basename}.png"), ds.pixel_array, cmap="gray")

    img_arr = ds.pixel_array.astype(np.float32)
    img_with_bbox = preprocess.draw_bbox_grayscale(img_arr, annotation, color=255, thickness=5)

    x, m = preprocess.crop(img_with_bbox)
    norm = preprocess.truncation_normalization(x, m)

    step1 = preprocess.median_denoise(norm, disk_radius=3)
    step2 = preprocess.unsharp_enhance(step1, radius=1.0, amount=1.5)
    step3 = preprocess.morphological_tophat(step2, selem_radius=15)
    step4 = preprocess.non_local_means_denoise(step3, patch_size=5, patch_distance=6, h_factor=0.8)
    step5 = preprocess.wavelet_enhancement(step4, wavelet='db8', level=1)
    final = preprocess.clahe(step5, clip_limit=0.02)
    disp = preprocess.normalize_for_display(final)
    disp = np.nan_to_num(disp)

    Image.fromarray(disp).save(os.path.join(save_dir, folder, f"{basename}_preprocessed.png"))
    print("Saved:", os.path.join(save_dir, folder, f"{basename}_preprocessed.png"))

    left = disp.astype(np.uint8)
    right = ds.pixel_array.astype(np.uint8)
    h1, w1 = left.shape
    h2, w2 = right.shape

    combined_height = max(h1, h2)
    combined_width = w1 + w2

    combined = np.zeros((combined_height, combined_width), dtype=np.uint8)
    combined[:h1, :w1] = left
    combined[:h2, w1:w1+w2] = right

    combined_img = Image.fromarray(combined)
    combined_path = os.path.join(save_dir, folder, f"{basename}_combined.png")
    combined_img.save(combined_path)
    # break