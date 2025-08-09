import os
import dataset
import preprocess
import pydicom
import numpy as np
from PIL import Image

save_dir = "out/check_preprocess"
os.makedirs(save_dir, exist_ok=True)

image_annotation_tuples = dataset.load_image_annotation_tuples()
for idx, (img_path, annotation) in enumerate(image_annotation_tuples):

    ds = pydicom.dcmread(img_path)
    img_arr = ds.pixel_array

    ymin = int(annotation["ymin"])
    ymax = int(annotation["ymax"])
    xmin = int(annotation["xmin"])
    xmax = int(annotation["xmax"])

    bbox_arr = np.zeros(img_arr.shape[:2], dtype=np.uint8)
    bbox_arr[ymin:ymax+1, xmin:xmax+1] = 1

    mammogram, breast_mask, mass_bbox = preprocess.crop(img_arr, bbox_arr)
    minmax_normalized = preprocess.minmax_normalization(mammogram)
    trunc_normalized = preprocess.truncation_normalization(mammogram, breast_mask)
    cl2 = preprocess.clahe(trunc_normalized, 0.01)

    minmax_disp = preprocess.normalize_for_display(minmax_normalized)
    trunc_disp  = preprocess.normalize_for_display(trunc_normalized)
    cl2_disp    = preprocess.normalize_for_display(cl2)

    folder = annotation['study_id']
    os.makedirs(os.path.join(save_dir, folder), exist_ok=True)
    
    basename = os.path.splitext(os.path.basename(img_path))[0]

    Image.fromarray(minmax_disp).save(os.path.join(save_dir, folder, f"{basename}_minmax.png"))
    Image.fromarray(trunc_disp).save(os.path.join(save_dir, folder, f"{basename}_trunc.png"))
    Image.fromarray(cl2_disp).save(os.path.join(save_dir, folder, f"{basename}_clahe.png"))

    # mammogram_stack = np.stack([minmax_normalized, trunc_normalized, cl2], axis=2)
    # mammogram_stack_disp = preprocess.normalize_for_display(mammogram_stack)
    # Image.fromarray(mammogram_stack_disp).save(os.path.join(save_dir, folder, f"{basename}_stack.png"))
    # break
