import os
import dataset
import pydicom
import preprocess
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

save_dir = "out/check_img"
os.makedirs(save_dir, exist_ok=True)

image_annotation_tuples = dataset.load_image_annotation_tuples()
for idx, (img_path, annotation) in enumerate(image_annotation_tuples):
    try:
        basename = os.path.splitext(os.path.basename(img_path))[0]
        img_png_path = os.path.join(save_dir, f"{basename}.png")
        
        ds = pydicom.dcmread(img_path)
        plt.imsave(img_png_path, ds.pixel_array, cmap="gray")

    except:
        print(img_path)