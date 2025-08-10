import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

from pipeline import MammoPipeline
pipeline = MammoPipeline()

import dataset
image_annotation_tuples = dataset.load_image_annotation_tuples(
    label_path="vindr/finding_annotations.csv", 
    images_path="vindr_20samples"
)

import os
import preprocess
import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image

for idx, (img_path, annotation) in enumerate(image_annotation_tuples):
    success, new_annotation = pipeline.run_inference(img_path, annotation)
    if not success:
        continue

    # try:
    #     folder = annotation['study_id']
    #     basename = annotation['image_id']
    #     img_png_path_pre = os.path.join(pipeline.save_dir, folder, f"{basename}_preprocessed.png")
    #     img_png_path_bbox = os.path.join(pipeline.save_dir, folder, f"{basename}_bbox.png")

    #     img1 = np.array(Image.open(img_png_path_pre))
    #     img1_bbox = preprocess.draw_bbox_grayscale(img1.copy(), new_annotation, color=255, thickness=5)

    #     img2 = np.array(Image.open(img_png_path_bbox))

    #     img3 = preprocess.draw_bbox_grayscale(img2.copy(), new_annotation, color=255, thickness=5)

    #     fig, axs = plt.subplots(1, 3, figsize=(9, 6))
    #     axs[0].imshow(img1_bbox, cmap='gray')
    #     axs[0].set_title("Old BBox")
    #     axs[0].axis("off")

    #     axs[1].imshow(img2, cmap='gray')
    #     axs[1].set_title("New Bbox")
    #     axs[1].axis("off")

    #     axs[2].imshow(img3, cmap='gray')
    #     axs[2].set_title("New BBox + Old BBox")
    #     axs[2].axis("off")

    #     plt.tight_layout()
    #     plt.show()

    #     print("Success at:", img_path)
    # except: 
    #     print("Error at: ", img_path)
        
    # break
