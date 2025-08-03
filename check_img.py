import os
import dataset
import preprocess
import numpy as np
from PIL import Image

image_annotation_tuples = dataset.load_image_annotation_tuples()
for idx, (img_path, annotation) in enumerate(image_annotation_tuples[:]):
    print(idx)
    print(img_path)
    # print(annotation)