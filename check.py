import dataset
image_annotation_tuples = dataset.load_image_annotation_tuples()
print(len(image_annotation_tuples))

import os
print(len(os.listdir("/root/letractien/Mammo-VLM/out/detect_qwen_with_preprocess")))