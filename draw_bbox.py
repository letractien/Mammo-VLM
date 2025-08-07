import os
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

label_path="./dataset/vindr/finding_annotations.csv"
images_path="./out/detect_qwen"
type_image=".png"

df = pd.read_csv(label_path)

folder_images = [
    f for f in os.listdir(images_path)
    if os.path.isdir(os.path.join(images_path, f))
]

image_id_to_path = {}
for folder in folder_images:
    folder_path = os.path.join(images_path, folder)
    for img_file in os.listdir(folder_path):
        image_id = img_file.replace(type_image, "")
        image_id_to_path[image_id] = os.path.join(folder_path, img_file)

image_annotation_tuples = []
for _, row in df.iterrows():
    image_id = row['image_id']
    if image_id in image_id_to_path:
        img_path = image_id_to_path[image_id]
        annotation = row
        image_annotation_tuples.append((img_path, annotation))

for img_path, annotation in image_annotation_tuples:
    
    save_dir = "out/detect_qwen"
    os.makedirs(save_dir, exist_ok=True)

    folder = annotation['study_id']
    os.makedirs(os.path.join(save_dir, folder), exist_ok=True)

    basename = annotation['image_id']
    img_png_path_in = os.path.join(save_dir, folder, f"{basename}.png")
    img_png_path_out = os.path.join(save_dir, folder, f"{basename}_realbbox.png")

    image = Image.open(img_png_path_in)

    ymin = int(annotation["ymin"])
    ymax = int(annotation["ymax"])
    xmin = int(annotation["xmin"])
    xmax = int(annotation["xmax"])

    draw = ImageDraw.Draw(image)
    draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=10)

    image.save(img_png_path_out)
    print(img_png_path_out)