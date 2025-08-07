import os
import pandas as pd

def load_image_annotation_tuples(root_path="./dataset",
                                  label_path="vindr/finding_annotations.csv",
                                  images_path="vindr/images",
                                  type_image=".dicom"):
    df = pd.read_csv(os.path.join(root_path, label_path))
    folder_images = os.listdir(os.path.join(root_path, images_path))

    image_id_to_path = {}
    for folder in folder_images:
        folder_path = os.path.join(root_path, images_path, folder)
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

    return image_annotation_tuples