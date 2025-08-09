import os
import requests
import pandas as pd

folder_txt_path = "/root/letractien/Mammo-VLM/dataset/vindr/folder.txt"
annotations_csv_path = "/root/letractien/Mammo-VLM/dataset/vindr/finding_annotations.csv"
output_root = "/root/letractien/Mammo-VLM/dataset/vindr/images"
cookies = { "sessionid": "5u95f4mleaw4k564dkitrz9m5i0hfjtd"}

with open(folder_txt_path, "r") as f: 
    folder_list = [line.strip() for line in f.readlines() if line.strip()]

df = pd.read_csv(annotations_csv_path)

cnt = 0
for idx, folder in enumerate(folder_list):
    matching_rows = df[(df["study_id"] == folder) & (df["finding_categories"].apply(lambda x: any(cat in x for cat in ["Mass", "Suspicious Calcification"])))]

    if matching_rows.empty:
        continue 

    for _, row in matching_rows.iterrows():
        image_id = row["image_id"]
        url = f"https://www.physionet.org/files/vindr-mammo/1.0.0/images/{folder}/{image_id}.dicom?download"
        
        save_folder = os.path.join(output_root, folder)
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, f"{image_id}.dicom")

        if os.path.exists(save_path):
            print(f"Existed: {save_path}")
            continue

        response = requests.get(url, stream=True, cookies=cookies)
        if response.status_code == 200:
            with open(save_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            cnt = cnt + 1
            print(f"Successed {cnt}: {save_path}")
        else:
            print(f"Failed {url}: {response.status_code}")
