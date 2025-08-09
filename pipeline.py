import os
import pydicom
import numpy as np

from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
import prompt
import preprocess

class MammoPipeline:
    def __init__(self, 
                 model_name="Qwen/Qwen-VL-Chat", 
                 cache_dir="/root/letractien/Mammo-VLM/.cache",
                 save_dir="out/detect_20samples",
                 log_name="log.txt",
                 seed=1234):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, cache_dir=cache_dir
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map=self.device, trust_remote_code=True, cache_dir=cache_dir
        ).eval()
        self.model.generation_config = GenerationConfig.from_pretrained(
            model_name, trust_remote_code=True, cache_dir=cache_dir
        )

        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.log_path = os.path.join(self.save_dir, log_name)

    def preprocess_image(self, img_arr, annotation):
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
        # disp = preprocess.draw_bbox_grayscale(disp.copy(), new_annotation, color=255, thickness=5)

        return disp, new_annotation

    def run_inference(self, img_path, annotation):    
        try:
            ds = pydicom.dcmread(img_path)
            img_arr = ds.pixel_array.astype(np.float32)
        except Exception as e:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(f"Error: {e}\n\n")
            return False, None

        folder = annotation['study_id']
        os.makedirs(os.path.join(self.save_dir, folder), exist_ok=True)

        basename = annotation['image_id']
        img_png_path_pre = os.path.join(self.save_dir, folder, f"{basename}_preprocessed.png")

        disp, new_annotation = self.preprocess_image(img_arr, annotation)
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

        query = self.tokenizer.from_list_format([
            {'image': img_png_path_pre},
            {'text': prompt.generate_request_description()}
        ])

        response, history = self.model.chat(
            self.tokenizer, 
            query=query, 
            history=history
        )

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"Response: {response}\n")
            f.write(f"History: {history}\n\n")

        image = self.tokenizer.draw_bbox_on_latest_picture(response, history)
        if image:
            bbox_path = img_png_path_pre.replace("_preprocessed.png", "_bbox.png")
            image.save(bbox_path)
            return True, new_annotation

        return False, new_annotation
        