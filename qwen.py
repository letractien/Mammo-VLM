import dataset

import os
import preprocess
import numpy as np

import pydicom
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

import torch
torch.manual_seed(1234)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map=device, trust_remote_code=True).eval()
model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

image_annotation_tuples = dataset.load_image_annotation_tuples()
img_path, annotation = image_annotation_tuples[0]

save_dir = "out/detect_qwen"
os.makedirs(save_dir, exist_ok=True)

folder = annotation['study_id']
os.makedirs(os.path.join(save_dir, folder), exist_ok=True)

basename = annotation['image_id']
img_png_path = os.path.join(save_dir, folder, f"{basename}.png")

ds = pydicom.dcmread(img_path)
plt.imsave(img_png_path, ds.pixel_array, cmap="gray")

def generate_mammogram_description(
    laterality,
    view_position,
    breast_density,
    breast_birads,
    finding_categories,
    finding_birads,
    width,
    height,
    xmin,
    ymin,
    xmax,
    ymax
):
    density_descriptions = {
        "DENSITY A": "乳房几乎完全由脂肪组织组成。",
        "DENSITY B": "乳房中散布有部分致密组织。",
        "DENSITY C": "乳腺组织密度相对较高。",
        "DENSITY D": "乳腺组织非常致密。"
    }

    birads_descriptions = {
        "BI-RADS 0": "需要进一步成像检查。",
        "BI-RADS 1": "未见明显异常。",
        "BI-RADS 2": "发现良性病变。",
        "BI-RADS 3": "存在轻微但不明确的异常。",
        "BI-RADS 4": "发现可疑病变，需要进一步评估。",
        "BI-RADS 5": "高度怀疑恶性病变。",
        "BI-RADS 6": "已确诊的恶性病变。"
    }

    category_descriptions = {
        "Mass": (
            "肿块（Mass）是可以在乳腺X线片上观察到的病变。"
            "它们通常呈圆形、椭圆形或不规则形状，边缘可能清晰、模糊或呈分叶状。"
        ),
        "Suspicious Calcification": (
            "可疑钙化灶（Suspicious Calcification）是细小、不规则的钙化点，"
            "通常成簇或线性分布，可能呈树枝状或形态不一致。"
        )
    }

    view_text = '头尾向（CC）' if view_position == 'CC' else '内外斜向（MLO）'
    side_text = '左侧' if laterality == 'L' else '右侧'
    density_description = density_descriptions.get(breast_density, "密度未明确。")
    breast_birads_desc = birads_descriptions.get(breast_birads, "未定义。")
    finding_birads_desc = birads_descriptions.get(finding_birads, "未定义。")

    if isinstance(finding_categories, list):
        finding_cat = finding_categories[0]
    elif isinstance(finding_categories, str):
        finding_cat = finding_categories.strip("[]'\" ")
    else:
        finding_cat = str(finding_categories)

    category_description = category_descriptions.get(finding_cat, "未识别的病变类型。")
    description = f"""该乳腺X线图像为{side_text}乳房（Laterality: {laterality}），拍摄视角为 {view_position}（{view_text}）。乳腺密度：{breast_density} – {density_description} 总体 BI-RADS 等级：{breast_birads} – {breast_birads_desc} 检测到的病变类型：{finding_cat} – {category_description} 病变 BI-RADS 等级：{finding_birads} – {finding_birads_desc} 图像尺寸：{width} × {height} 像素。病变框选区域从 ({xmin:.2f}, {ymin:.2f}) 到 ({xmax:.2f}, {ymax:.2f})。"""
    return description

query = tokenizer.from_list_format([
    {'image': img_png_path},
    {'text': '框出图中病变的位置'}
])

history = [(
    f'Picture 1: <img>{img_png_path}</img>\n这是什么?', 
    generate_mammogram_description(
        laterality=annotation['laterality'],
        view_position=annotation['view_position'],
        breast_density=annotation['breast_density'],
        breast_birads=annotation['breast_birads'],
        finding_categories=annotation['finding_categories'],
        finding_birads=annotation['finding_birads'],
        width=annotation['width'],
        height=annotation['height'],
        xmin=annotation['xmin'],
        ymin=annotation['ymin'],
        xmax=annotation['xmax'],
        ymax=annotation['ymax'],
    )
)]

response, history = model.chat(tokenizer, query=query, history=history)
print("Response:", response)
print("History:", history)

# query = tokenizer.from_list_format([
#     {'image': img_png_path},
#     {'text': '这是什么?'}
# ])

# response, history = model.chat(tokenizer, query=query, history=None)
# print("Response 1:", response)
# print("History 1:", history)

# response, history = model.chat(tokenizer, '框出图中乳房的位置', history=history)
# print("Response 2:", response)
# print("History 2:", history)

# History 1: [('Picture 1: <img>out/detect_qwen/08ed8aa45238cb39cd8b5f177225b7f6/a27937a232c49b913dccf46fd810b7bf.png</img>\n这是什么?', '这是一张乳房x光片，也称为乳腺癌筛查或乳腺癌预防。该图像显示了一个女性的乳房，用于检测乳房肿瘤或异常肿块。这项检查可以帮助医生检测和治疗乳腺癌早期症状，从而提高治疗效果和生存率。')]
# History 2: [('Picture 1: <img>out/detect_qwen/08ed8aa45238cb39cd8b5f177225b7f6/a27937a232c49b913dccf46fd810b7bf.png</img>\n这是什么?', '这是一张乳房x光片，也称为乳腺癌筛查或乳腺癌预防。该图像显示了一个女性的乳房，用于检测乳房肿瘤或异常肿块。这项检查可以帮助医生检测和治疗乳腺癌早期症状，从而提高治疗效果和生存率。'), ('框出图中乳房的位置', '<ref>乳房</ref><box>(2,206),(297,756)</box>')]

image = tokenizer.draw_bbox_on_latest_picture(response, history)
if image:
    image.save(os.path.join(save_dir, folder, f"{basename}_bbox.png"))
else:
    print("No bbox")
