import os
import dataset
import preprocess
import numpy as np
from PIL import Image

image_annotation_tuples = dataset.load_image_annotation_tuples()
img_path, annotation = image_annotation_tuples[0]
img_arr = np.array(Image.open(img_path))

# Ép kiểu int
ymin = int(annotation["ymin"])
ymax = int(annotation["ymax"])
xmin = int(annotation["xmin"])
xmax = int(annotation["xmax"])

# Tạo mask 2D (dùng nếu chỉ cần mặt nạ vùng bbox)
bbox_arr = np.zeros(img_arr.shape[:2], dtype=np.uint8)
bbox_arr[ymin:ymax+1, xmin:xmax+1] = 1

mammogram, breast_mask, mass_bbox = preprocess.crop(img_arr, bbox_arr)
minmax_normalized = preprocess.minmax_normalization(mammogram)
trunc_normalized = preprocess.truncation_normalization(mammogram, breast_mask)
cl2 = preprocess.clahe(trunc_normalized, 0.01)

# Chuyển đổi ảnh nếu cần
minmax_disp = preprocess.normalize_for_display(minmax_normalized)
trunc_disp  = preprocess.normalize_for_display(trunc_normalized)
cl2_disp    = preprocess.normalize_for_display(cl2)

# Tạo thư mục nếu chưa tồn tại
save_dir = "out/check_preprocess"
os.makedirs(save_dir, exist_ok=True)

# Tạo tên ảnh dựa trên tên file gốc
basename = os.path.splitext(os.path.basename(img_path))[0]

# Lưu ảnh
Image.fromarray(minmax_disp).save(os.path.join(save_dir, f"{basename}_minmax.png"))
Image.fromarray(trunc_disp).save(os.path.join(save_dir, f"{basename}_trunc.png"))
Image.fromarray(cl2_disp).save(os.path.join(save_dir, f"{basename}_clahe.png"))

# Stack 3 ảnh grayscale thành 1 ảnh 3 kênh và chuẩn hóa về uint8
mammogram_stack = np.stack([minmax_normalized, trunc_normalized, cl2], axis=2)
mammogram_stack_disp = preprocess.normalize_for_display(mammogram_stack)
Image.fromarray(mammogram_stack_disp).save(os.path.join(save_dir, f"{basename}_stack.png"))
