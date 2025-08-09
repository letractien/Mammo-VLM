import os
import dataset
import preprocess
import pydicom
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu, gaussian, median, unsharp_mask
from skimage.measure import label, regionprops


def crop(data, mask=None, annotation=None):
    img_blurred = gaussian(data, sigma=10)
    thresh = threshold_otsu(img_blurred)
    breast_mask = (img_blurred > thresh).astype(np.uint8)
    labeled_img = label(breast_mask)
    regions = regionprops(labeled_img)
    largest_region = max(regions, key=lambda x: x.area)
    minr, minc, maxr, maxc = largest_region.bbox

    # Cắt ảnh và mask
    cropped_data = data[minr:maxr, minc:maxc]
    cropped_mask = breast_mask[minr:maxr, minc:maxc]

    cropped_height = cropped_data.shape[0]
    cropped_width = cropped_data.shape[1]

    if annotation is not None:
        ymin = int(annotation["ymin"])
        ymax = int(annotation["ymax"])
        xmin = int(annotation["xmin"])
        xmax = int(annotation["xmax"])

        # Điều chỉnh lại toạ độ bbox
        new_annotation = {
            "ymin": max(0, ymin - minr),
            "ymax": max(0, ymax - minr),
            "xmin": max(0, xmin - minc),
            "xmax": max(0, xmax - minc),
            "width": cropped_width,
            "height": cropped_height
        }

        if mask is None:
            return cropped_data, cropped_mask, new_annotation
        else:
            cropped_mask_data = mask[minr:maxr, minc:maxc]
            return cropped_data, cropped_mask, cropped_mask_data, new_annotation

    else:
        if mask is None:
            return cropped_data, cropped_mask
        else:
            return cropped_data, cropped_mask, mask[minr:maxr, minc:maxc]

save_dir = "out/check_crop"
os.makedirs(save_dir, exist_ok=True)

image_annotation_tuples = dataset.load_image_annotation_tuples()
for idx, (img_path, annotation) in enumerate(image_annotation_tuples):
    folder = annotation.get('study_id', f"case_{idx}")
    folder_path = os.path.join(save_dir, folder)
    os.makedirs(folder_path, exist_ok=True)

    basename = os.path.splitext(os.path.basename(img_path))[0]

    # Load ảnh DICOM và chuyển về float32
    ds = pydicom.dcmread(img_path)
    img_arr = ds.pixel_array.astype(np.float32)

    # Vẽ bbox lên ảnh gốc để lưu lại
    img_with_bbox = preprocess.draw_bbox_grayscale(img_arr.copy(), annotation, color=255, thickness=5)
    plt.imsave(os.path.join(folder_path, f"{basename}_bbox_orig.png"), img_with_bbox, cmap="gray")

    # CROP + cập nhật bbox mới
    cropped_img, breast_mask, new_annotation = crop(img_arr, annotation=annotation)

    # Vẽ lại bbox sau crop
    cropped_img_with_bbox = preprocess.draw_bbox_grayscale(cropped_img.copy(), new_annotation, color=255, thickness=5)
    plt.imsave(os.path.join(folder_path, f"{basename}_bbox_crop.png"), cropped_img_with_bbox, cmap="gray")

    # Lưu thêm ảnh mask nếu cần
    plt.imsave(os.path.join(folder_path, f"{basename}_mask.png"), breast_mask, cmap="gray")
    break