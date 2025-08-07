import numpy as np
from skimage.filters import threshold_otsu, gaussian, median, unsharp_mask
from skimage.measure import label, regionprops
from skimage.exposure import equalize_adapthist
from skimage.morphology import disk, white_tophat, black_tophat
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage import exposure

def crop(data, mask=None):
    img_blurred = gaussian(data, sigma=10)
    thresh = threshold_otsu(img_blurred)
    breast_mask = (img_blurred > thresh).astype(np.uint8)
    labeled_img = label(breast_mask)
    regions = regionprops(labeled_img)
    largest_region = max(regions, key=lambda x: x.area)
    minr, minc, maxr, maxc = largest_region.bbox
    
    if mask is None: 
        return data[minr:maxr, minc:maxc], breast_mask[minr:maxr, minc:maxc]
    else:
        return data[minr:maxr, minc:maxc], breast_mask[minr:maxr, minc:maxc], mask[minr:maxr, minc:maxc]

def minmax_normalization(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def truncation_normalization(data, mask):
    Pmin = np.percentile(data[mask != 0], 5)
    Pmax = np.percentile(data[mask != 0], 99)
    
    if Pmax - Pmin == 0:
        normalized = np.zeros_like(data)
        normalized[mask != 0] = 1.0
    else:
        truncated = np.clip(data, Pmin, Pmax)
        normalized = (truncated - Pmin) / (Pmax - Pmin)
        normalized[mask == 0] = 0
    
    return normalized

def clahe(data, clip_limit=0.01):
    return equalize_adapthist(data, clip_limit=clip_limit)

def normalize_for_display(img):
    if img.max() <= 1.0:
        return (img * 255).astype(np.uint8)
    return img.astype(np.uint8)

def normalize_to_uint16(data):
    norm = (data - np.min(data)) / (np.max(data) - np.min(data))
    return (norm * 65535).astype(np.uint16)

def median_denoise(data, disk_radius=3):
    """Lọc median để loại bỏ nhiễu salt-and-pepper và giữ biên sắc nét."""
    return median(data, disk(disk_radius))

def unsharp_enhance(data, radius=1.0, amount=1.0):
    """Unsharp masking để làm sắc biên lesion/vôi hóa."""
    return unsharp_mask(data, radius=radius, amount=amount)

def morphological_tophat(data, selem_radius=15):
    """Áp dụng top-hat và bottom-hat để cải thiện tương phản các vùng sáng/tối nhỏ."""
    selem = disk(selem_radius)
    tophat = white_tophat(data, selem)
    bottomhat = black_tophat(data, selem)
    # Kết hợp tophat và bottomhat: làm rõ microstructure
    enhanced = data + tophat - bottomhat
    return enhanced / np.max(enhanced)

def non_local_means_denoise(data, patch_size=5, patch_distance=6, h_factor=1.0):
    """Khử nhiễu phi local means cho ảnh grayscale (phù hợp với ảnh DICOM, X-quang...)."""
    sigma_est = estimate_sigma(data, channel_axis=None)
    return denoise_nl_means(
        data,
        h=h_factor * sigma_est,
        patch_size=patch_size,
        patch_distance=patch_distance,
        fast_mode=True,
        channel_axis=None
    )

def global_histogram_windowing(data, pmin=5, pmax=99):
    """Window-level: cắt và chuẩn hóa lại histogram toàn ảnh như truncation."""
    Pmin = np.percentile(data, pmin)
    Pmax = np.percentile(data, pmax)
    win = np.clip(data, Pmin, Pmax)
    return (win - Pmin) / (Pmax - Pmin)

def wavelet_enhancement(data, wavelet='db8', level=1):
    """
    Chú thích: scikit-image không hỗ trợ trực tiếp wavelet.
    Bạn có thể dùng pywt: Discrete Wavelet Transform để tăng cường contrast.
    """
    import pywt
    coeffs = pywt.wavedec2(data, wavelet=wavelet, level=level)
    coeffs_enh = list(coeffs)
    for i in range(1, len(coeffs_enh)):
        cH, cV, cD = coeffs_enh[i]
        coeffs_enh[i] = (1.2 * cH, 1.2 * cV, 1.2 * cD)
    enhanced = pywt.waverec2(coeffs_enh, wavelet=wavelet)
    return np.clip(enhanced, 0, 1)

def draw_bbox_grayscale(img: np.ndarray, bbox: dict, color: float = 255.0, thickness: int = 3) -> np.ndarray:
    img_with_bbox = img.copy()
    ymin = int(bbox["ymin"])
    ymax = int(bbox["ymax"])
    xmin = int(bbox["xmin"])
    xmax = int(bbox["xmax"])

    for t in range(thickness):
        img_with_bbox[ymin + t, xmin:xmax] = color
        img_with_bbox[ymax - t, xmin:xmax] = color
        img_with_bbox[ymin:ymax, xmin + t] = color
        img_with_bbox[ymin:ymax, xmax - t] = color

    return img_with_bbox