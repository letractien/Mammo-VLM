import numpy as np
from skimage.filters import threshold_otsu, gaussian
from skimage.measure import label, regionprops
from skimage.exposure import equalize_adapthist

def crop(data, mask=None):
    # Otsu's thresholding after Gaussian filtering
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
    # Pixel clipped and normalized in breast ROI
    Pmin = np.percentile(data[mask!=0], 5)
    Pmax = np.percentile(data[mask!=0], 99)
    truncated = np.clip(data,Pmin, Pmax)  
    normalized = (truncated - Pmin)/(Pmax - Pmin)
    normalized[mask==0]=0
    
    return normalized

def clahe(data, clip_limit=0.01):
    # Contrast enhancement
    return equalize_adapthist(data, clip_limit=clip_limit)

def normalize_for_display(img):
    if img.max() <= 1.0:
        return (img * 255).astype(np.uint8)
    return img.astype(np.uint8)