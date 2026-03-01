import cv2
import os
import numpy as np

TEMPLATE_DIR = "PCB_DATASET/PCB_DATASET/PCB_USED"
DEFECT_DIR = "PCB_DATASET/PCB_DATASET/images/Missing_hole"
OUTPUT_DIR = "preprocessing/output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Pick one defective image
defect_img_name = os.listdir(DEFECT_DIR)[0]
defect_img_path = os.path.join(DEFECT_DIR, defect_img_name)

# Extract template id (e.g. '01')
template_id = defect_img_name.split('_')[0]
template_img_path = os.path.join(TEMPLATE_DIR, f"{template_id}.JPG")

print("Template:", template_img_path)
print("Defective:", defect_img_path)

# Read images
template = cv2.imread(template_img_path)
defect = cv2.imread(defect_img_path)

# Resize defective to template size
defect = cv2.resize(defect, (template.shape[1], template.shape[0]))

# Convert to grayscale
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
defect_gray = cv2.cvtColor(defect, cv2.COLOR_BGR2GRAY)

# Absolute difference
diff = cv2.absdiff(template_gray, defect_gray)

# Otsu thresholding
_, thresh = cv2.threshold(
    diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# Morphological cleanup
kernel = np.ones((3, 3), np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# Save outputs
cv2.imwrite(os.path.join(OUTPUT_DIR, "difference.png"), diff)
cv2.imwrite(os.path.join(OUTPUT_DIR, "defect_mask.png"), thresh)

print("Subtraction & mask generation done.")
