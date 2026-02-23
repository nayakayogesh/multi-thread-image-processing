import cv2
import os

original_path = "../images/4k-image.jpg"
output_dir = "test_images"
os.makedirs(output_dir, exist_ok=True)

img = cv2.imread(original_path)

targets = [3840, 1920, 1280, 854, 426, 216]
labels  = ["4k", "1080p", "720p", "480p", "240p", "122p"]

for w, label in zip(targets, labels):
    h = int(img.shape[0] * (w / img.shape[1]))
    resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_LANCZOS4)
    out_path = os.path.join(output_dir, f"test_{label}.jpg")
    cv2.imwrite(out_path, resized)
    print(f"Saved: {out_path} ({w}x{h})")