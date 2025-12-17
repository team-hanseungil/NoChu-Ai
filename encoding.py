import cv2
import os

img_dir = "C:/face_data/Validation/images/"

for fname in os.listdir(img_dir):
    if not fname.lower().endswith((".jpg", ".jpeg")):
        continue

    path = os.path.join(img_dir, fname)

    img = cv2.imread(path)
    if img is None:
        print("❌ 완전 손상, 삭제:", fname)
        os.remove(path)
        continue

    # 🔑 재인코딩 (이 단계에서 SOS 깨짐 제거)
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
