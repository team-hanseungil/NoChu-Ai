import json
import os
from PIL import Image

# ✅ 전역 클래스 고정 (YOLO에서 매우 중요)
CLASSES = {
    "기쁨": 0,
    "당황": 1,
    "분노": 2,
    "불안": 3,
    "상처": 4,
    "슬픔": 5
}

def json_to_yolo_txt(json_path, image_dir, label_dir, ix):
    data_count = 0
    all_data = 0
    os.makedirs(label_dir, exist_ok=True)

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    rename_index = ix

    for item in data:
        all_data += 1

        filename = item.get("filename")
        label = item.get("faceExp_uploader")
        if not filename or not label:
            continue

        class_id = CLASSES[label]

        boxes = item.get("annot_B", {}).get("boxes")
        if not boxes:
            continue


        if isinstance(boxes, dict):
            boxes = [boxes]

        original_img_path = os.path.join(image_dir, filename)

        try:
            with Image.open(original_img_path) as img:
                img_w, img_h = img.size
        except Exception as e:
            print(f"[경고] 이미지 로드 실패: {original_img_path} -> {e}")
            continue


        new_base = f"{rename_index:06d}"
        new_img_name = new_base + ".jpg"
        new_txt_name = new_base + ".txt"

        new_img_path = os.path.join(image_dir, new_img_name)
        txt_path = os.path.join(label_dir, new_txt_name)

        if os.path.exists(new_img_path):
            print(f"[주의] {new_img_path} 이미 존재 → 건너뜀")
            continue


        os.rename(original_img_path, new_img_path)


        with open(txt_path, 'w', encoding='utf-8') as f:
            for box in boxes:
                minX = float(box["minX"])
                minY = float(box["minY"])
                maxX = float(box["maxX"])
                maxY = float(box["maxY"])

                x_center = ((minX + maxX) / 2.0) / img_w
                y_center = ((minY + maxY) / 2.0) / img_h
                width = (maxX - minX) / img_w
                height = (maxY - minY) / img_h

                f.write(
                    f"{class_id} "
                    f"{x_center:.6f} {y_center:.6f} "
                    f"{width:.6f} {height:.6f}\n"
                )

        data_count += 1
        rename_index += 1

    print(f"[완료] {all_data}개 중 {data_count}개 라벨 생성")
    return rename_index


if __name__ == "__main__":
    image_dir = "C:/face_data/Training/images/"
    label_dir = "C:/face_data/Training/labels/"

    # ✅ classes.txt는 한 번만 생성
    with open(os.path.join(label_dir, "classes.txt"), "w", encoding="utf-8") as f:
        for name in CLASSES.keys():
            f.write(f"{name}\n")

    json_dir = [
        "C:/face_data/Validation/labels/img_emotion_validation_data(기쁨).json",
        "C:/face_data/Validation/labels/img_emotion_validation_data(당황).json",
        "C:/face_data/Validation/labels/img_emotion_validation_data(분노).json",
        "C:/face_data/Validation/labels/img_emotion_validation_data(불안).json",
        "C:/face_data/Validation/labels/img_emotion_validation_data(상처).json",
        "C:/face_data/Validation/labels/img_emotion_validation_data(슬픔).json"
    ]

    i_count = 1
    for json_path in json_dir:
        i_count = json_to_yolo_txt(json_path, "C:/face_data/Validation/images/", "C:/face_data/Validation/labels/", i_count)
