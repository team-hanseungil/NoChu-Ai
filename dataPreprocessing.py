import json
import os
from PIL import Image
import glob

def json_to_yolo_txt(json_path, image_dir, label_dir, ix):
    data_count = 0
    all_data = 0
    os.makedirs(label_dir, exist_ok=True)

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

        # 클래스 매핑 저장
        classes = {}
        next_id = 0
        for item in data:
            cls = item.get("faceExp_uploader")
            if cls and cls not in classes:
                classes[cls] = next_id
                next_id += 1

        # classes.txt 생성
        with open(os.path.join(label_dir, 'classes.txt'), 'w', encoding='utf-8') as f:
            for cls_name in sorted(classes, key=lambda x: classes[x]):
                f.write(f"{cls_name}\n")

        rename_index = ix  # ★ 이미지/라벨 rename에 사용할 번호

        for item in data:
            all_data += 1
            filename = item["filename"]
            label = item.get("faceExp_uploader")
            if not label:
                continue
            class_id = classes[label]

            annot = item.get("annot_B", {}).get("boxes")
            if not annot:
                continue

            minX = float(annot["minX"])
            minY = float(annot["minY"])
            maxX = float(annot["maxX"])
            maxY = float(annot["maxY"])

            Image.MAX_IMAGE_PIXELS = None
            original_img_path = os.path.join(image_dir, filename)

            # 이미지 불러오기
            try:
                with Image.open(original_img_path) as img:
                    img_w, img_h = img.size
            except Exception as e:
                print(f"[경고] 이미지 불러오기 실패: {original_img_path} -> {e}")
                continue

            # 📌 YOLO 좌표 변환
            x_center = ((minX + maxX) / 2.0) / img_w
            y_center = ((minY + maxY) / 2.0) / img_h
            width = (maxX - minX) / img_w
            height = (maxY - minY) / img_h

            # ------------------------------------------------------------------
            # 📌 새 이름 생성 (예: 000001.jpg → 000001.txt)
            # ------------------------------------------------------------------
            new_base_name = f"{rename_index:06d}"
            new_img_name = new_base_name + ".jpg"
            new_txt_name = new_base_name + ".txt"

            # ------------------------------------------------------------------
            # 📌 이미지 파일 rename
            # ------------------------------------------------------------------
            new_img_path = os.path.join(image_dir, new_img_name)
            if not os.path.exists(new_img_path):  # 덮어쓰기 방지
                os.rename(original_img_path, new_img_path)
            else:
                print(f"[주의] {new_img_path} 이미 존재하여 rename 건너뜀")
                continue

            # ------------------------------------------------------------------
            # 📌 라벨 txt 생성 (new 이름으로 저장)
            # ------------------------------------------------------------------
            txt_path = os.path.join(label_dir, new_txt_name)
            with open(txt_path, 'w', encoding='utf-8') as f:
                data_count += 1
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

            # rename index 증가
            rename_index += 1

    print(f"완료: {all_data}중에 {data_count}개 이미지 txt 생성됨")
    print(f"클래스 매핑: {classes}")
    return rename_index

if __name__ == "__main__":
    json_dir=[
        "C:/face_data/Training/labels/img_emotion_training_data(기쁨).json",
        "C:/face_data/Training/labels/img_emotion_training_data(당황).json",
        "C:/face_data/Training/labels/img_emotion_training_data(분노).json",
        "C:/face_data/Training/labels/img_emotion_training_data(불안).json",
        "C:/face_data/Training/labels/img_emotion_training_data(상처).json",
        "C:/face_data/Training/labels/img_emotion_training_data(슬픔).json"
    ]
    i_count=1
    for i_dir in json_dir:
        i_count=json_to_yolo_txt(i_dir, "C:/face_data/Training/images/", "C:/face_data/Training/labels", i_count)
        print(i_count)

