import cv2
import supervision as sv
from super_gradients.training import models
from super_gradients import setup_device
import sys

from sympy.codegen.ast import stderr

# 1. 디바이스 설정
setup_device(device="cuda")

# 2. 학습된 모델 로드
model = models.get(
    "yolo_nas_s",
    checkpoint_path="./checkpoints/IdeaFev_experiment/RUN_20251218_021451_420740/ckpt_best.pth",  # ← 본인 best 모델 경로
    num_classes=6                     # 클래스 수 맞게
)

# 3. 이미지 로드
image_path = r"C:\Users\User\Desktop\image\image1.png"
image = cv2.imread(image_path)

# 4. 추론
result = model.predict(
    image,
    conf=0.25,     # confidence threshold
    iou=0.7
)

# 5. Supervision 형식으로 변환
detections = sv.Detections.from_yolo_nas(result)

# 클래스 이름
CLASS_NAMES = [
    "happy",
    "surprise",
    "anger",
    "anxiety",
    "hurt",
    "sad",
]
# 6. 박스 + 라벨 시각화 (수정된 부분)
box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_scale=0.5)

# 안전하게 라벨 문자열 생성
labels = []
try:
    # detections.class_id는 배열 또는 리스트 형태이므로 안전하게 int로 변환
    for class_id, confidence in zip(detections.class_id, detections.confidence):
        labels.append(f"{CLASS_NAMES[int(class_id)]} {float(confidence):.2f}")
except Exception:
    # 예외가 발생하면 빈 라벨로 처리
    labels = []
emotion_scores_raw = {name: 0.0 for name in CLASS_NAMES}

for class_id, confidence in zip(detections.class_id, detections.confidence):
    class_name = CLASS_NAMES[int(class_id)]
    emotion_scores_raw[class_name] += float(confidence)

total_score = sum(emotion_scores_raw.values())

if total_score > 0:
    emotion_scores = {
        k: v / total_score
        for k, v in emotion_scores_raw.items()
    }
else:
    # 아무 것도 검출 안 된 경우
    emotion_scores = {
        k: 0.0 for k in CLASS_NAMES
    }

print("Emotion scores:", emotion_scores, file=sys.stderr)
# 이미지 복사본에 박스 먼저 그리고 라벨 추가
annotated = box_annotator.annotate(
    scene=image.copy(),
    detections=detections,
)

if len(labels) > 0:
    annotated = label_annotator.annotate(
        scene=annotated,
        detections=detections,
        labels=labels
    )

# 7. 결과 출력 (수정: 올바른 변수 사용)
cv2.imshow("YOLO-NAS Result", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
