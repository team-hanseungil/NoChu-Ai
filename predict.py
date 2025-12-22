import cv2
import supervision as sv
from super_gradients.training import models
from super_gradients import setup_device
import sys

from sympy.codegen.ast import stderr

setup_device(device="cuda")

model = models.get(
    "yolo_nas_s",
    checkpoint_path="./checkpoints/IdeaFev_experiment/RUN_20251218_021451_420740/ckpt_best.pth",
    num_classes=6
)

image_path = r"C:\Users\User\Desktop\image\image1.png"
image = cv2.imread(image_path)

result = model.predict(
    image,
    conf=0.25,
    iou=0.7
)

detections = sv.Detections.from_yolo_nas(result)

CLASS_NAMES = [
    "happy",
    "surprise",
    "anger",
    "anxiety",
    "hurt",
    "sad",
]

box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_scale=0.5)

labels = []
try:
    for class_id, confidence in zip(detections.class_id, detections.confidence):
        labels.append(f"{CLASS_NAMES[int(class_id)]} {float(confidence):.2f}")
except Exception:
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
    emotion_scores = {
        k: 0.0 for k in CLASS_NAMES
    }

print("Emotion scores:", emotion_scores, file=sys.stderr)
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

cv2.imshow("YOLO-NAS Result", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
