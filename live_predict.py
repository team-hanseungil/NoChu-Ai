import cv2
import supervision as sv
from super_gradients.training import models
from super_gradients import setup_device

setup_device(device="cuda")

model = models.get(
    "yolo_nas_s",
    checkpoint_path="./checkpoints/IdeaFev_experiment/RUN_20251218_021451_420740/ckpt_best.pth",
    num_classes=6
)
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
cap = cv2.VideoCapture(0)  # 0 = 기본 웹캠

if not cap.isOpened():
    raise RuntimeError("웹캠을 열 수 없습니다.")

while True:
    ret, frame = cap.read()
    if not ret:
        break


    result = model.predict(
        frame,
        conf=0.25,
        iou=0.7
    )


    detections = sv.Detections.from_yolo_nas(result)


    labels = [
        f"{CLASS_NAMES[int(cid)]} {conf:.2f}"
        for cid, conf in zip(
            detections.class_id,
            detections.confidence
        )
    ]


    annotated = box_annotator.annotate(
        scene=frame.copy(),
        detections=detections
    )

    if labels:
        annotated = label_annotator.annotate(
            scene=annotated,
            detections=detections,
            labels=labels
        )


    cv2.imshow("YOLO-NAS Real-time", annotated)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
