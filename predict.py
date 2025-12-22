import cv2
import supervision as sv
from super_gradients.training import models
from super_gradients import setup_device
from fastapi import UploadFile, File, APIRouter
import numpy as np

router = APIRouter(prefix="/api/ai", tags=["emotions"])

setup_device(device="cuda")

model = models.get(
    "yolo_nas_s",
    checkpoint_path="./model/ckpt_best.pth",
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

@router.post("/emotions")
async def emotion_detection(image: UploadFile = File(...)):

    image_bytes = await image.read()

    np_arr = np.frombuffer(image_bytes, np.uint8)

    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    result = model.predict(
        img,
        conf=0.25,
        iou=0.7
    )

    detections = sv.Detections.from_yolo_nas(result)

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

    return emotion_scores