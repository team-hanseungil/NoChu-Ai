import cv2
import supervision as sv
from super_gradients.training import models
from super_gradients import setup_device
from fastapi import UploadFile, File, APIRouter
import numpy as np

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import os
import dotenv

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

CLASS_KO = {
        "happy": "행복",
        "surprise": "놀람",
        "anger": "분노",
        "anxiety": "불안",
        "hurt": "상처",
        "sad": "슬픔",
    }

dotenv.load_dotenv()

chat_model = ChatGoogleGenerativeAI(model="gemini-flash-latest",
                                   google_api_key=os.getenv("GOOGLE_API_KEY"))

prompt = ChatPromptTemplate.from_messages([
    ("system", """
        당신은 감정 점수를 해석하여
        현재 감정 상태를 설명하는 전문가입니다.
       
        규칙:
        - 입력으로 주어지는 감정 점수는 0~1 사이입니다.
        - 가장 점수가 높은 감정을 중심으로 현재 상태를 해석하세요.
        - 감정 이름을 반드시 출력에 포함하세요.
        - 극단적 표현, 위협적인 문장은 절대 사용하지 마세요.
        - 항상 존댓말을 사용하세요.
        - 마크다운 사용금지 및 이스케이프문 사용금지
        - 문장 길이는 100자 내외로 간결하게 작성하세요.
        - 한글만 사용
        - 감정은 한글만으로 표현
          
        **무조건 지켜야할 출력 형식 예시**:
          오늘은 전반적으로 긍정적인 감정 상태입니다. 행복함이 주된 감정으로 나타났으며, 평온한 상태를 유지하고 있습니다.

    """),

    ("human", "{emotion_scores}"),
])

chain = prompt | chat_model

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

    resp = chain.invoke({"emotion_scores": emotion_scores})

    try:
      content = resp.content[0]["text"]
    except Exception as e:
      content = resp.content

    return {
        "emotions": emotion_scores,
        "emotion" : CLASS_KO[max(emotion_scores, key=emotion_scores.get)],
        "comment" : content
    }