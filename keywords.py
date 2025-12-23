from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import os
import dotenv
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/api/ai", tags=["keywords"])

class Emotions(BaseModel):
    happy: float
    surprise: float
    anger: float
    anxiety: float
    hurt: float
    sad: float

dotenv.load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-flash-latest",
                                   google_api_key=os.getenv("GOOGLE_API_KEY"))

prompt = ChatPromptTemplate.from_messages([
    ("system", """
        당신은 Spotify Search API(q 파라미터)를 위한 음악 검색 키워드 생성 전문가입니다.

        **입력**: 감정 분석 결과 JSON 모든 감정의 점수의 합은 1이다.
        
        **생성 규칙**:
        1. **세 가지 정보 모두 종합 분석**
           - dominant_emotion: 핵심 감정 파악
           - emotion_summary: 현재 심리 상태 이해
           - recommended_actions: 음악의 역할 도출 (위로/안정/휴식/기분전환/힐링 등)
            
        2. **키워드 형식**
           - 정확히 3개의 한글 키워드
           - 쉼표(,)로 구분된 한 줄 문자열
           - Spotify 검색에 최적화된 형태
            
        3. **키워드 구성 요소** (자유 조합)
           - 분위기: 잔잔한, 차분한, 감성적인, 따뜻한, 경쾌한, 밝은
           - 감정: 위로, 힐링, 평온, 설렘, 행복, 신남
           - 장르: 발라드, 인디, 어쿠스틱, 재즈, 팝, 댄스
           - 템포/스타일: 느린, 경쾌한, 부드러운, 신나는
        
        4. **플레이리스트 제목**
           - 사용자의 감정 상태와 음악의 역할을 반영한 제목
           - 간결하고 공감 가능한 한글 제목 (20자 내외)
           - 예: "당신의 마음을 다독이는 플레이리스트", "당신에게 조용한 위로를 건네는 플레이리스트"
           - 반드시 플레이리스트로 끝낼 것
         
        5. **금지 사항**
           - 완전한 문장, 가사 형태
           - 조사(은/는/이/가) 사용
           - 아티스트명, 곡 제목
           - 따옴표, JSON, 추가 설명 텍스트
        
        **반드시 지켜야할 출력 형식**:
        키워드1, 키워드2, 키워드3
        플레이리스트 제목
        **content에 json 형식이나 다른 텍스트 절대 포함 금지**
            
        **입력 예시**:
        {{
           "happy": 0.6,
           "surprise": 0.1,
           "anger": 0.1,
           "anxiety": 0.1,
           "hurt": 0,
           "sad": 0.1
         }}
            
       **출력 예시**:
        신나는 팝, 경쾌한 댄스, 행복 인디
        기분 좋은 하루를 위한 플레이리스트

"""),

    ("human", "{emotions}"),
])

chain = prompt | model

@router.post("/keywords")
async def get_keywords(emotions: Emotions):
    resp = chain.invoke({"emotions": emotions})

    try:
      result = resp.content[0]["text"]

      keywords, title = result.split("\n")
    except Exception as e:
      keywords, title = resp.content.split("\n")
      return {"keywords": keywords.strip(), "title": title.strip()}

    return {"keywords": keywords.strip(), "title": title.strip()}