from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import os
import dotenv
from emotion_explanation import get_emotion_explanation
import json


def get_keywords(emotions: json):
    dotenv.load_dotenv()

    model = ChatGoogleGenerativeAI(model="gemini-flash-latest",
                                   google_api_key=os.getenv("GOOGLE_API_KEY"))

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
            당신은 Spotify Search API(q 파라미터)를 위한 음악 검색 키워드 생성 전문가입니다.

            **입력**: 감정 분석 결과 JSON (dominant_emotion, emotion_summary, recommended_actions 포함)

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
               - 분위기: 잔잔한, 차분한, 감성적인, 따뜻한
               - 감정: 위로, 힐링, 평온, 설렘
               - 장르: 발라드, 인디, 어쿠스틱, 재즈
               - 템포/스타일: 느린, 경쾌한, 부드러운

            4. **금지 사항**
               - 완전한 문장, 가사 형태
               - 조사(은/는/이/가) 사용
               - 아티스트명, 곡 제목
               - 따옴표, JSON, 설명 텍스트
               - 3개 초과 또는 미만

            **입력 예시**:
            {{
              "dominant_emotion": "슬픔",
              "emotion_summary": "마음이 무겁고 위로가 필요한 상태입니다.",
              "recommended_actions": ["눈을 감고 쉬기", "따뜻한 차 마시기", "잔잔한 음악 듣기"]
            }}

            **출력 예시**:
            [위로 발라드, 감성 인디, 잔잔한 피아노], [힐링 발라드, 어쿠스틱 인디, 감성 R&B]


    """),

        ("human", "{emotions}"),
    ])

    chain = prompt | model

    resp = chain.invoke({"emotions": emotions})

    return resp.content


if __name__ == "__main__":
    emotion = get_emotion_explanation()
    print("Emotion Explanation JSON:", emotion)
    keywords = get_keywords(emotion)
    print("Generated Keywords JSON:", keywords)