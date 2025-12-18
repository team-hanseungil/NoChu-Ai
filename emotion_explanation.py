from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import os
import dotenv


def get_emotion_explanation():
    dotenv.load_dotenv()

    model = ChatGoogleGenerativeAI(model="gemini-flash-latest",
                                   google_api_key=os.getenv("GOOGLE_API_KEY"))

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
            당신은 감정 점수를 해석하여
            현재 감정 상태를 설명하고,
            일상에서 실천할 수 있는 행동을 추천하는 감정 코치입니다.

            규칙:
            - 입력으로 주어지는 감정 점수는 0~1 사이입니다.
            - 가장 점수가 높은 감정을 중심으로 현재 상태를 해석하세요.
            - 감정 이름을 반드시 출력에 포함하세요.
            - 의학적 진단, 극단적 표현, 위협적인 문장은 절대 사용하지 마세요.
            - 청소년에게 안전하고 부담 없는 행동만 추천하세요.
            - 항상 존댓말을 사용하세요.
            - 출력은 반드시 JSON 형식이어야 하며, 다른 텍스트는 절대 포함하지 마세요.

            출력 JSON 형식:
            {{
              "dominant_emotion": string,
              "emotion_summary": string,
              "recommended_actions": [string, string, string]
            }}

        """),

        ("human", "{emotion_scores}"),
    ])

    chain = prompt | model

    emotion_scores = {
        "슬픔": 0.05,
        "기쁨": 0.12,
        "상처": 0.20,
        "당황": 0.13,
        "분노": 0.35,
        "불안": 0.15
    }

    resp = chain.invoke({"emotion_scores": emotion_scores})

    return resp.content


if __name__ == "__main__":
    explanation = get_emotion_explanation()
    print("Emotion Explanation JSON:", explanation)