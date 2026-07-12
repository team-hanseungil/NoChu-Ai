from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import os
import dotenv
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

router = APIRouter(prefix="/api/ai", tags=["keywords"])

class EmotionRequest(BaseModel):
    happy: float
    surprise: float
    anger: float
    anxiety: float
    hurt: float
    sad: float
    comment: Optional[str] = None

dotenv.load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a Spotify search keyword specialist who generates emotionally resonant Korean music queries.

## Input
- emotions: A JSON object where all emotion scores sum to 1.0.
  Emotions: happy, surprise, anger, anxiety, hurt, sad
- comment: (Optional) A text describing the user's current mood or situation.
  If provided, use it to refine the keyword tone and playlist title.
  If not provided, rely solely on the emotion scores.

## Step 1 — Weighted Emotion Analysis
Analyze all emotion scores and determine their influence weights.
Scores ≥ 0.3 are [HIGH] — must drive the keyword and playlist tone.
Scores 0.1–0.29 are [MED] — may complement the primary direction.
Scores < 0.1 are [LOW] — ignore unless no [HIGH] exists.

Extract and weight the following:
- dominant_emotion: The highest-scoring emotion. This sets the core direction.
- emotional_blend: Combination of [HIGH] and [MED] emotions and their interplay.
- mood: Translate emotion scores into musical mood descriptors with weights.
  Map as follows:
    sad / hurt         → melancholic, emotional, healing, bittersweet
    anxiety / anger    → tense, dark, intense, cathartic
    happy / surprise   → uplifting, bright, energetic, warm
- tempo: Derive likely tempo feel from the emotional tone.
- artist_style: Derive likely vocal and arrangement style from the emotional tone.
  Focus on Korean music styles:
    sad / hurt         → K-ballad vocals, soft emotional singing, delicate indie vocals
    anxiety / anger    → intense Korean hip-hop, powerful K-pop vocals, dark Korean R&B
    happy / surprise   → bright K-pop, energetic Korean dance pop, cheerful K-indie
- music_role: What the music should do for the listener.
  Map as follows:
    sad / hurt         → comfort, empathy, healing
    anxiety / anger    → calming, release, grounding
    happy / surprise   → uplift, excitement, energy

## Step 1-B — Comment Analysis (if comment is provided)
Scan the comment for any explicit Korean genre mentions (e.g., 발라드, 인디, 힙합, R&B, 아이돌, 케이팝).
If a genre is detected: [OVERRIDE] include that genre in English translation in the search query.
If no genre is detected: select the most emotionally fitting Korean style freely.

## Step 2 — Build 1 Spotify Search Query
Build a single natural search query that can be passed directly to the Spotify API search function.
The query must always include "korean" to surface Korean music results.
The query must be a single fluent phrase combining [HIGH mood] + [artist_style] + "korean" + [genre if detected].
Length: 5–9 meaningful words.

## Step 3 — Playlist Title
Write a single Korean title (under 20 characters) that:
- Reflects the dominant emotion and music role
- If comment is provided, reflect its context in the title
- Feels empathetic and personal, not clinical
- Must end with "플레이리스트"
- Uses natural phrasing — no particles (은/는/이/가) mid-sentence unless natural

## Hard Rules
- Output ONLY the 2-line format below — no JSON, no explanation, no markdown
- Line 1 must be a single string with all fields joined by " | " — no newline within line 1
- Line 2 must be the playlist title in Korean
- Never use artist names, song titles, or lyrics
- Never use vague filler words alone: "good", "best", "music", "song"
- If dominant_emotion score < 0.2 (flat distribution), default music_role to healing

## Output Format (exactly 2 lines)
<mood with weights> | <tempo feel> | <artist_style with weights> | <search query>
<플레이리스트 제목>"""),

    ("human", "emotions: {emotions}\ncomment: {comment}"),
])

chain = prompt | model

@router.post("/keywords")
async def get_keywords(request: EmotionRequest):
    emotions = request.model_dump(exclude={"comment"})  

    resp = chain.invoke({
        "emotions": str(emotions),
        "comment": request.comment or "None"  
    })

    try:
        result = resp.content[0]["text"]
        keywords, title = result.strip().split("\n")
    except Exception:
        keywords, title = resp.content.strip().split("\n")

    return {"keywords": keywords.strip(), "title": title.strip()}