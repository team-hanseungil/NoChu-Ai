from fastapi import FastAPI
from predict import router as emotion_router
from keywords import router as keyword_router

app = FastAPI()

app.include_router(emotion_router)
app.include_router(keyword_router)

@app.get("/health")
async def health():
    return {"status": "ok"}