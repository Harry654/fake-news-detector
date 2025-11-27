from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    access_control_allow_origin=["http://localhost:3000"]
)


model = joblib.load(os.path.join("model", "classifier.pkl"))
vectorizer = joblib.load(os.path.join("model", "vectorizer.pkl"))


class NewsInput(BaseModel):
    text: str


@app.get("/")
def home():
    return {"message": "Fake News Detection API is running"}


@app.post("/predict")
def predict(data: NewsInput):
    text_vec = vectorizer.transform([data.text])
    prediction = model.predict(text_vec)[0]
    prob = model.predict_proba(text_vec)[0]
    return {
        "prediction": "Real" if prediction == 1 else "Fake",
        "fake_probability": round(prob[0], 3),
        "real_probability": round(prob[1], 3),
    }
