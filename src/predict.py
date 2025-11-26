import joblib

# Load saved model and vectorizer
model = joblib.load("../model/classifier.pkl")
vectorizer = joblib.load("../model/vectorizer.pkl")


def predict_news(text):
    # Transform input text using vectorizer
    text_vec = vectorizer.transform([text])

    # Predict using model
    prediction = model.predict(text_vec)[0]
    probability = model.predict_proba(text_vec)[0]

    return {
        "prediction": "Real" if prediction == 1 else "Fake",
        "fake_probability": round(probability[0], 3),
        "real_probability": round(probability[1], 3),
    }


# Test
if __name__ == "__main__":
    sample_text = "Breaking news: Government announces new economic policy."
    print(predict_news(sample_text))
