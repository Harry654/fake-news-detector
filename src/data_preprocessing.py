import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
ps = PorterStemmer()


def clean_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = " ".join(
        [ps.stem(word) for word in text.split() if word not in stop_words]  # Stemming
    )  # Remove stopwords
    return text


def load_and_preprocess(fake_path, real_path):
    df_fake = pd.read_csv(fake_path)
    df_real = pd.read_csv(real_path)

    df_fake["label"] = 0
    df_real["label"] = 1

    df = pd.concat([df_fake, df_real], axis=0).reset_index(drop=True)
    df["text"] = df["text"].apply(clean_text)
    return df
