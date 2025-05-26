from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from utils import nettoyer_texte
from utils import extract_country_and_moment
from utils import meilleur_moment_pour_pays
import joblib
from scipy.sparse import hstack

app = FastAPI()

train_df = pd.read_csv("projet_data/train.csv", encoding="latin-1", on_bad_lines='skip')

# Charge les modèles entraînés
vectorizer = joblib.load("models/vectorizer.joblib")
encoder = joblib.load("models/encoder.joblib")
rf_model = joblib.load("models/rf_model.joblib")

class Question(BaseModel):
    question: str

class PredictRequest(BaseModel):
    country: str
    time_of_tweet: str
    text: str

# ----------- ROUTE /ask REFAITE -----------
@app.post("/ask")
def ask_question(q: Question):
    # Liste de pays & moments disponibles dans le dataset
    countries = [c for c in train_df["Country"].dropna().unique() if isinstance(c, str)]
    # moments = ["morning", "noon", "night"]   # anglais de base
    moments = ["morning", "noon", "night", "matin", "midi", "nuit", "soir"]  # + FR

    # Extraction du pays et du moment
    country, moment = extract_country_and_moment(q.question, countries, moments)

    if country and moment:
        moment_opt, score = meilleur_moment_pour_pays(train_df, country)
        if moment_opt is None:
            return {"response": "Pas de données suffisantes pour répondre."}
        if moment == moment_opt:
            return {"response": f"{moment_opt.title()} est le meilleur moment pour poster en {country} ({int(score*100)}% de tweets positifs)."}
        else:
            return {"response": f"{moment.title()} n'est pas le meilleur moment pour poster en {country}. "
                                f"Il vaudrait mieux poster à {moment_opt} ({int(score*100)}% de tweets positifs)."}
    else:
        exemples = (
            "- Exemple : 'Est-ce que le matin est un bon moment pour poster en France ?'\n"
            "- Exemple : 'Is night a good time to tweet in United States?'"
        )
        return {"response": "Merci de préciser un pays et un moment de la journée dans votre question.\n" + exemples}

def sentiment_dominant(df, pays, moment):
    res = df[(df["Country"] == pays) & (df["Time of Tweet"] == moment)]
    if len(res) == 0:
        return None
    return res["sentiment"].value_counts().idxmax()

@app.post("/predict")
def predict_sentiment(request: PredictRequest):
    try:
        # Nettoyage et vectorisation
        cleaned_text = nettoyer_texte(request.text)
        X_text = vectorizer.transform([cleaned_text])
        X_cat = encoder.transform([[request.country, request.time_of_tweet]])
        X = hstack([X_text, X_cat])

        # Prédiction du sentiment du tweet
        pred = rf_model.predict(X)[0]

        # Statistiques sur ce créneau
        df_filtre = train_df[(train_df["Country"] == request.country) &
                             (train_df["Time of Tweet"] == request.time_of_tweet)]
        if len(df_filtre) > 0:
            dominant_sent = df_filtre["sentiment"].value_counts().idxmax()
            positive_ratio = (df_filtre["sentiment"] == "positive").mean()
            stat_msg = (f"Dans ce créneau ({request.time_of_tweet}) en {request.country}, "
                        f"le sentiment dominant est **{dominant_sent}** "
                        f"({int(positive_ratio * 100)}% de tweets positifs).")
        else:
            stat_msg = "Pas de données pour ce créneau/pays."

        moment_opt, score_opt = meilleur_moment_pour_pays(train_df, request.country)
        if moment_opt is None or score_opt is None:
            advice = "Pas de conseil optimal, pas assez de données sur ce pays ou ce moment."
        elif request.time_of_tweet == moment_opt:
            advice = f"Tu es au meilleur moment pour poster en {request.country} !"
        else:
            advice = (f"{request.time_of_tweet.capitalize()} n'est pas le moment optimal pour poster en {request.country}. "
                      f"Il vaudrait mieux poster à {moment_opt} ({int(score_opt*100)}% de tweets positifs).")

        return {
            "prediction": pred,
            "sentiment_stats": stat_msg,
            "advice": advice
        }
    except Exception as e:
        return {"error": str(e)}
