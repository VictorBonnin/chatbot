import re
import string

def nettoyer_texte(tweet):
    tweet = re.sub(r'@\w+', '', tweet)
    tweet = re.sub(r'#\w+', '', tweet)
    tweet = re.sub(r'http\S+|www\S+', '', tweet)
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    tweet = tweet.lower()
    tweet = re.sub(r'\d+', '', tweet)
    tweet = re.sub(r'\s+', ' ', tweet).strip()
    return tweet

def meilleur_moment_pour_pays(df, country, sentiment="positive"):
    res = df[df["Country"] == country].groupby("Time of Tweet")["sentiment"].value_counts(normalize=True).unstack()
    if sentiment in res.columns:
        best_period = res[sentiment].idxmax()
        score = res[sentiment].max()
        return best_period, score
    return None, None

def extract_country_and_moment(question, countries, moments):
    country = None
    moment = None
    question = question.lower()
    # Dictionnaire FR->EN
    country_fr_en = {
        "france": "France",
        "états-unis": "United States",
        "usa": "United States",
        "royaume-uni": "United Kingdom",
        "angleterre": "United Kingdom",
        "allemagne": "Germany",
        "italie": "Italy",
        "espagne": "Spain",
        "brésil": "Brazil",
        "canada": "Canada",
        "japon": "Japan",
    }
    # Recherche du pays (FR ou EN)
    for c_fr, c_en in country_fr_en.items():
        if c_fr in question:
            country = c_en
            break
    if not country:  # Si pas trouvé en FR, chercher en anglais
        for c in countries:
            if c.lower() in question:
                country = c
                break
    # Recherche du moment (FR ou EN)
    for m in moments:
        if m in question:
            moment = m
            break
    # Mapping FR-EN pour le moment
    mapping_moment = {
        "matin": "morning", "morning": "morning",
        "midi": "noon", "noon": "noon",
        "après-midi": "noon",
        "nuit": "night", "soir": "night", "night": "night"
    }
    if moment in mapping_moment:
        moment = mapping_moment[moment]
    return country, moment
