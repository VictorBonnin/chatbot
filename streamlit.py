import streamlit as st
import requests

st.title("Chatbot temporalité")
st.text("Cet onglet permet de poser une question sur quel moment dans la journée est le plus opportun pour poster quelque chose dans un pays.")

# Historique de la conversation
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Bouton Reset pour effacer l'historique
if st.button("Reset l'historique"):
    st.session_state["messages"] = []

question = st.text_input("Pose ta question (doit inclure un pays et un moment de la journée) :")

if st.button("Envoyer"):
    if question.strip():
        payload = {"question": question}
        response = requests.post("http://localhost:8000/ask", json=payload)
        result = response.json()["response"]

        st.session_state["messages"].append(("Vous", question))
        st.session_state["messages"].append(("Chatbot", result))

for sender, msg in st.session_state["messages"]:
    if sender == "Vous":
        st.markdown(f"**Vous :** {msg}")
    else:
        st.markdown(f"**Chatbot :** {msg}")

# ----------------------------------

st.title("Analyse de tweets par pays et par moment de la journée")

if "predictions" not in st.session_state:
    st.session_state["predictions"] = []

# Formulaire pour envoyer une prédiction
with st.form("predict_form"):
    tweet = st.text_area("Texte du tweet à analyser")
    country = st.text_input("Pays")
    time_of_tweet = st.selectbox("Moment de la journée", ["morning", "noon", "night"])
    submitted = st.form_submit_button("Analyser")

    if submitted and tweet.strip() and country.strip():
        payload = {
            "country": country,
            "time_of_tweet": time_of_tweet,
            "text": tweet
        }
        response = requests.post("http://localhost:8000/predict", json=payload)
        data = response.json()

        if "prediction" in data:
            st.session_state["predictions"].append(
                (
                    tweet,
                    data["prediction"],
                    data.get("sentiment_stats", ""),
                    data.get("advice", "")
                )
            )
        else:
            st.error(data.get("error", "Erreur inconnue."))

# Bouton reset
if st.button("Reset les analyses"):
    st.session_state["predictions"] = []

# Afficher l’historique
for tweet, prediction, sentiment_stats, advice in reversed(st.session_state["predictions"]):
    st.markdown("---")
    st.markdown(f"**Texte :** {tweet}")
    st.markdown(f"**Sentiment prédit :** {prediction}")
    st.markdown(f"**Statistiques :** {sentiment_stats}")
    st.markdown(f"**Conseil :** {advice}")
