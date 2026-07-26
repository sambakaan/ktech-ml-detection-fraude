"""
Chargement robuste du modèle de détection de fraude, partagé entre `app/main.py` et les
pages qui ont besoin de garder l'utilisateur informé quand aucun modèle n'est disponible
(au lieu de lui demander de lancer une commande en ligne de commande).
"""
import os
import sys

import joblib
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "ktech_bank_transaction_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "fraud_model.pkl")


@st.cache_resource
def charger_artefacts():
    """Charge le modèle depuis le disque. Si le fichier est absent, ou illisible (ex :
    pickle entraîné avec une version de scikit-learn/xgboost/lightgbm incompatible avec
    l'environnement courant), relance automatiquement l'entraînement à partir des données
    par défaut plutôt que de faire planter l'application."""
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception as e:
            st.warning(
                f"Le modèle enregistré est illisible dans cet environnement ({e}) — "
                "un nouvel entraînement automatique va être lancé."
            )

    if not os.path.exists(CSV_PATH):
        return None

    with st.spinner(
        "Aucun modèle entraîné valide trouvé — entraînement automatique en cours "
        "(quelques dizaines de secondes, une seule fois)..."
    ):
        sys.path.append(os.path.join(BASE_DIR, "ml"))
        import train  # noqa: E402
        train.main()

    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            return None
    return None


def bouton_entrainer_maintenant(cle: str):
    """Bouton qui vide le cache du modèle et relance la page pour déclencher un entraînement."""
    if st.button(":material/model_training: Entraîner le modèle maintenant", type="primary", key=cle):
        charger_artefacts.clear()
        st.rerun()


def garde_modele(artefacts, cle_bouton: str) -> bool:
    """Affiche un message et un bouton d'entraînement si aucun modèle n'est disponible.

    Retourne True si l'appelant doit arrêter le rendu de la page (`st.stop()`)."""
    if artefacts is not None:
        return False
    st.warning(
        "Aucun modèle entraîné n'est disponible pour le moment. Cliquez ci-dessous pour "
        "lancer l'entraînement automatique à partir des données actuellement chargées."
    )
    bouton_entrainer_maintenant(cle_bouton)
    return True
