"""
Application Streamlit - Détection de la fraude bancaire.

Point d'entrée : configuration de la page, thème KTech Solutions (clair/sombre/système),
chargement des données/du modèle (mis en cache), navigation native (`st.navigation`), puis
délégation à chaque page de `app/pages/` pour le rendu de la page active.

Lancement :
    streamlit run app/main.py
"""
import os
import sys

import streamlit as st
from sklearn.model_selection import train_test_split

APP_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(APP_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "ml"))

from preprocessing import charger_donnees, encoder_features, nettoyer_et_enrichir, preparer_transaction
from common.auth import exiger_authentification, render_bouton_deconnexion
from common.theme import injecter_theme, render_sidebar_brand, render_theme_switcher, resoudre_theme_actif
from common.theme_bridge import injecter_pont_js
from common.history_store import charger_historique_persistant
from common.model_utils import bouton_entrainer_maintenant, charger_artefacts

from app.pages import dashboard, data_import, eda, evaluation, training, prediction, reports

CSV_PATH = os.path.join(BASE_DIR, "data", "ktech_bank_transaction_dataset.csv")
FAVICON_PATH = os.path.join(APP_DIR, "assets", "favicon.png")

# Répertoire de persistance des données évolutives (historique des prédictions, historique
# d'entraînement) — distinct du dataset de référence, qui doit rester embarqué dans l'image
# Docker. Vaut data/ par défaut (comportement local inchangé) ; en production, pointé vers un
# volume monté via FRAUD_APP_PERSIST_DIR (voir Dockerfile/docker-compose.yml/render.yaml).
PERSIST_DIR = os.environ.get("FRAUD_APP_PERSIST_DIR", os.path.join(BASE_DIR, "data"))
HISTORY_PATH = os.path.join(PERSIST_DIR, "predictions_history.csv")
TRAINING_HISTORY_PATH = os.path.join(PERSIST_DIR, "training_runs_history.csv")

st.set_page_config(
    page_title="Détection de fraude bancaire",
    page_icon=FAVICON_PATH if os.path.exists(FAVICON_PATH) else ":material/account_balance:",
    layout="wide",
)

theme_actif = resoudre_theme_actif()
injecter_theme(theme_actif)
injecter_pont_js(st.session_state.get("theme_choice", "system") == "system")

nom_utilisateur, identifiant_utilisateur = exiger_authentification()


# ----------------------------------------------------------------------------
# Chargement des données et du modèle (mis en cache)
# ----------------------------------------------------------------------------
@st.cache_data
def _lire_fichier_defaut():
    if not os.path.exists(CSV_PATH):
        return None
    return charger_donnees(CSV_PATH)


def get_raw_data():
    """Retourne le jeu de données actif : celui chargé/généré depuis la page
    *Import & gestion des données* si présent, sinon le fichier par défaut."""
    df_actif = st.session_state.get("df_actif")
    if df_actif is not None:
        return df_actif
    return _lire_fichier_defaut()


@st.cache_data
def get_test_split(top_localisations, feature_columns):
    """Reconstruit le même split train/test que celui utilisé à l'entraînement (même
    random_state), à partir du fichier de données PAR DÉFAUT (le modèle entraîné y est lié)."""
    df = _lire_fichier_defaut()
    df_clean, _ = nettoyer_et_enrichir(df, top_localisations=top_localisations)
    X, _ = encoder_features(df_clean, feature_columns=feature_columns)

    artefacts = charger_artefacts()
    y = artefacts["label_encoder"].transform(df_clean["Target"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_test, y_test


df_raw = get_raw_data()
artefacts = charger_artefacts()

if "historique_predictions" not in st.session_state:
    st.session_state["historique_predictions"] = charger_historique_persistant(HISTORY_PATH).to_dict("records")

model = artefacts["model"] if artefacts else None
scaler = artefacts["scaler"] if artefacts else None
le_target = artefacts["label_encoder"] if artefacts else None
feature_columns = artefacts["feature_columns"] if artefacts else None
top_localisations = artefacts["top_localisations"] if artefacts else None
best_threshold_f1 = artefacts["best_threshold_f1"] if artefacts else 0.5


# ----------------------------------------------------------------------------
# Barre latérale / navigation
# ----------------------------------------------------------------------------
render_sidebar_brand()

pages = [
    st.Page(
        lambda: dashboard.render(df_raw, artefacts, model, theme_actif),
        title="Dashboard", icon=":material/dashboard:", url_path="dashboard", default=True,
    ),
    st.Page(
        lambda: data_import.render(),
        title="Ingestion des données", icon=":material/database:", url_path="import-donnees",
    ),
    st.Page(
        lambda: eda.render(df_raw, theme_actif),
        title="Analyse exploratoire", icon=":material/query_stats:", url_path="exploration",
    ),
    st.Page(
        lambda: training.render(artefacts, model, theme_actif, TRAINING_HISTORY_PATH),
        title="Modélisation", icon=":material/model_training:", url_path="entrainement",
    ),
    st.Page(
        lambda: evaluation.render(
            artefacts, model, feature_columns, top_localisations, scaler, le_target,
            best_threshold_f1, get_test_split, theme_actif,
        ),
        title="Performance du modèle", icon=":material/insights:", url_path="evaluation",
    ),
    st.Page(
        lambda: prediction.render(
            df_raw, artefacts, model, scaler, le_target, feature_columns, top_localisations,
            best_threshold_f1, preparer_transaction, HISTORY_PATH, theme_actif,
        ),
        title="Détection en temps réel", icon=":material/online_prediction:", url_path="prediction",
    ),
    st.Page(
        lambda: reports.render(HISTORY_PATH, theme_actif),
        title="Journal des prédictions", icon=":material/history:", url_path="historique",
    ),
]
pg = st.navigation(pages, position="hidden")

st.sidebar.markdown('<div class="kt-sidebar-spacer"></div>', unsafe_allow_html=True)
for page in pages:
    st.sidebar.page_link(page)

st.sidebar.markdown("---")
st.sidebar.caption(f":material/account_circle: Connecté en tant que **{nom_utilisateur}**")
render_bouton_deconnexion()
render_theme_switcher()
st.sidebar.markdown(
    '<p class="sidebar-footer-credit">Samba Bery KANE @ DIT Master IA</p>',
    unsafe_allow_html=True,
)

if df_raw is None:
    st.error(
        "Fichier de données introuvable : `data/ktech_bank_transaction_dataset.csv`. "
        "Placez-le dans le dossier `data/`, ou chargez votre propre fichier depuis la page "
        "*Ingestion des données*."
    )
    st.stop()

if artefacts is None:
    st.warning(
        "Aucun modèle entraîné n'est disponible pour le moment. "
        "Les pages *Dashboard* et *Analyse exploratoire* restent disponibles en attendant."
    )
    bouton_entrainer_maintenant("train_now_main")

pg.run()
