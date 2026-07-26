"""
Fonctions de prétraitement partagées entre l'entraînement (train_model.py)
et l'application Streamlit (app.py).

Centraliser cette logique garantit qu'une nouvelle transaction saisie dans
l'application est transformée EXACTEMENT de la même façon que les données
utilisées à l'entraînement du modèle.
"""
import numpy as np
import pandas as pd

# Colonnes catégorielles encodées en one-hot
COLONNES_CATEGORIELLES = ["Type de transaction", "Status operation", "Localisation_grp"]

# Colonnes numériques (brutes + dérivées)
COLONNES_NUMERIQUES = [
    "log_montant", "heure", "jour_semaine", "est_weekend", "est_nuit", "nb_transactions_client"
]

CLASSES_CIBLE = ["Normal", "Suspect", "Fraude"]


def charger_donnees(csv_path: str) -> pd.DataFrame:
    """Charge le fichier de transactions bancaires (séparateur ';')."""
    return pd.read_csv(csv_path, sep=";")


def nettoyer_et_enrichir(df: pd.DataFrame, top_localisations=None, nb_localisations: int = 15):
    """
    Nettoie le dataframe brut et ajoute les variables dérivées utilisées par le modèle :
    - variables temporelles (heure, jour de la semaine, week-end, nuit)
    - montant transformé en log
    - fréquence de transactions par client
    - regroupement des localisations rares sous 'Autre'

    Si `top_localisations` est fourni (cas de l'application, à l'inférence), les mêmes
    localisations que celles apprises à l'entraînement sont réutilisées.
    """
    df_clean = df.copy()

    # Uniformisation de quelques libellés de localisation (variantes d'écriture identiques)
    df_clean["Localisation"] = df_clean["Localisation"].replace({"Saint-Louis": "Saint Louis"})

    # Conversion de la date
    df_clean["Date"] = pd.to_datetime(df_clean["Date"], errors="coerce")

    # Variables temporelles
    df_clean["heure"] = df_clean["Date"].dt.hour
    df_clean["jour_semaine"] = df_clean["Date"].dt.dayofweek  # 0 = lundi ... 6 = dimanche
    df_clean["est_weekend"] = df_clean["jour_semaine"].isin([5, 6]).astype(int)
    df_clean["est_nuit"] = df_clean["heure"].apply(lambda h: 1 if (h >= 22 or h < 6) else 0)

    # Montant (transformation log pour réduire l'asymétrie de la distribution)
    df_clean["log_montant"] = np.log1p(df_clean["Montant"])

    # Fréquence du client dans le jeu de données
    df_clean["nb_transactions_client"] = df_clean.groupby("ID Clients")["Identifiant operation"].transform("count")

    # Regroupement des localisations rares
    if top_localisations is None:
        top_localisations = df_clean["Localisation"].value_counts().head(nb_localisations).index
    df_clean["Localisation_grp"] = df_clean["Localisation"].apply(
        lambda x: x if x in top_localisations else "Autre"
    )

    return df_clean, top_localisations


def encoder_features(df_clean: pd.DataFrame, feature_columns=None):
    """
    Encode les variables catégorielles en one-hot et sélectionne les colonnes finales.
    Si `feature_columns` est fourni, les colonnes sont réalignées dessus (cas de l'inférence).
    """
    df_encoded = pd.get_dummies(df_clean, columns=COLONNES_CATEGORIELLES, drop_first=True)
    colonnes_onehot = [
        c for c in df_encoded.columns if any(c.startswith(p + "_") for p in COLONNES_CATEGORIELLES)
    ]
    colonnes = COLONNES_NUMERIQUES + colonnes_onehot

    X = df_encoded[colonnes]
    if feature_columns is not None:
        X = X.reindex(columns=feature_columns, fill_value=0)
    return X, colonnes


def preparer_transaction(transaction: dict, top_localisations, feature_columns) -> pd.DataFrame:
    """
    Transforme une transaction brute (dictionnaire saisi dans l'application) en une ligne
    de features prête à être passée au scaler puis au modèle.

    transaction attend les clés :
        'Type de transaction', 'Status operation', 'Localisation', 'Date' (str ou datetime),
        'Montant', 'nb_transactions_client' (optionnel, défaut = 1)
    """
    date = pd.to_datetime(transaction["Date"])
    localisation = transaction["Localisation"] if transaction["Localisation"] in top_localisations else "Autre"

    ligne = {
        "log_montant": np.log1p(transaction["Montant"]),
        "heure": date.hour,
        "jour_semaine": date.dayofweek,
        "est_weekend": int(date.dayofweek in [5, 6]),
        "est_nuit": int(date.hour >= 22 or date.hour < 6),
        "nb_transactions_client": transaction.get("nb_transactions_client", 1),
        f"Type de transaction_{transaction['Type de transaction']}": 1,
        f"Status operation_{transaction['Status operation']}": 1,
        f"Localisation_grp_{localisation}": 1,
    }

    df_ligne = pd.DataFrame([ligne]).reindex(columns=feature_columns, fill_value=0)
    return df_ligne
