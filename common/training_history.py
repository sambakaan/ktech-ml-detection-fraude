"""
Historique des runs d'entraînement (un run = un appel à `ml/train.py::main()`).

Contrairement à `common/history_store.py`, ce module n'importe PAS `streamlit` : il est utilisé
par `ml/train.py`, qui s'exécute aussi en ligne de commande pure (y compris pendant le build
Docker), sans contexte Streamlit disponible.
"""
import os

import pandas as pd

COLONNES_HISTORIQUE = [
    "horodatage", "model_nom", "accuracy", "precision_fraude", "rappel_fraude", "f1_fraude",
    "rappel_macro", "f1_macro", "best_threshold_f1", "grid_search_best_params", "n_test",
    "duree_entrainement_sec",
]


def enregistrer_run_entrainement(ligne: dict, history_path: str):
    """Ajoute une ligne à l'historique des runs d'entraînement (append-only — contrairement à
    `models/fraud_model.pkl` qui est écrasé à chaque run, cet historique est cumulatif)."""
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    df_ligne = pd.DataFrame([ligne])
    fichier_existe = os.path.exists(history_path)
    df_ligne.to_csv(history_path, mode="a", header=not fichier_existe, index=False)


def charger_historique_entrainement(history_path: str) -> pd.DataFrame:
    """Charge l'historique des runs d'entraînement déjà enregistré sur disque (le cas échéant)."""
    if os.path.exists(history_path):
        try:
            return pd.read_csv(history_path, parse_dates=["horodatage"])
        except Exception:
            return pd.DataFrame(columns=COLONNES_HISTORIQUE)
    return pd.DataFrame(columns=COLONNES_HISTORIQUE)
