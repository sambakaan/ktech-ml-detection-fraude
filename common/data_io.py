"""
Chargement des données (CSV / Excel) et génération d'un jeu de données synthétique,
pour la page "Import & Gestion des données".
"""
import numpy as np
import pandas as pd


def charger_fichier_transactions(fichier, sep_csv: str = ";") -> pd.DataFrame:
    """Charge un fichier CSV ou Excel (upload Streamlit ou chemin disque) en DataFrame.

    Le CSV de référence de ce projet utilise le point-virgule comme séparateur ; on l'essaie
    en premier puis on retente avec la virgule si le résultat ne comporte qu'une seule colonne
    (signe que le séparateur ne correspond pas).
    """
    nom = getattr(fichier, "name", str(fichier)).lower()

    if nom.endswith((".xlsx", ".xls")):
        return pd.read_excel(fichier)

    df = pd.read_csv(fichier, sep=sep_csv)
    if df.shape[1] == 1:
        if hasattr(fichier, "seek"):
            fichier.seek(0)
        df = pd.read_csv(fichier, sep=",")
    return df


def generer_donnees_synthetiques(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Génère un jeu de données synthétique plausible, avec le même schéma que
    `ktech_bank_transaction_dataset.csv`, pour tester l'application sans fichier réel."""
    rng = np.random.default_rng(seed)

    types_transaction = ["ATM", "Paiement en ligne", "Paiement électronique"]
    statuts_operation = ["Validé", "Echoué", "En attente"]
    localisations = [
        "Dakar", "Thiès", "Kaolack", "Saint Louis", "Tambacounda",
        "Ziguinchor", "Louga", "Fatick", "Kolda", "Matam",
    ]

    id_clients = rng.integers(1000, 1200, size=n)
    dates = (
        pd.Timestamp("2025-01-01")
        + pd.to_timedelta(rng.integers(0, 200, size=n), unit="D")
        + pd.to_timedelta(rng.integers(0, 86400, size=n), unit="s")
    )
    montants = np.round(rng.lognormal(mean=10.5, sigma=1.1, size=n), 0)

    # Répartition Normal / Suspect / Fraude proche du jeu de données réel (~76% / 20% / 4%)
    tirage = rng.random(n)
    target = np.where(tirage < 0.037, "Fraude", np.where(tirage < 0.24, "Suspect", "Normal"))

    df = pd.DataFrame({
        "ID Clients": id_clients,
        "Numero de compte": rng.integers(100000, 999999, size=n),
        "Identifiant operation": [f"TR{rng.integers(100, 999)}/{i:04d}" for i in range(n)],
        "Type de transaction": rng.choice(types_transaction, size=n),
        "Status operation": rng.choice(statuts_operation, size=n),
        "Localisation": rng.choice(localisations, size=n),
        "Date": dates,
        "Montant": montants,
        "Target": target,
    })
    return df


def diagnostiquer_dataframe(df: pd.DataFrame) -> dict:
    """Calcule les indicateurs de santé d'un DataFrame (pour la page Import & Gestion)."""
    return {
        "n_lignes": df.shape[0],
        "n_colonnes": df.shape[1],
        "memoire_mo": round(df.memory_usage(deep=True).sum() / (1024 ** 2), 2),
        "valeurs_manquantes": int(df.isnull().sum().sum()),
        "doublons": int(df.duplicated().sum()),
        "types": df.dtypes.astype(str).to_dict(),
    }
