"""
Script d'entraînement du modèle de détection de fraude bancaire.

Reprend et enrichit le pipeline développé dans le notebook
`detection_fraude_bancaire.ipynb` :
    - feature engineering
    - comparaison de 6 modèles supervisés : Régression Logistique, Arbre de
      décision, Random Forest, Gradient Boosting, XGBoost, LightGBM
    - détection non supervisée complémentaire avec Isolation Forest
      (utile pour repérer des schémas de fraude inédits, jamais vus à
      l'entraînement, puisqu'elle n'utilise pas la variable cible)
    - sélection automatique du meilleur modèle supervisé (rappel de la
      classe "Fraude") puis optimisation par GridSearchCV
    - calibrage d'un seuil de décision optimal (F1) sur la classe "Fraude"

Usage (depuis la racine du projet) :
    python ml/train.py

Génère `models/fraud_model.pkl`, utilisé ensuite par `app/main.py`.
"""
import os
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.ensemble import GradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import CLASSES_CIBLE, charger_donnees, encoder_features, nettoyer_et_enrichir

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from common.training_history import enregistrer_run_entrainement  # noqa: E402

CSV_PATH = os.path.join(BASE_DIR, "data", "ktech_bank_transaction_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "fraud_model.pkl")

# Répertoire de persistance des données évolutives (historique d'entraînement, historique de
# prédictions) — distinct du dataset de référence, qui doit rester embarqué dans l'image Docker.
# En local, vaut data/ (comportement inchangé) ; en production, pointé vers un volume monté
# via la variable d'environnement FRAUD_APP_PERSIST_DIR (voir Dockerfile/docker-compose.yml).
PERSIST_DIR = os.environ.get("FRAUD_APP_PERSIST_DIR", os.path.join(BASE_DIR, "data"))
TRAINING_HISTORY_PATH = os.path.join(PERSIST_DIR, "training_runs_history.csv")

# ------------------------------------------------------------------------
# Modèles optionnels (installés via requirements.txt). Le script continue
# de fonctionner même s'ils sont absents, avec un avertissement.
# ------------------------------------------------------------------------
try:
    from xgboost import XGBClassifier
    XGBOOST_DISPONIBLE = True
except ImportError:
    XGBOOST_DISPONIBLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_DISPONIBLE = True
except ImportError:
    LIGHTGBM_DISPONIBLE = False


# ------------------------------------------------------------------------
# Fabriques d'estimateurs "vierges" (nécessaire pour ré-instancier le
# modèle gagnant avant de lancer GridSearchCV dessus)
# ------------------------------------------------------------------------
def get_estimateurs():
    estimateurs = {
        "Régression Logistique": lambda: LogisticRegression(max_iter=5000, random_state=42),
        "Arbre de décision": lambda: DecisionTreeClassifier(random_state=42),
        "Random Forest": lambda: RandomForestClassifier(random_state=42),
        "Gradient Boosting": lambda: GradientBoostingClassifier(random_state=42),
    }
    if XGBOOST_DISPONIBLE:
        estimateurs["XGBoost"] = lambda: XGBClassifier(eval_metric="mlogloss", random_state=42)
    if LIGHTGBM_DISPONIBLE:
        estimateurs["LightGBM"] = lambda: LGBMClassifier(random_state=42, verbosity=-1)
    return estimateurs


# Grilles d'hyperparamètres utilisées lors de l'optimisation du modèle sélectionné
PARAM_GRIDS = {
    "Régression Logistique": {"C": [0.01, 0.1, 1, 10]},
    "Arbre de décision": {
        "max_depth": [None, 5, 10, 15, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "Random Forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 8, 12, 20],
        "min_samples_split": [2, 5, 10],
    },
    "Gradient Boosting": {
        "n_estimators": [100, 200, 300],
        "max_depth": [2, 3, 4],
        "learning_rate": [0.05, 0.1, 0.2],
    },
    "XGBoost": {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1, 0.2],
    },
    "LightGBM": {
        "n_estimators": [100, 200, 300],
        "max_depth": [-1, 5, 10],
        "learning_rate": [0.05, 0.1, 0.2],
    },
}


def main():
    debut_entrainement = time.time()
    print("=" * 70)
    print("ENTRAÎNEMENT DU MODÈLE DE DÉTECTION DE FRAUDE BANCAIRE")
    print("=" * 70)
    if not XGBOOST_DISPONIBLE:
        print("Avertissement : xgboost n'est pas installé (pip install xgboost) : modèle ignoré.")
    if not LIGHTGBM_DISPONIBLE:
        print("Avertissement : lightgbm n'est pas installé (pip install lightgbm) : modèle ignoré.")

    # ---------- 1. Chargement ----------
    print("\n[1/7] Chargement des données...")
    df = charger_donnees(CSV_PATH)
    print(f"      {df.shape[0]} transactions, {df.shape[1]} colonnes brutes.")

    # ---------- 2. Feature engineering ----------
    print("\n[2/7] Feature engineering...")
    df_clean, top_localisations = nettoyer_et_enrichir(df)

    le_target = LabelEncoder()
    le_target.fit(CLASSES_CIBLE)
    y = le_target.transform(df_clean["Target"])
    label_fraude = le_target.transform(["Fraude"])[0]

    X, feature_columns = encoder_features(df_clean)
    print(f"      {X.shape[1]} variables explicatives après encodage.")

    # ---------- 3. Split + standardisation ----------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"      Train : {X_train.shape[0]} lignes | Test : {X_test.shape[0]} lignes")

    # Pondération des classes (utilisée pour tous les modèles, y compris ceux
    # sans paramètre class_weight natif comme XGBoost) afin de compenser la
    # rareté de la classe Fraude (~3,7%).
    sample_weight_train = compute_sample_weight("balanced", y_train)

    # ---------- 4. Comparaison des modèles supervisés ----------
    print("\n[3/7] Comparaison des modèles supervisés (pondération 'balanced')...")
    estimateurs = get_estimateurs()
    resultats = []
    modeles_entraines = {}

    for nom, fabrique in estimateurs.items():
        modele = fabrique()
        modele.fit(X_train_scaled, y_train, sample_weight=sample_weight_train)
        modeles_entraines[nom] = modele

        y_pred = modele.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        rappel_fraude = recall_score(y_test, y_pred, labels=[label_fraude], average="macro")
        precision_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rappel_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

        resultats.append({
            "Modèle": nom,
            "Accuracy": round(acc, 4),
            "Précision (macro)": round(precision_macro, 4),
            "Rappel (macro)": round(rappel_macro, 4),
            "F1-score (macro)": round(f1_macro, 4),
            "Rappel classe Fraude": round(rappel_fraude, 4),
        })
        print(f"      {nom:25s} accuracy={acc:.3f}  rappel(Fraude)={rappel_fraude:.3f}")

    resultats.sort(key=lambda r: r["Rappel classe Fraude"], reverse=True)

    # ---------- 5. Détection non supervisée (Isolation Forest) ----------
    print("\n[4/7] Détection non supervisée avec Isolation Forest (n'utilise pas la cible)...")
    contamination = float((y_train == label_fraude).mean())
    iso_forest = IsolationForest(
        n_estimators=300, contamination=contamination, random_state=42
    )
    iso_forest.fit(X_train_scaled)  # entraînement non supervisé : aucune utilisation de y_train

    pred_iso = iso_forest.predict(X_test_scaled)          # -1 = anomalie, 1 = normal
    pred_iso_bin = (pred_iso == -1).astype(int)            # 1 = anomalie détectée (potentielle fraude)
    y_test_bin_pour_iso = (y_test == label_fraude).astype(int)

    metriques_isolation_forest = {
        "contamination": round(contamination, 4),
        "precision": round(precision_score(y_test_bin_pour_iso, pred_iso_bin, zero_division=0), 4),
        "rappel": round(recall_score(y_test_bin_pour_iso, pred_iso_bin, zero_division=0), 4),
        "f1": round(f1_score(y_test_bin_pour_iso, pred_iso_bin, zero_division=0), 4),
    }
    print(f"      Isolation Forest (anomalie vs Fraude réelle) : "
          f"précision={metriques_isolation_forest['precision']:.3f}  "
          f"rappel={metriques_isolation_forest['rappel']:.3f}  "
          f"f1={metriques_isolation_forest['f1']:.3f}")
    print("      Note : Isolation Forest est non supervisé : il détecte des anomalies statistiques, "
          "pas spécifiquement des fraudes. Ses résultats sont donc rapportés séparément, à titre "
          "complémentaire des modèles supervisés ci-dessus.")

    # ---------- 6. Sélection et optimisation du meilleur modèle supervisé ----------
    meilleur_nom = resultats[0]["Modèle"]
    print(f"\n[5/7] Modèle supervisé sélectionné pour l'optimisation : {meilleur_nom} "
          f"(meilleur rappel sur la classe Fraude)")

    param_grid = PARAM_GRIDS.get(meilleur_nom, {})
    if param_grid:
        print("      Optimisation par GridSearchCV (cv=5, scoring='recall_macro')...")
        grid_search = GridSearchCV(
            estimateurs[meilleur_nom](),
            param_grid, cv=5, scoring="recall_macro", n_jobs=-1,
        )
        grid_search.fit(X_train_scaled, y_train, sample_weight=sample_weight_train)
        meilleur_modele = grid_search.best_estimator_
        meilleurs_params = grid_search.best_params_
        print(f"      Meilleurs paramètres : {meilleurs_params}")
    else:
        meilleur_modele = modeles_entraines[meilleur_nom]
        meilleurs_params = {}

    y_pred_final = meilleur_modele.predict(X_test_scaled)
    labels_ordre = le_target.inverse_transform(sorted(set(y))).tolist()
    rapport = classification_report(y_test, y_pred_final, target_names=labels_ordre, output_dict=True)
    print("\n[6/7] Rapport de classification (jeu de test, modèle optimisé) :")
    print(classification_report(y_test, y_pred_final, target_names=labels_ordre))

    # ---------- 7. Calibrage du seuil de décision (classe Fraude) ----------
    print("[7/7] Calibrage du seuil de décision optimal (F1, classe Fraude)...")
    if hasattr(meilleur_modele, "predict_proba"):
        idx_fraude = list(meilleur_modele.classes_).index(label_fraude)
        proba_fraude = meilleur_modele.predict_proba(X_test_scaled)[:, idx_fraude]
        y_test_bin_fraude = (y_test == label_fraude).astype(int)

        precisions, rappels, seuils = precision_recall_curve(y_test_bin_fraude, proba_fraude)
        if len(seuils) > 0:
            f1_scores = np.array([
                f1_score(y_test_bin_fraude, (proba_fraude >= s).astype(int)) for s in seuils
            ])
            meilleur_seuil_f1 = float(seuils[np.argmax(f1_scores)])
        else:
            meilleur_seuil_f1 = 0.5
    else:
        meilleur_seuil_f1 = 0.5
    print(f"      Seuil optimal (F1) : {meilleur_seuil_f1:.3f}")

    # ---------- 8. Sauvegarde des artefacts ----------
    print("\nSauvegarde du modèle et des artefacts...")
    metriques_test = {
        "accuracy": accuracy_score(y_test, y_pred_final),
        "rapport_classification": rapport,
        "matrice_confusion": confusion_matrix(y_test, y_pred_final).tolist(),
        "labels_ordre": labels_ordre,
        "n_test": int(X_test.shape[0]),
    }

    artefacts = {
        "model": meilleur_modele,
        "model_nom": meilleur_nom,
        "scaler": scaler,
        "label_encoder": le_target,
        "feature_columns": list(feature_columns),
        "top_localisations": list(top_localisations),
        "best_threshold_f1": meilleur_seuil_f1,
        "metriques_test": metriques_test,
        "grid_search_best_params": meilleurs_params,
        "comparaison_modeles": resultats,
        "metriques_isolation_forest": metriques_isolation_forest,
    }

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(artefacts, MODEL_PATH)
    print(f"Modèle sauvegardé : {MODEL_PATH}")

    ligne_historique = {
        "horodatage": pd.Timestamp.now().isoformat(),
        "model_nom": meilleur_nom,
        "accuracy": round(metriques_test["accuracy"], 4),
        "precision_fraude": round(rapport["Fraude"]["precision"], 4),
        "rappel_fraude": round(rapport["Fraude"]["recall"], 4),
        "f1_fraude": round(rapport["Fraude"]["f1-score"], 4),
        "rappel_macro": resultats[0]["Rappel (macro)"],
        "f1_macro": resultats[0]["F1-score (macro)"],
        "best_threshold_f1": round(meilleur_seuil_f1, 4),
        "grid_search_best_params": str(meilleurs_params),
        "n_test": metriques_test["n_test"],
        "duree_entrainement_sec": round(time.time() - debut_entrainement, 1),
    }
    enregistrer_run_entrainement(ligne_historique, TRAINING_HISTORY_PATH)
    print(f"Run d'entraînement enregistré : {TRAINING_HISTORY_PATH}")
    print("\nTerminé. Vous pouvez maintenant lancer l'application : streamlit run app/main.py")


if __name__ == "__main__":
    main()
