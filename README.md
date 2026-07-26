<div align="center">

# Fraude Bancaire — Détection & Scoring en Temps Réel

**Une application Streamlit qui transforme un pipeline de machine learning en outil de décision opérationnel.**

Réalisé par **Samba Bery KANE**

</div>

---

## Sommaire

- [En bref](#en-bref)
- [Ce que l'application permet de faire](#ce-que-lapplication-permet-de-faire)
- [Sous le capot : la stack](#sous-le-capot--la-stack)
- [Démarrage rapide](#démarrage-rapide)
- [Authentification](#authentification)
- [Organisation du code](#organisation-du-code)
- [Le pipeline de machine learning, en détail](#le-pipeline-de-machine-learning-en-détail)
- [Déploiement avec Docker](#déploiement-avec-docker)
- [Format de données attendu](#format-de-données-attendu)
- [Limites connues & pistes d'évolution](#limites-connues--pistes-dévolution)
- [Auteur](#auteur)

---

## En bref

Une banque enregistre des milliers de transactions par jour ; une poignée seulement sont
frauduleuses. Ce projet part d'un notebook d'exploration (`detection_fraude_bancaire.ipynb`) et le
transforme en une véritable application : un pipeline d'entraînement reproductible d'un côté
(`ml/`), une interface Streamlit de l'autre (`app/`), reliées par une couche de prétraitement
partagée qui garantit qu'une transaction saisie à la main est traitée exactement comme les données
d'entraînement.

Le problème central adressé : la classe **Fraude** ne représente qu'environ **3,7 %** des
transactions. Tout, dans ce projet — pondération des modèles, choix de la métrique d'optimisation,
calibrage du seuil de décision — est pensé autour de ce déséquilibre.

## Ce que l'application permet de faire

| Page | Ce qu'on y trouve |
|---|---|
| **Dashboard** | KPIs clés (volume, taux de fraude, montants, clients uniques), tendance temporelle, répartition globale des statuts. |
| **Ingestion des données** | Chargement d'un CSV/Excel personnel, diagnostic qualité (valeurs manquantes, doublons, mémoire). |
| **Analyse exploratoire** | Vue d'ensemble, explorateur univarié, relations avec la cible, localisation, saisonnalité (heure/jour de semaine), corrélations — organisés en onglets. |
| **Modélisation** | Comparatif de 6 modèles supervisés (filtrable), détecteur d'anomalies non supervisé (Isolation Forest), et suivi des performances d'entraînement dans le temps — organisés en onglets. |
| **Performance du modèle** | Matrice de confusion, courbes ROC / précision-rappel, importance des variables, curseur interactif pour ajuster le seuil de décision. |
| **Détection en temps réel** | Formulaire de saisie d'une transaction → verdict, probabilités par classe, jauge de risque colorée. |
| **Journal des prédictions** | Historique des prédictions effectuées, export CSV / PDF / Excel. |

L'accès à l'application est protégé par authentification (voir [Authentification](#authentification)).

## Sous le capot : la stack

| Couche | Outils |
|---|---|
| Interface | Streamlit, Plotly |
| Authentification | streamlit-authenticator |
| Modélisation | scikit-learn, XGBoost, LightGBM |
| Données | pandas, numpy |
| Export | ReportLab (PDF), openpyxl (Excel) |
| Exécution | Docker / Docker Compose |

## Démarrage rapide

```bash
# 1. Environnement
python -m venv venv
source venv/bin/activate          # Windows : venv\Scripts\activate
pip install -r requirements.txt

# 2. Authentification (voir la section dédiée ci-dessous)
cp .streamlit/secrets.toml.example .streamlit/secrets.toml

# 3. Entraînement (génère models/fraud_model.pkl)
python ml/train.py

# 4. Lancement de l'application
streamlit run app/main.py
```

L'application démarre sur **http://localhost:8501**. Si aucun modèle n'est trouvé au lancement,
l'application l'entraîne automatiquement à la volée (utile pour un premier déploiement cloud sans
artefact commité) — mais lancer `ml/train.py` en amont reste recommandé pour garder la main sur le
processus. Chaque exécution de `ml/train.py` enregistre aussi une ligne dans
`data/training_runs_history.csv`, affichée dans l'onglet "Suivi dans le temps" de la page
*Modélisation* — relancez l'entraînement plusieurs fois pour voir apparaître une tendance.

## Authentification

L'accès à l'application est protégé par [`streamlit-authenticator`](https://github.com/mkhorasani/Streamlit-Authenticator).
Les identifiants et la configuration du cookie de session vivent dans `.streamlit/secrets.toml`
(jamais commité ni copié dans l'image Docker — voir `.gitignore`/`.dockerignore`).

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Un identifiant de démonstration est fourni par défaut : `admin` / `changeme123`. Pour ajouter un
utilisateur ou changer ce mot de passe, générez un nouveau hash bcrypt puis modifiez
`.streamlit/secrets.toml` :

```bash
python -c "import streamlit_authenticator as stauth; print(stauth.Hasher.hash('votre_mot_de_passe'))"
```

> Limite connue : pas de flux self-service de réinitialisation de mot de passe — adapté à un
> usage démo ou en équipe restreinte, pas à un produit grand public.

## Organisation du code

Le projet sépare nettement trois responsabilités : la **modélisation** (`ml/`), l'**interface**
(`app/`), et le **socle partagé** (`common/`) utilisé par les deux.

```
ml/            → pipeline d'entraînement, indépendant de Streamlit
  preprocessing.py    fonctions de transformation partagées entraînement/inférence
  train.py            script d'entraînement (comparaison de modèles + Isolation Forest)

models/        → artefacts générés (non versionnés)
  fraud_model.pkl

common/        → briques réutilisées par toutes les pages
  theme.py             palette, typographies, composants visuels
  auth.py              authentification (streamlit-authenticator)
  data_io.py           import CSV/Excel
  history_store.py     persistance de l'historique des prédictions + génération de rapports
  training_history.py  persistance de l'historique des runs d'entraînement

app/           → application Streamlit
  main.py             point d'entrée : navigation, cache, orchestration
  pages/
    dashboard.py, data_import.py, eda.py, training.py,
    evaluation.py, prediction.py, reports.py

data/          → jeu de données de référence + historique généré à l'exécution
```

Le point clé de cette organisation : `ml/preprocessing.py` est importé à la fois par
`ml/train.py` et par `app/main.py`. Aucune transformation n'est jamais dupliquée entre
l'entraînement et l'inférence — c'est ce qui évite le décalage classique entre "ce que le modèle a
appris" et "ce que l'app lui envoie en production".

## Le pipeline de machine learning, en détail

`python ml/train.py` exécute, dans l'ordre :

1. **Chargement** de `data/ktech_bank_transaction_dataset.csv`.
2. **Feature engineering** : variables temporelles (heure, jour de semaine, nuit, week-end),
   passage du montant en log, fréquence de transactions par client, regroupement des localisations
   rares.
3. **Comparaison de 6 modèles supervisés** — Régression Logistique, Arbre de décision, Random
   Forest, Gradient Boosting, XGBoost, LightGBM — tous pondérés `balanced` pour compenser la rareté
   de la fraude.
4. **Isolation Forest**, en parallèle, entraîné *sans jamais voir la variable cible* : un détecteur
   d'anomalies statistiques purement non supervisé, complémentaire aux modèles ci-dessus pour
   repérer des schémas de fraude inédits.
5. **Sélection automatique** du modèle supervisé au meilleur rappel sur la classe Fraude, puis
   **optimisation par `GridSearchCV`** (validation croisée à 5 plis, `scoring="recall_macro"`).
6. **Calibrage du seuil de décision** : plutôt que le seuil implicite de 0,5, un seuil optimisant le
   F1-score de la classe Fraude est calculé et proposé par défaut dans l'app (ajustable via curseur).
7. **Sauvegarde** de tout — modèle, scaler, encodeur, comparatif des modèles, métriques — dans
   `models/fraud_model.pkl`.

> XGBoost et LightGBM sont optionnels : absents, le script continue avec un avertissement et les
> modèles restants.

## Déploiement avec Docker

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # requis (voir Authentification)
docker compose up --build
```

L'image entraîne le modèle **pendant le build** (voir `Dockerfile`) : pas besoin de lancer
`ml/train.py` à part. Pour ré-entraîner après avoir modifié le CSV de données :

```bash
docker compose up --build --force-recreate
```

Sans Compose :

```bash
docker build -t fraud-detection-app .
docker run -p 8501:8501 \
  -v "$(pwd)/.streamlit/secrets.toml:/app/.streamlit/secrets.toml:ro" \
  --name fraud_detection_app fraud-detection-app
```

Le `Dockerfile` est multi-stage (image finale sans toolchain de compilation) et tourne en
utilisateur non-root. Le port est piloté par la variable d'environnement `PORT` (défaut `8501`),
et l'historique des prédictions/entraînements est séparé du dataset de référence via
`FRAUD_APP_PERSIST_DIR` (défaut `data/` en local ; pointé vers un volume monté en production —
voir `docker-compose.yml`).

## Format de données attendu

CSV avec **point-virgule** (`;`) comme séparateur, colonnes minimales :

| Colonne | Description |
|---|---|
| `ID Clients` | Identifiant du client |
| `Numero de compte` | Numéro de compte bancaire |
| `Identifiant operation` | Identifiant unique de la transaction |
| `Type de transaction` | ATM / Paiement en ligne / Paiement électronique |
| `Status operation` | Validé / Echoué / En attente |
| `Localisation` | Ville de la transaction |
| `Date` | Date et heure de la transaction |
| `Montant` | Montant de la transaction |
| `Target` | Normal / Suspect / Fraude (variable cible) |

Le jeu de données par défaut (`data/ktech_bank_transaction_dataset.csv`) respecte déjà ce schéma ;
la page *Ingestion des données* permet de charger votre propre fichier au même format.

## Limites connues & pistes d'évolution

- L'historique des prédictions et des runs d'entraînement vit sur le disque du conteneur (ou du
  volume `FRAUD_APP_PERSIST_DIR` monté) : sur une plateforme sans disque persistant configuré
  (ex. Streamlit Community Cloud), il est réinitialisé à chaque redéploiement.
- Authentification adaptée à un usage démo/équipe restreinte : pas de flux self-service de
  réinitialisation de mot de passe ni de gestion des rôles.
- Le suivi des performances dans le temps ne montre une tendance qu'après plusieurs
  ré-entraînements — un seul run enregistré ne permet pas encore de comparaison.
- Isolation Forest est évalué à l'entraînement mais pas encore branché sur la page de prédiction en
  temps réel.
- Pistes pour aller plus loin : rééquilibrage par SMOTE, autres détecteurs d'anomalies (One-Class
  SVM, Local Outlier Factor, autoencodeur), variables comportementales enrichies par client
  (vélocité, écarts au comportement habituel), ré-entraînement planifié en pipeline MLOps.

## Auteur

**Samba Bery KANE**
