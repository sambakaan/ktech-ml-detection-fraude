"""Page 2 : Import & Gestion des données."""
import streamlit as st

from common.theme import render_hero
from common.data_io import charger_fichier_transactions, diagnostiquer_dataframe

COLONNES_ATTENDUES = [
    "ID Clients", "Type de transaction", "Status operation",
    "Localisation", "Date", "Montant", "Target",
]


def render():
    render_hero(
        "database", "Source des données", "Ingestion des données",
        "Chargez votre propre fichier pour l'utiliser dans l'application.",
    )

    utilise_defaut = "df_actif" not in st.session_state or st.session_state["df_actif"] is None
    if utilise_defaut:
        st.info("L'application utilise actuellement le jeu de données par défaut "
                "(`data/ktech_bank_transaction_dataset.csv`).")
    else:
        st.success(f"Jeu de données actif : **{st.session_state.get('df_actif_nom', 'personnalisé')}** "
                   f"({st.session_state['df_actif'].shape[0]} lignes).")

    st.caption(
        "Le modèle prédictif déjà entraîné reste basé sur le schéma du jeu de données d'origine. "
        "Changer la source ici met à jour les pages *Dashboard* et *Analyse exploratoire* ; "
        "pour ré-entraîner le modèle sur ces nouvelles données, relancez `python ml/train.py` "
        "après avoir remplacé `data/ktech_bank_transaction_dataset.csv`."
    )

    st.subheader(":material/upload_file: Charger un fichier")
    fichier = st.file_uploader("CSV (séparateur `;`) ou Excel (.xlsx)", type=["csv", "xlsx", "xls"])
    if fichier is not None:
        try:
            with st.spinner("Lecture du fichier..."):
                df_upload = charger_fichier_transactions(fichier)
            colonnes_manquantes = [c for c in COLONNES_ATTENDUES if c not in df_upload.columns]
            if colonnes_manquantes:
                st.error(f"Colonnes manquantes par rapport au schéma attendu : {colonnes_manquantes}")
            else:
                if st.button(":material/check_circle: Utiliser ce fichier", use_container_width=True):
                    st.session_state["df_actif"] = df_upload
                    st.session_state["df_actif_nom"] = fichier.name
                    st.rerun()
        except Exception as e:
            st.error(f"Erreur de lecture du fichier : {e}")

    if not utilise_defaut:
        if st.button(":material/restart_alt: Revenir au jeu de données par défaut"):
            st.session_state["df_actif"] = None
            st.session_state["df_actif_nom"] = None
            st.rerun()

    st.markdown("---")
    st.subheader(":material/troubleshoot: Diagnostic du jeu de données actif")

    df_diag = st.session_state.get("df_actif")
    if df_diag is not None:
        diag = diagnostiquer_dataframe(df_diag)
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Lignes", f"{diag['n_lignes']:,}".replace(",", " "))
        col2.metric("Colonnes", diag["n_colonnes"])
        col3.metric("Mémoire", f"{diag['memoire_mo']} Mo")
        col4.metric("Valeurs manquantes", diag["valeurs_manquantes"])
        col5.metric("Doublons", diag["doublons"])

        with st.expander("Types de colonnes"):
            st.json(diag["types"])

        with st.expander("Aperçu des données (20 premières lignes)"):
            st.dataframe(df_diag.head(20), use_container_width=True)
    else:
        st.caption("Le diagnostic complet du jeu de données par défaut est disponible sur la page "
                   "*Analyse exploratoire*.")
