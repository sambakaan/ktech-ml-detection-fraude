"""
Calculs mis en cache et composants réutilisables pour la page "Analyse exploratoire"
(et potentiellement d'autres pages qui voudraient un aperçu univarié rapide).
"""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from common.charts import couleurs_target, styliser_figure

COLONNES_HAUTE_CARDINALITE = {"Identifiant operation", "Numero de compte"}
SEUIL_CARDINALITE_AFFICHAGE = 30
JOURS_SEMAINE = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]


@st.cache_data(show_spinner=False)
def calculer_kpis(df: pd.DataFrame) -> dict:
    """KPIs de synthèse sur le jeu de données (filtré ou non)."""
    dates = pd.to_datetime(df["Date"], errors="coerce") if "Date" in df.columns else pd.Series(dtype="datetime64[ns]")
    return {
        "n_transactions": int(df.shape[0]),
        "taux_fraude": float((df["Target"] == "Fraude").mean() * 100) if len(df) else 0.0,
        "taux_suspect": float((df["Target"] == "Suspect").mean() * 100) if len(df) else 0.0,
        "montant_total": float(df["Montant"].sum()) if "Montant" in df.columns else 0.0,
        "date_min": dates.min() if dates.notna().any() else None,
        "date_max": dates.max() if dates.notna().any() else None,
    }


@st.cache_data(show_spinner=False)
def repartition_cible(df: pd.DataFrame) -> pd.DataFrame:
    """Nombre et pourcentage de transactions par statut (Normal/Suspect/Fraude)."""
    counts = df["Target"].value_counts().reindex(["Normal", "Suspect", "Fraude"]).fillna(0)
    return pd.DataFrame({
        "Statut": counts.index,
        "Nombre": counts.values.astype(int),
        "Pourcentage": (counts.values / max(counts.values.sum(), 1) * 100).round(1),
    })


@st.cache_data(show_spinner=False)
def taux_par_localisation(df: pd.DataFrame, nb_top: int) -> pd.DataFrame:
    """Taux (%) de chaque statut par localisation, limité aux `nb_top` localisations
    les plus actives (le sujet actif reste passé en paramètre pour ne pas appeler un
    widget Streamlit à l'intérieur d'une fonction mise en cache)."""
    top_loc = df["Localisation"].value_counts().head(nb_top).index
    sous_df = df[df["Localisation"].isin(top_loc)]
    if sous_df.empty:
        return pd.DataFrame()
    ct = pd.crosstab(sous_df["Localisation"], sous_df["Target"], normalize="index").reindex(top_loc) * 100
    return ct.reset_index()


@st.cache_data(show_spinner=False)
def volume_localisation_type(df: pd.DataFrame, nb_top: int) -> pd.DataFrame:
    """Volume de transactions par localisation × type, limité aux `nb_top` localisations."""
    top_loc = df["Localisation"].value_counts().head(nb_top).index
    sous_df = df[df["Localisation"].isin(top_loc)]
    if sous_df.empty:
        return pd.DataFrame()
    return sous_df.groupby(["Localisation", "Type de transaction"]).size().reset_index(name="Nombre")


@st.cache_data(show_spinner=False)
def statut_par_type(df: pd.DataFrame) -> pd.DataFrame:
    """Volume de transactions par type, réparti par statut."""
    return df.groupby(["Type de transaction", "Target"]).size().reset_index(name="Nombre")


@st.cache_data(show_spinner=False)
def agregat_temporel(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retourne (taux par heure, taux par jour de semaine) : % de chaque statut."""
    df_temps = df.copy()
    df_temps["heure"] = pd.to_datetime(df_temps["Date"], errors="coerce").dt.hour
    df_temps["jour_semaine"] = pd.to_datetime(df_temps["Date"], errors="coerce").dt.dayofweek

    par_heure = pd.DataFrame()
    if df_temps["heure"].notna().any():
        par_heure = (pd.crosstab(df_temps["heure"], df_temps["Target"], normalize="index") * 100).reset_index()

    par_jour = pd.DataFrame()
    if df_temps["jour_semaine"].notna().any():
        ct = pd.crosstab(df_temps["jour_semaine"], df_temps["Target"], normalize="index") * 100
        ct.index = ct.index.map(lambda i: JOURS_SEMAINE[int(i)] if 0 <= i <= 6 else str(i))
        ct.index.name = "Jour"
        par_jour = ct.reindex(JOURS_SEMAINE).reset_index()

    return par_heure, par_jour


@st.cache_data(show_spinner=False)
def matrice_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Matrice de corrélation enrichie : montant (brut + log), composantes temporelles,
    fréquence client, et deux indicateurs binaires liés à la cible (plus interprétables en
    corrélation linéaire qu'un encodage ordinal Normal/Suspect/Fraude)."""
    dates = pd.to_datetime(df["Date"], errors="coerce")
    df_num = pd.DataFrame(index=df.index)
    df_num["Montant"] = df["Montant"]
    df_num["Montant_log"] = np.log1p(df["Montant"].clip(lower=0))
    df_num["heure"] = dates.dt.hour
    df_num["jour_semaine"] = dates.dt.dayofweek
    df_num["est_weekend"] = df_num["jour_semaine"].isin([5, 6]).astype(int)
    if "ID Clients" in df.columns:
        df_num["nb_transactions_client"] = df.groupby("ID Clients")["ID Clients"].transform("count")
    df_num["est_fraude"] = (df["Target"] == "Fraude").astype(int)
    df_num["est_suspect_ou_fraude"] = df["Target"].isin(["Suspect", "Fraude"]).astype(int)
    return df_num.corr()


def explorateur_univarie(df: pd.DataFrame, theme_actif: str, colonne_couleur: str = "Target",
                          key_prefix: str = "eda_univ"):
    """Composant générique : l'utilisateur choisit une colonne, on affiche automatiquement sa
    distribution (histogramme si numérique/datetime, bar chart de fréquence si catégorielle,
    tronqué au-delà d'un seuil de cardinalité) + ses statistiques descriptives à côté."""
    colonnes = [c for c in df.columns if c not in COLONNES_HAUTE_CARDINALITE]
    col_choisie = st.selectbox("Variable à explorer", colonnes, key=f"{key_prefix}_col")

    est_numerique = pd.api.types.is_numeric_dtype(df[col_choisie])
    est_datetime = pd.api.types.is_datetime64_any_dtype(df[col_choisie])
    couleur = colonne_couleur if (colonne_couleur and colonne_couleur != col_choisie
                                   and colonne_couleur in df.columns) else None

    col_graph, col_stats = st.columns([2, 1])
    with col_graph:
        if est_numerique or est_datetime:
            fig = px.histogram(
                df, x=col_choisie, color=couleur,
                color_discrete_map=couleurs_target(theme_actif) if couleur == "Target" else None,
                title=f"Distribution — {col_choisie}",
            )
            styliser_figure(fig, theme_actif)
            st.plotly_chart(fig, use_container_width=True)
        else:
            frequence = df[col_choisie].value_counts().rename_axis(col_choisie).reset_index(name="Nombre")
            tronque = len(frequence) > SEUIL_CARDINALITE_AFFICHAGE
            if tronque:
                frequence = frequence.head(SEUIL_CARDINALITE_AFFICHAGE)
            fig = px.bar(frequence, x=col_choisie, y="Nombre", title=f"Fréquence — {col_choisie}")
            styliser_figure(fig, theme_actif, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            if tronque:
                st.caption(f"Limité aux {SEUIL_CARDINALITE_AFFICHAGE} valeurs les plus fréquentes.")

    with col_stats:
        if est_numerique:
            st.dataframe(df[col_choisie].describe().to_frame("Valeur"), use_container_width=True)
        else:
            st.dataframe(
                df[col_choisie].value_counts().rename_axis(col_choisie).reset_index(name="Nombre"),
                hide_index=True, use_container_width=True, height=280,
            )
