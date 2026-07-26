"""Pages 4-5 : Analyse exploratoire & Visualisations Plotly."""
import pandas as pd
import plotly.express as px
import streamlit as st

from common.charts import couleurs_target, styliser_figure
from common.data_io import diagnostiquer_dataframe
from common.eda_widgets import (
    agregat_temporel,
    calculer_kpis,
    explorateur_univarie,
    matrice_correlation,
    repartition_cible,
    statut_par_type,
    taux_par_localisation,
    volume_localisation_type,
)
from common.theme import colorer_statut_fn, render_hero


def render(df_raw: pd.DataFrame, theme_actif: str = "light"):
    render_hero(
        "query_stats", "Exploration des données", "Analyse exploratoire",
        "Vue d'ensemble, distributions, relations avec la fraude, localisation, saisonnalité et corrélations.",
    )

    # ---------------------------------------------------------------- Filtres (popover compact)
    df = df_raw.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    col_filtre, col_resume = st.columns([1, 4])
    with col_filtre:
        with st.popover(":material/tune: Filtres", use_container_width=True):
            n_filtres_actifs = 0
            if df["Date"].notna().any():
                date_min, date_max = df["Date"].min().date(), df["Date"].max().date()
                plage = st.date_input("Plage de dates", (date_min, date_max),
                                       min_value=date_min, max_value=date_max)
                if plage and isinstance(plage, tuple) and len(plage) == 2 and plage != (date_min, date_max):
                    n_filtres_actifs += 1
            else:
                plage = None

            montant_min, montant_max = float(df["Montant"].min()), float(df["Montant"].max())
            plage_montant = st.slider("Montant (FCFA)", montant_min, montant_max, (montant_min, montant_max))
            if plage_montant != (montant_min, montant_max):
                n_filtres_actifs += 1

            types_dispo = sorted(df["Type de transaction"].unique())
            types_sel = st.multiselect("Type de transaction", types_dispo, default=types_dispo)
            if len(types_sel) != len(types_dispo):
                n_filtres_actifs += 1

            statuts_dispo = sorted(df["Target"].unique())
            statuts_sel = st.multiselect("Statut (Target)", statuts_dispo, default=statuts_dispo)
            if len(statuts_sel) != len(statuts_dispo):
                n_filtres_actifs += 1

    if plage and isinstance(plage, tuple) and len(plage) == 2:
        df = df[(df["Date"].dt.date >= plage[0]) & (df["Date"].dt.date <= plage[1])]
    df = df[
        (df["Montant"] >= plage_montant[0]) & (df["Montant"] <= plage_montant[1])
        & (df["Type de transaction"].isin(types_sel))
        & (df["Target"].isin(statuts_sel))
    ]
    with col_resume:
        libelle_filtres = f"{n_filtres_actifs} filtre(s) actif(s)" if n_filtres_actifs else "Aucun filtre actif"
        st.caption(f"{libelle_filtres} — {df.shape[0]} / {df_raw.shape[0]} transactions affichées.")

    if df.empty:
        st.warning("Aucune transaction ne correspond à ces filtres.")
        return

    # ---------------------------------------------------------------- KPIs de synthèse
    kpis = calculer_kpis(df)
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Transactions", f"{kpis['n_transactions']:,}".replace(",", " "))
    col2.metric("Taux de Fraude", f"{kpis['taux_fraude']:.1f} %")
    col3.metric("Taux de Suspect", f"{kpis['taux_suspect']:.1f} %")
    col4.metric("Montant total", f"{kpis['montant_total']:,.0f} FCFA".replace(",", " "))
    periode = (f"{kpis['date_min']:%d/%m/%y} → {kpis['date_max']:%d/%m/%y}"
               if kpis["date_min"] is not None else "—")
    col5.metric("Période couverte", periode)

    st.markdown("---")

    onglet_vue, onglet_dist, onglet_cible, onglet_loc, onglet_temps, onglet_corr = st.tabs([
        ":material/dashboard: Vue d'ensemble",
        ":material/bar_chart: Distributions",
        ":material/insights: Relations avec la cible",
        ":material/location_on: Localisation",
        ":material/schedule: Temporel",
        ":material/scatter_plot: Corrélations",
    ])

    # ---------------------------------------------------------------- Vue d'ensemble
    with onglet_vue:
        with st.expander("Aperçu des données brutes", expanded=False):
            diag = diagnostiquer_dataframe(df)
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Lignes", f"{diag['n_lignes']:,}".replace(",", " "))
            c2.metric("Colonnes", diag["n_colonnes"])
            c3.metric("Mémoire", f"{diag['memoire_mo']} Mo")
            c4.metric("Valeurs manquantes", diag["valeurs_manquantes"])
            c5.metric("Doublons", diag["doublons"])
            st.dataframe(df.head(20), use_container_width=True)

        st.subheader("Répartition de la variable cible")
        col1, col2 = st.columns([1, 1.4])
        with col1:
            df_counts = repartition_cible(df)
            st.dataframe(
                df_counts.style.map(colorer_statut_fn(theme_actif), subset=["Statut"]),
                hide_index=True, use_container_width=True,
            )
        with col2:
            fig = px.bar(
                df_counts, x="Statut", y="Nombre", color="Statut",
                color_discrete_map=couleurs_target(theme_actif),
                title="Répartition des transactions par statut",
            )
            styliser_figure(fig, theme_actif, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        st.info(
            "La classe **Fraude** est fortement minoritaire (~3,7% des transactions dans le jeu de "
            "référence), ce qui est pris en compte lors de l'entraînement du modèle (`class_weight='balanced'`)."
        )

    # ---------------------------------------------------------------- Distributions (univarié)
    with onglet_dist:
        st.caption("Choisissez n'importe quelle variable pour en voir la distribution, colorée par statut.")
        explorateur_univarie(df, theme_actif, colonne_couleur="Target", key_prefix="eda_dist")

    # ---------------------------------------------------------------- Relations avec la cible
    with onglet_cible:
        st.subheader("Type de transaction et statut d'opération")
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(
                df, x="Type de transaction", color="Target", barnorm="percent",
                color_discrete_map=couleurs_target(theme_actif),
                title="Répartition (%) du statut selon le type de transaction",
            )
            styliser_figure(fig, theme_actif)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.histogram(
                df, x="Status operation", color="Target", barnorm="percent",
                color_discrete_map=couleurs_target(theme_actif),
                title="Répartition (%) du statut selon le statut d'opération",
            )
            styliser_figure(fig, theme_actif)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Distribution du montant selon le statut")
        fig = px.box(
            df, x="Target", y="Montant", color="Target",
            category_orders={"Target": ["Normal", "Suspect", "Fraude"]},
            color_discrete_map=couleurs_target(theme_actif), points=False,
            title="Distribution du montant par statut (échelle log)",
        )
        fig.update_yaxes(type="log")
        styliser_figure(fig, theme_actif)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Répartition du statut par type de transaction")
        df_type_target = statut_par_type(df)
        fig = px.bar(
            df_type_target, x="Type de transaction", y="Nombre", color="Target",
            color_discrete_map=couleurs_target(theme_actif), barmode="stack",
            title="Volume de transactions par type, réparti par statut",
        )
        styliser_figure(fig, theme_actif)
        st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------------------- Localisation
    with onglet_loc:
        nb_top = st.slider("Nombre de localisations à afficher", 5, 20, 10)

        st.subheader("Taux de fraude / suspicion par localisation")
        ct = taux_par_localisation(df, nb_top)
        if not ct.empty:
            colonnes_dispo = [c for c in ["Fraude", "Suspect"] if c in ct.columns]
            fig = px.bar(
                ct, x="Localisation", y=colonnes_dispo, barmode="group",
                color_discrete_map=couleurs_target(theme_actif),
                labels={"value": "Pourcentage (%)", "variable": "Statut"},
                title=f"Taux (%) de Fraude / Suspect — top {nb_top} localisations",
            )
            styliser_figure(fig, theme_actif)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Volume de transactions par localisation et type")
        df_vol = volume_localisation_type(df, nb_top)
        if not df_vol.empty:
            fig = px.treemap(
                df_vol, path=["Localisation", "Type de transaction"], values="Nombre",
                color="Nombre", color_continuous_scale="Blues",
                title=f"Volume de transactions — top {nb_top} localisations",
            )
            styliser_figure(fig, theme_actif)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Limité aux localisations les plus actives sélectionnées ci-dessus, pour rester lisible.")

        st.caption(
            "Aucune colonne de coordonnées (`lat`/`lon`) n'est présente dans ce jeu de données : "
            "la carte géographique n'est donc pas disponible ici. Les graphiques ci-dessus offrent une vue "
            "équivalente de la répartition par localisation."
        )

    # ---------------------------------------------------------------- Temporel
    with onglet_temps:
        par_heure, par_jour = agregat_temporel(df)
        colonnes_statuts = [c for c in ["Fraude", "Suspect", "Normal"] if c in par_heure.columns]

        st.subheader("Taux de statut par heure de la journée")
        if not par_heure.empty:
            fig = px.line(
                par_heure, x="heure", y=colonnes_statuts, markers=True,
                color_discrete_map=couleurs_target(theme_actif),
                labels={"heure": "Heure", "value": "Pourcentage (%)", "variable": "Statut"},
                title="Répartition (%) du statut par heure de la journée",
            )
            styliser_figure(fig, theme_actif)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Pas de date exploitable pour une analyse par heure.")

        st.subheader("Taux de statut par jour de la semaine")
        if not par_jour.empty:
            colonnes_statuts_jour = [c for c in ["Fraude", "Suspect", "Normal"] if c in par_jour.columns]
            fig = px.bar(
                par_jour, x="Jour", y=colonnes_statuts_jour, barmode="group",
                color_discrete_map=couleurs_target(theme_actif),
                labels={"value": "Pourcentage (%)", "variable": "Statut"},
                title="Répartition (%) du statut par jour de la semaine",
            )
            styliser_figure(fig, theme_actif)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Pas de date exploitable pour une analyse par jour de la semaine.")

    # ---------------------------------------------------------------- Corrélations
    with onglet_corr:
        st.subheader("Corrélation entre variables numériques et indicateurs de fraude")
        corr = matrice_correlation(df)
        fig = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            title="Matrice de corrélation",
        )
        styliser_figure(fig, theme_actif)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "`est_fraude` et `est_suspect_ou_fraude` sont des indicateurs binaires (0/1) — leur "
            "corrélation avec les autres variables est une lecture linéaire indicative, pas un test "
            "statistique formel de dépendance."
        )
