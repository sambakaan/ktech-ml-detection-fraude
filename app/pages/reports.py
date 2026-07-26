"""Page 13 (+9) : Historique des prédictions & génération de rapports (CSV / PDF / Excel)."""
import pandas as pd
import streamlit as st

from common.theme import colorer_statut_fn, render_hero
from common.history_store import generer_excel_historique, generer_pdf_historique, vider_historique


def render(history_path: str, theme_actif: str = "light"):
    render_hero(
        "history", "Suivi & audit", "Journal des prédictions",
        "Transactions analysées via la page *Détection en temps réel*.",
    )
    st.caption(
        "Cet historique est conservé sur le disque de l'application tant qu'elle n'est pas "
        "redémarrée : ce n'est pas une base de données persistante à long terme (particulièrement "
        "sur Streamlit Community Cloud, où le disque est réinitialisé à chaque redéploiement)."
    )

    historique = st.session_state.get("historique_predictions", [])

    if not historique:
        st.info(
            "Aucune prédiction enregistrée pour l'instant. Rendez-vous sur la page "
            "*Détection en temps réel* pour analyser une première transaction."
        )
        return

    df_hist = pd.DataFrame(historique)

    col1, col2, col3 = st.columns(3)
    col1.metric("Prédictions enregistrées", len(df_hist))
    col2.metric("Dont Fraude détectée", int((df_hist["Statut prédit"] == "Fraude").sum()))
    montant_total = df_hist["Montant"].sum()
    col3.metric("Montant total analysé", f"{montant_total:,.0f} FCFA".replace(",", " "))

    statuts_dispo = sorted(df_hist["Statut prédit"].unique())
    filtre_statut = st.multiselect("Filtrer par statut prédit", statuts_dispo, default=statuts_dispo)
    df_filtre = df_hist[df_hist["Statut prédit"].isin(filtre_statut)].sort_values(
        "Horodatage", ascending=False
    )

    st.dataframe(
        df_filtre.style.map(colorer_statut_fn(theme_actif), subset=["Statut prédit"]),
        use_container_width=True, hide_index=True,
    )

    resume = {
        "Prédictions enregistrées": len(df_filtre),
        "Dont Fraude détectée": int((df_filtre["Statut prédit"] == "Fraude").sum()),
        "Montant total analysé (FCFA)": round(float(df_filtre["Montant"].sum()), 2),
        "Généré le": pd.Timestamp.now().strftime("%d/%m/%Y à %H:%M"),
    }

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        csv_bytes = df_filtre.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            ":material/description: CSV", data=csv_bytes,
            file_name=f"historique_predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv", use_container_width=True,
        )
    with col2:
        with st.spinner("Génération du PDF..."):
            pdf_bytes = generer_pdf_historique(df_filtre)
        st.download_button(
            ":material/picture_as_pdf: PDF", data=pdf_bytes,
            file_name=f"historique_predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf", use_container_width=True,
        )
    with col3:
        with st.spinner("Génération du fichier Excel..."):
            excel_bytes = generer_excel_historique(df_filtre, resume)
        st.download_button(
            ":material/table_view: Excel", data=excel_bytes,
            file_name=f"historique_predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    with col4:
        if st.button(":material/delete: Vider l'historique", use_container_width=True):
            vider_historique(history_path)
            st.rerun()
