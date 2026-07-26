"""Pages 10-12 : Prédiction temps réel + jauge de risque + enregistrement dans l'historique."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common.charts import couleur_primaire, couleurs_target, hex_vers_rgba, styliser_figure
from common.model_utils import garde_modele
from common.theme import PALETTES, colorer_statut_fn, render_hero, render_verdict
from common.history_store import ajouter_ligne_historique


def render_gauge(proba_fraude: float, seuil: float, theme_actif: str = "light"):
    """Jauge de niveau de risque (vert / orange / rouge) pour la probabilité de fraude."""
    statuts = couleurs_target(theme_actif)
    p = PALETTES.get(theme_actif, PALETTES["light"])
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=proba_fraude * 100,
        number={"suffix": " %", "font": {"size": 34}},
        title={"text": "Niveau de risque de fraude", "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": couleur_primaire(theme_actif)},
            "bgcolor": "rgba(0,0,0,0)",
            "steps": [
                {"range": [0, 33], "color": hex_vers_rgba(statuts["Normal"], 0.35)},
                {"range": [33, 66], "color": hex_vers_rgba(statuts["Suspect"], 0.35)},
                {"range": [66, 100], "color": hex_vers_rgba(statuts["Fraude"], 0.35)},
            ],
            "threshold": {"line": {"color": p["ink"], "width": 3}, "thickness": 0.85, "value": seuil * 100},
        },
    ))
    fig.update_layout(
        height=260, margin=dict(l=20, r=20, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font={"color": p["ink"]},
    )
    st.plotly_chart(fig, use_container_width=True)


def render(df_raw, artefacts, model, scaler, le_target, feature_columns, top_localisations,
           best_threshold_f1, preparer_transaction_fn, history_path, theme_actif: str = "light"):
    render_hero(
        "online_prediction", "Analyse instantanée", "Détection en temps réel",
        "Renseignez les caractéristiques d'une transaction pour obtenir sa classification.",
    )

    if garde_modele(artefacts, "train_now_prediction"):
        st.stop()

    if st.session_state.get("df_actif") is not None:
        st.caption(
            "Le modèle utilisé ici reste entraîné sur le jeu de données par défaut : changer "
            "la source sur la page *Ingestion des données* ne modifie pas ses prédictions."
        )

    types_transaction = sorted(df_raw["Type de transaction"].unique())
    statuts_operation = sorted(df_raw["Status operation"].unique())
    localisations = sorted(df_raw["Localisation"].unique())

    with st.form("formulaire_transaction"):
        col1, col2 = st.columns(2)
        with col1:
            type_transaction = st.selectbox("Type de transaction", types_transaction)
            statut_operation = st.selectbox("Statut de l'opération", statuts_operation)
            localisation = st.selectbox("Localisation", localisations)
        with col2:
            montant = st.number_input("Montant (FCFA)", min_value=0.0, value=50000.0, step=1000.0)
            date_transaction = st.date_input("Date de la transaction")
            heure_transaction = st.time_input("Heure de la transaction")
            nb_transactions_client = st.number_input(
                "Nombre de transactions historiques du client", min_value=1, value=10, step=1
            )

        seuil_pred = st.slider(
            "Seuil de décision utilisé pour la classe Fraude",
            0.0, 1.0, float(round(best_threshold_f1, 2)), 0.01,
        )

        submit = st.form_submit_button(":material/search: Analyser la transaction", use_container_width=True)

    if not submit:
        return

    date_complete = pd.Timestamp.combine(date_transaction, heure_transaction)

    transaction = {
        "Type de transaction": type_transaction,
        "Status operation": statut_operation,
        "Localisation": localisation,
        "Date": date_complete,
        "Montant": montant,
        "nb_transactions_client": nb_transactions_client,
    }

    df_ligne = preparer_transaction_fn(transaction, top_localisations, feature_columns)
    df_ligne_scaled = scaler.transform(df_ligne)

    proba = model.predict_proba(df_ligne_scaled)[0]
    classes = le_target.inverse_transform(model.classes_)
    probas_dict = dict(zip(classes, proba))

    label_fraude = le_target.transform(["Fraude"])[0]
    idx_fraude = list(model.classes_).index(label_fraude)
    proba_fraude = proba[idx_fraude]

    est_fraude = proba_fraude >= seuil_pred
    if est_fraude:
        statut_hors_fraude = None
        statut_affiche = "Fraude"
    else:
        statut_hors_fraude = max(
            ((k, v) for k, v in probas_dict.items() if k != "Fraude"), key=lambda kv: kv[1]
        )[0]
        statut_affiche = statut_hors_fraude

    render_verdict(statut_affiche, proba_fraude, seuil_pred, theme_actif)

    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col1:
        st.subheader("Probabilités par classe")
        probas_df = pd.DataFrame({
            "Statut": list(probas_dict.keys()),
            "Probabilité": [v * 100 for v in probas_dict.values()],
        }).sort_values("Probabilité", ascending=False)
        st.dataframe(
            probas_df.round(1).style.map(colorer_statut_fn(theme_actif), subset=["Statut"]),
            hide_index=True, use_container_width=True,
        )

    with col2:
        fig = px.bar(
            probas_df, x="Statut", y="Probabilité", color="Statut",
            color_discrete_map=couleurs_target(theme_actif),
            labels={"Probabilité": "Probabilité (%)"},
            title="Répartition des probabilités prédites",
        )
        fig.add_hline(y=seuil_pred * 100, line_dash="dash",
                      line_color=PALETTES[theme_actif]["ink"],
                      annotation_text="Seuil de décision")
        styliser_figure(fig, theme_actif, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        render_gauge(proba_fraude, seuil_pred, theme_actif)

    with st.expander("Détails de la transaction analysée"):
        st.json({k: str(v) for k, v in transaction.items()})

    # ---- Enregistrement dans l'historique (session + fichier persistant) ----
    statut_predit = "Fraude" if est_fraude else statut_hors_fraude
    ligne_historique = {
        "Horodatage": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Type de transaction": type_transaction,
        "Status operation": statut_operation,
        "Localisation": localisation,
        "Date transaction": date_complete.strftime("%Y-%m-%d %H:%M:%S"),
        "Montant": montant,
        "Nb transactions client": nb_transactions_client,
        "Seuil utilisé": seuil_pred,
        "Statut prédit": statut_predit,
    }
    for k, v in probas_dict.items():
        ligne_historique[f"Probabilité {k} (%)"] = round(v * 100, 2)

    ajouter_ligne_historique(ligne_historique, history_path)
    st.caption(":material/check_circle: Cette analyse a été ajoutée à l'historique des prédictions "
               "(voir la page *Journal des prédictions*).")
