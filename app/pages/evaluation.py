"""Pages 8-9 : Évaluation détaillée (matrice de confusion, ROC, importance) + Seuil de décision."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from common.charts import couleur_accent, couleurs_target, styliser_figure
from common.model_utils import garde_modele
from common.theme import render_hero


def render(artefacts, model, feature_columns, top_localisations, scaler, le_target,
           best_threshold_f1, get_test_split_fn, theme_actif: str = "light"):
    render_hero(
        "insights", "Évaluation approfondie", "Performance du modèle",
        "Matrice de confusion, courbe ROC, importance des variables, et calibrage du seuil de décision.",
    )

    if garde_modele(artefacts, "train_now_evaluation"):
        st.stop()

    m = artefacts["metriques_test"]

    st.subheader("Rapport de classification détaillé (modèle retenu)")
    rapport_df = pd.DataFrame(m["rapport_classification"]).T
    rapport_df = rapport_df.loc[m["labels_ordre"] + ["accuracy", "macro avg", "weighted avg"]]
    st.dataframe(rapport_df.round(3), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Matrice de confusion")
        cm = np.array(m["matrice_confusion"])
        fig = px.imshow(
            cm, text_auto=True, color_continuous_scale="Blues",
            x=m["labels_ordre"], y=m["labels_ordre"],
            labels={"x": "Prédiction", "y": "Réalité", "color": "Nombre"},
        )
        styliser_figure(fig, theme_actif)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Importance des variables")
        if hasattr(model, "feature_importances_"):
            importances = pd.Series(model.feature_importances_, index=feature_columns)
            importances = importances.sort_values(ascending=False).head(12)
            fig = px.bar(
                importances[::-1], orientation="h",
                labels={"value": "Importance", "index": ""},
                color_discrete_sequence=[couleur_accent(theme_actif)],
            )
            styliser_figure(fig, theme_actif, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Ce modèle ne fournit pas d'importance de variables native.")

    st.markdown("---")
    st.subheader(":material/tune: Seuil de décision — classe Fraude")
    st.markdown(
        """
        Par défaut, une transaction est classée « Fraude » lorsque c'est la classe de probabilité la
        plus élevée (seuil implicite ≈ 0,5). En abaissant ce seuil, on détecte davantage de fraudes
        réelles, au prix de plus de fausses alertes. Ajustez le curseur ci-dessous pour explorer ce
        compromis.
        """
    )

    X_test, y_test = get_test_split_fn(tuple(top_localisations), tuple(feature_columns))
    X_test_scaled = scaler.transform(X_test)

    label_fraude = le_target.transform(["Fraude"])[0]
    idx_fraude = list(model.classes_).index(label_fraude)
    proba_fraude = model.predict_proba(X_test_scaled)[:, idx_fraude]
    y_test_bin = (y_test == label_fraude).astype(int)

    seuil = st.slider(
        "Seuil de décision (probabilité de Fraude)",
        min_value=0.0, max_value=1.0,
        value=float(round(best_threshold_f1, 2)), step=0.01,
    )
    st.caption(f":material/lightbulb: Seuil optimal (F1) calculé à l'entraînement : **{best_threshold_f1:.3f}** — "
               f"seuil par défaut d'un modèle multiclasse : **0.50**")

    pred_seuil = (proba_fraude >= seuil).astype(int)
    precision = precision_score(y_test_bin, pred_seuil, zero_division=0)
    rappel = recall_score(y_test_bin, pred_seuil, zero_division=0)
    f1 = f1_score(y_test_bin, pred_seuil, zero_division=0)
    faux_negatifs = int(((pred_seuil == 0) & (y_test_bin == 1)).sum())
    faux_positifs = int(((pred_seuil == 1) & (y_test_bin == 0)).sum())

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Précision", f"{precision * 100:.1f} %")
    col2.metric("Rappel", f"{rappel * 100:.1f} %")
    col3.metric("F1-score", f"{f1 * 100:.1f} %")
    col4.metric("Fraudes manquées", faux_negatifs)
    col5.metric("Fausses alertes", faux_positifs)

    couleur_fraude = couleurs_target(theme_actif)["Fraude"]
    couleur_normal = couleurs_target(theme_actif)["Normal"]

    col1, col2, col3 = st.columns(3)
    with col1:
        precisions, rappels, _ = precision_recall_curve(y_test_bin, proba_fraude)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rappels, y=precisions, mode="lines", name="Courbe P-R",
                                  line=dict(color=couleur_fraude)))
        fig.update_layout(
            title="Courbe précision-rappel", xaxis_title="Rappel", yaxis_title="Précision",
        )
        styliser_figure(fig, theme_actif)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fpr, tpr, _ = roc_curve(y_test_bin, proba_fraude)
        auc_val = roc_auc_score(y_test_bin, proba_fraude)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={auc_val:.3f})",
                                  line=dict(color=couleur_normal)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Aléatoire",
                                  line=dict(color="gray", dash="dash")))
        fig.update_layout(
            title="Courbe ROC", xaxis_title="Taux de faux positifs", yaxis_title="Taux de vrais positifs",
        )
        styliser_figure(fig, theme_actif)
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        cm_seuil = confusion_matrix(y_test_bin, pred_seuil)
        fig = px.imshow(
            cm_seuil, text_auto=True, color_continuous_scale="Blues",
            x=["Non-Fraude", "Fraude"], y=["Non-Fraude", "Fraude"],
            labels={"x": "Prédiction", "y": "Réalité", "color": "Nombre"},
            title=f"Matrice de confusion (seuil {seuil:.2f})",
        )
        styliser_figure(fig, theme_actif)
        st.plotly_chart(fig, use_container_width=True)
