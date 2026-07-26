"""Pages 6-7 : Entraînement supervisé (comparaison de modèles) + Isolation Forest + suivi."""
import pandas as pd
import plotly.express as px
import streamlit as st

from common.charts import couleur_accent, styliser_figure
from common.model_utils import garde_modele
from common.theme import render_hero
from common.training_history import charger_historique_entrainement


def render(artefacts: dict, model, theme_actif: str = "light", training_history_path: str | None = None):
    render_hero(
        "model_training", "Entraînement & comparaison", "Modélisation",
        "Comparaison des modèles supervisés, détection non supervisée complémentaire, et suivi "
        "des performances dans le temps.",
    )

    if garde_modele(artefacts, "train_now_training"):
        st.stop()

    m = artefacts["metriques_test"]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Modèle retenu", artefacts.get("model_nom", type(model).__name__))
    col2.metric("Accuracy (test)", f"{m['accuracy'] * 100:.1f} %")
    col3.metric("Rappel - Fraude", f"{m['rapport_classification']['Fraude']['recall'] * 100:.1f} %")
    col4.metric("Précision - Fraude", f"{m['rapport_classification']['Fraude']['precision'] * 100:.1f} %")

    st.caption(f"Meilleurs hyperparamètres (GridSearchCV) : `{artefacts['grid_search_best_params']}` — "
               f"évalué sur {m['n_test']} transactions de test.")

    st.markdown("---")

    onglet_supervise, onglet_iso, onglet_suivi = st.tabs([
        ":material/emoji_events: Comparaison des modèles supervisés",
        ":material/search: Isolation Forest (non supervisé)",
        ":material/timeline: Suivi dans le temps",
    ])

    with onglet_supervise:
        st.caption(
            "Tous les modèles sont entraînés avec une pondération 'balanced' des classes pour compenser "
            "la rareté de la classe Fraude. Le modèle avec le meilleur rappel sur la classe Fraude est "
            "automatiquement sélectionné puis optimisé par GridSearchCV."
        )
        if "comparaison_modeles" in artefacts:
            comp_df = pd.DataFrame(artefacts["comparaison_modeles"]).set_index("Modèle")

            modeles_disponibles = comp_df.index.tolist()
            modeles_choisis = st.multiselect(
                "Modèles à comparer", modeles_disponibles, default=modeles_disponibles,
            )
            comp_df_filtre = comp_df.loc[modeles_choisis] if modeles_choisis else comp_df
            comp_df_pct = (comp_df_filtre * 100).round(1)

            def surligner_meilleur(row):
                est_meilleur = row.name == artefacts.get("model_nom")
                couleur = couleur_accent(theme_actif)
                return [f"background-color: {couleur}33" if est_meilleur else "" for _ in row]

            st.dataframe(comp_df_pct.style.apply(surligner_meilleur, axis=1), use_container_width=True)

            fig = px.bar(
                comp_df_filtre.reset_index(), x="Modèle",
                y=["Accuracy", "Rappel (macro)", "F1-score (macro)", "Rappel classe Fraude"],
                barmode="group",
                title="Comparaison des modèles (jeu de test)",
            )
            fig.update_layout(yaxis_title="Score", legend_title="Métrique")
            styliser_figure(fig, theme_actif)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Isolation Forest n'est pas comparable sur les mêmes axes (non supervisé, n'utilise "
                "jamais la variable cible) — voir l'onglet suivant."
            )
        else:
            st.info("Comparaison indisponible — réentraînez le modèle avec la dernière version de "
                    "`train_model.py` pour l'obtenir.")

    with onglet_iso:
        st.caption(
            "Contrairement aux modèles précédents, Isolation Forest n'utilise **jamais** la variable "
            "cible : il repère des transactions statistiquement atypiques (anomalies), sans avoir appris "
            "ce qu'est réellement une fraude. Il est donc moins précis ici, mais reste utile en pratique "
            "pour détecter des schémas de fraude inédits, jamais rencontrés à l'entraînement."
        )
        if "metriques_isolation_forest" in artefacts:
            iso = artefacts["metriques_isolation_forest"]
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Taux d'anomalies attendu", f"{iso['contamination'] * 100:.1f} %")
            col2.metric("Précision (vs Fraude réelle)", f"{iso['precision'] * 100:.1f} %")
            col3.metric("Rappel (vs Fraude réelle)", f"{iso['rappel'] * 100:.1f} %")
            col4.metric("F1-score", f"{iso['f1'] * 100:.1f} %")

            st.caption(
                "Lecture : la précision indique la part d'anomalies détectées qui sont réellement des "
                "fraudes ; le rappel indique la part des fraudes réelles effectivement repérées comme "
                "anomalies. Les faux positifs correspondent aux transactions normales signalées à tort."
            )
        else:
            st.info("Résultats Isolation Forest indisponibles — réentraînez le modèle avec la dernière "
                    "version de `train_model.py` pour les obtenir.")

    with onglet_suivi:
        if not training_history_path:
            st.info("Historique des entraînements indisponible.")
        else:
            historique = charger_historique_entrainement(training_history_path)
            if historique.empty:
                st.info(
                    "Aucun run d'entraînement enregistré pour l'instant. Relancez l'entraînement "
                    "(bouton ci-dessus si le modèle est absent, ou `python ml/train.py`) pour "
                    "commencer à construire une tendance."
                )
            else:
                historique = historique.sort_values("horodatage")
                if len(historique) == 1:
                    st.info(
                        "Un seul run enregistré pour l'instant — relancez l'entraînement pour "
                        "construire une tendance."
                    )
                else:
                    dernier = historique.iloc[-1]
                    avant_dernier = historique.iloc[-2]
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Accuracy (dernier run)", f"{dernier['accuracy'] * 100:.1f} %",
                                f"{(dernier['accuracy'] - avant_dernier['accuracy']) * 100:+.1f} pt")
                    col2.metric("Rappel Fraude (dernier run)", f"{dernier['rappel_fraude'] * 100:.1f} %",
                                f"{(dernier['rappel_fraude'] - avant_dernier['rappel_fraude']) * 100:+.1f} pt")
                    col3.metric("F1 Fraude (dernier run)", f"{dernier['f1_fraude'] * 100:.1f} %",
                                f"{(dernier['f1_fraude'] - avant_dernier['f1_fraude']) * 100:+.1f} pt")

                    fig = px.line(
                        historique, x="horodatage", y=["accuracy", "rappel_fraude", "f1_fraude"],
                        markers=True, labels={"value": "Score", "horodatage": "Date", "variable": "Métrique"},
                        title="Évolution des métriques par run d'entraînement",
                    )
                    styliser_figure(fig, theme_actif)
                    st.plotly_chart(fig, use_container_width=True)

                st.dataframe(
                    historique[["horodatage", "model_nom", "accuracy", "rappel_fraude", "f1_fraude",
                                "best_threshold_f1", "duree_entrainement_sec"]].sort_values(
                        "horodatage", ascending=False
                    ),
                    use_container_width=True, hide_index=True,
                )
