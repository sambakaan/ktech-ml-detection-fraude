"""
Authentification utilisateur (via `streamlit-authenticator`), pour restreindre l'accès à
l'application à des comptes connus.

Configuration attendue dans `.streamlit/secrets.toml` (voir `secrets.toml.example`) :

    [cookie]
    name = "..."
    key = "..."
    expiry_days = 1

    [credentials.usernames.<identifiant>]
    email = "..."
    first_name = "..."
    last_name = "..."
    password = "<hash bcrypt>"
"""
import streamlit as st
import streamlit_authenticator as stauth

from common.theme import masquer_sidebar


def _secrets_vers_dict(section) -> dict:
    """Convertit récursivement un objet `st.secrets` (proxy en lecture seule) en dict Python
    natif — nécessaire car `streamlit_authenticator` fait des `isinstance(..., dict)` en
    interne, incompatibles avec le type renvoyé par `st.secrets`."""
    return {
        cle: (_secrets_vers_dict(valeur) if hasattr(valeur, "keys") else valeur)
        for cle, valeur in section.items()
    }


def construire_authenticator() -> stauth.Authenticate:
    credentials = _secrets_vers_dict(st.secrets["credentials"])
    cookie_cfg = st.secrets["cookie"]
    return stauth.Authenticate(
        credentials,
        cookie_cfg["name"],
        cookie_cfg["key"],
        cookie_cfg.get("expiry_days", 1),
    )


def exiger_authentification():
    """Gate à appeler tout en haut de `app/main.py`, avant tout chargement de données/modèle.
    Arrête le rendu (`st.stop()`) tant que l'utilisateur n'est pas authentifié.

    Retourne (nom_complet, identifiant) une fois l'authentification confirmée."""
    if "credentials" not in st.secrets:
        masquer_sidebar()
        st.error(
            "Authentification non configurée : créez `.streamlit/secrets.toml` à partir de "
            "`.streamlit/secrets.toml.example`."
        )
        st.stop()

    if "authenticator" not in st.session_state:
        st.session_state["authenticator"] = construire_authenticator()
    authenticator = st.session_state["authenticator"]
    authenticator.login(location="main")

    statut = st.session_state.get("authentication_status")
    if statut is False:
        masquer_sidebar()
        st.error("Identifiant ou mot de passe incorrect.")
        st.stop()
    if statut is None:
        masquer_sidebar()
        st.info("Veuillez vous connecter pour accéder à l'application.")
        st.stop()

    return st.session_state.get("name"), st.session_state.get("username")


def render_bouton_deconnexion():
    st.session_state["authenticator"].logout("Déconnexion", "sidebar")
