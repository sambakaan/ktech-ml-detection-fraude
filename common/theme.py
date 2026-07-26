"""
Thème visuel KTech Solutions : palettes claire et sombre, typographie Poppins,
bascule Clair / Sombre / Système, et composants partagés (hero, badges, verdict,
emblème de marque dans la sidebar).
"""
import html
from functools import partial
from pathlib import Path

import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = BASE_DIR / "app" / "assets"

# ---------------------------------------------------------------------------
# Palettes de marque KTech Solutions
# ---------------------------------------------------------------------------
PALETTES = {
    "light": {
        "bg": "#FFFFFF",
        "bg_alt": "#F1F6FC",
        "panel": "#FFFFFF",
        "panel_alt": "#EAF1FB",
        "line": "#CDDFF4",
        "primary": "#3462AC",
        "primary_hover": "#2A5089",
        "accent": "#1FAE7A",
        "ink": "#153B58",
        "ink_muted": "#4C6E90",
        "ink_onprimary": "#FFFFFF",
    },
    "dark": {
        "bg": "#0E2A40",
        "bg_alt": "#123049",
        "panel": "#153B58",
        "panel_alt": "#1D4A6E",
        "line": "#2C5878",
        "primary": "#5B8FD6",
        "primary_hover": "#7DA8E0",
        "accent": "#2FD79A",
        "ink": "#EAF2FB",
        "ink_muted": "#9DBEE5",
        "ink_onprimary": "#0B1F30",
    },
}

# Palette sémantique des statuts (utilisée par les graphiques Plotly ET les badges HTML),
# assombrie en mode clair pour rester lisible (contraste AA) sur fond blanc.
STATUTS = {
    "light": {"Normal": "#17945F", "Suspect": "#C97A00", "Fraude": "#C22B4E"},
    "dark": {"Normal": "#34D399", "Suspect": "#F5A524", "Fraude": "#FB4570"},
}
CLASSE_CSS = {"Normal": "badge-normal", "Suspect": "badge-suspect", "Fraude": "badge-fraude"}
ICONE_VERDICT = {"Normal": "check_circle", "Suspect": "warning", "Fraude": "dangerous"}


def couleurs_target(theme_actif: str) -> dict:
    """Retourne le dict de couleurs Normal/Suspect/Fraude adapté au thème actif."""
    return STATUTS.get(theme_actif, STATUTS["light"])


# ---------------------------------------------------------------------------
# Résolution du thème actif (Clair / Sombre / Système)
# ---------------------------------------------------------------------------
def resoudre_theme_actif() -> str:
    """Détermine le thème actif ('light' ou 'dark') à partir du choix utilisateur.

    Le choix explicite (Clair/Sombre) est mémorisé dans `st.session_state` et dans l'URL
    (`?theme=`), ce qui le fait survivre à un rechargement de page. En mode "Système", la
    préférence du système d'exploitation est détectée côté navigateur (voir
    `common.theme_bridge`) puis transmise à Python via le paramètre d'URL `resolved`.
    """
    choix = st.session_state.get("theme_choice")
    if choix is None:
        choix = st.query_params.get("theme", "system")
        st.session_state["theme_choice"] = choix

    if choix in ("light", "dark"):
        return choix

    return st.query_params.get("resolved", "light")


def render_theme_switcher() -> str:
    """Affiche le sélecteur Clair / Sombre / Système dans la sidebar et retourne le thème
    actif résolu ('light' ou 'dark')."""
    options = ["system", "light", "dark"]
    labels = {"system": ":material/routine: Système", "light": ":material/light_mode: Clair",
              "dark": ":material/dark_mode: Sombre"}
    choix_actuel = st.session_state.get("theme_choice", "system")

    if hasattr(st.sidebar, "segmented_control"):
        choix = st.sidebar.segmented_control(
            "Thème", options, format_func=lambda o: labels[o],
            default=choix_actuel, key="theme_choice_widget",
        )
        choix = choix or choix_actuel
    else:
        choix = st.sidebar.radio(
            "Thème", options, format_func=lambda o: labels[o],
            index=options.index(choix_actuel), key="theme_choice_widget",
        )

    if choix != st.session_state.get("theme_choice"):
        st.session_state["theme_choice"] = choix
        st.query_params["theme"] = choix
        st.rerun()

    return resoudre_theme_actif()


# ---------------------------------------------------------------------------
# Injection du CSS
# ---------------------------------------------------------------------------
def injecter_theme(theme_actif: str = "light"):
    """Injecte la feuille de style de l'application pour le thème actif donné
    (à appeler à chaque run, après résolution du thème)."""
    p = PALETTES.get(theme_actif, PALETTES["light"])

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;700&display=swap');
        @import url('https://fonts.googleapis.com/icon?family=Material+Symbols+Outlined');

        .material-symbols-outlined {{
            font-family: 'Material Symbols Outlined';
            font-weight: normal; font-style: normal; font-size: 1.2rem;
            display: inline-block; line-height: 1; text-transform: none;
            letter-spacing: normal; word-wrap: normal; white-space: nowrap;
            direction: ltr; vertical-align: middle;
            -webkit-font-smoothing: antialiased;
        }}

        :root {{
            --kt-bg: {p['bg']};
            --kt-bg-alt: {p['bg_alt']};
            --kt-panel: {p['panel']};
            --kt-panel-alt: {p['panel_alt']};
            --kt-line: {p['line']};
            --kt-primary: {p['primary']};
            --kt-primary-hover: {p['primary_hover']};
            --kt-accent: {p['accent']};
            --kt-ink: {p['ink']};
            --kt-ink-muted: {p['ink_muted']};
            --kt-ink-onprimary: {p['ink_onprimary']};
        }}

        html, body, [class*="css"] {{ font-family: 'Poppins', sans-serif; font-weight: 400; }}
        h1, h2, h3 {{ font-family: 'Poppins', sans-serif !important; font-weight: 700 !important; }}
        .stCaption, small, [data-testid="stCaptionContainer"] {{ font-weight: 300; }}

        .stApp {{ background: var(--kt-bg); color: var(--kt-ink); }}

        /* ---------- Barre latérale (toujours navy, cf. .streamlit/config.toml) ---------- */
        /* Logo en haut, spacer flexible, liens de navigation poussés en bas */
        section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {{
            display: flex; flex-direction: column; min-height: 100vh;
        }}
        .kt-sidebar-spacer {{ flex: 1 1 auto; }}
        section[data-testid="stSidebar"] [data-testid="stPageLink"] {{
            border-radius: 10px; margin-bottom: 2px;
        }}
        section[data-testid="stSidebar"] [data-testid="stPageLink"] a[aria-current="page"] {{
            background: rgba(255,255,255,0.12);
            border-left: 3px solid var(--kt-accent);
            font-weight: 700;
        }}
        .sidebar-emblem-wrap {{ display:flex; align-items:center; gap:12px; padding: 4px 0 20px 0; }}
        .sidebar-emblem {{ width:40px; height:40px; flex-shrink:0; }}
        .sidebar-emblem img {{ width:100%; height:100%; }}
        .sidebar-brand-title {{
            font-family:'Poppins', sans-serif; font-weight:700;
            font-size:1.05rem; line-height:1.2; margin:0; color:#FFFFFF;
        }}
        .sidebar-brand-sub {{
            font-size:0.68rem; color:#9DBEE5; text-transform:uppercase;
            letter-spacing:0.09em; margin:0; font-weight:300;
        }}
        .sidebar-footer-credit {{
            font-size:0.7rem; color:#9DBEE5; text-align:center;
            margin:10px 0 0 0; font-weight:300; opacity:0.75;
        }}

        /* ---------- En-tête de page (hero) ---------- */
        .hero {{ padding: 0 0 16px 0; margin-bottom: 10px; border-bottom: 1px solid var(--kt-line); }}
        .hero-eyebrow {{
            font-size:0.72rem; color:var(--kt-accent); font-weight:700;
            letter-spacing:0.14em; text-transform:uppercase; margin:0 0 6px 0;
        }}
        .hero-title {{
            font-family:'Poppins', sans-serif; font-size: 2rem; font-weight:700; margin:0;
            color: var(--kt-ink); display:flex; align-items:center; gap:10px;
        }}
        .hero-title .material-symbols-outlined {{ font-size: 1.7rem; color: var(--kt-primary); }}
        .hero-subtitle {{ color: var(--kt-ink-muted); margin-top:8px; font-size:0.95rem; max-width: 820px; font-weight:300; }}

        /* ---------- Cartes de métriques ---------- */
        div[data-testid="stMetric"] {{
            background: var(--kt-panel); border:1px solid var(--kt-line); border-left: 3px solid var(--kt-accent);
            border-radius: 12px; padding: 14px 16px 10px 16px;
        }}
        div[data-testid="stMetricValue"] {{ color: var(--kt-ink); font-weight:700; }}
        div[data-testid="stMetricLabel"] {{ color: var(--kt-ink-muted); }}

        /* ---------- Badges de statut ---------- */
        .badge {{
            display:inline-flex; align-items:center; gap:7px; padding:5px 12px;
            border-radius:999px; font-size:0.82rem; font-weight:700;
        }}
        .badge-dot {{ width:8px; height:8px; border-radius:50%; flex-shrink:0; }}

        /* ---------- Carte de verdict (résultat de prédiction) ---------- */
        .verdict-card {{
            border-radius: 14px; padding: 20px 22px; margin: 14px 0 18px 0;
            background: var(--kt-panel); border: 1px solid var(--kt-line); border-left-width: 4px;
        }}
        .verdict-title {{
            font-family:'Poppins', sans-serif; font-size:1.15rem; font-weight:700; margin:0 0 4px 0;
            display:flex; align-items:center; gap:8px; color: var(--kt-ink);
        }}
        .verdict-detail {{ color: var(--kt-ink-muted); font-size:0.9rem; font-weight:300; }}

        /* ---------- Boutons ---------- */
        .stButton>button, .stDownloadButton>button, button[kind="formSubmit"] {{
            background: var(--kt-primary); color: var(--kt-ink-onprimary); border:none; border-radius:10px;
            font-weight:600; transition: transform 0.12s ease, box-shadow 0.12s ease;
        }}
        .stButton>button:hover, .stDownloadButton>button:hover, button[kind="formSubmit"]:hover {{
            background: var(--kt-primary-hover); transform: translateY(-1px);
        }}

        /* ---------- Tableaux et alertes ---------- */
        div[data-testid="stDataFrame"] {{ border:1px solid var(--kt-line); border-radius:10px; overflow:hidden; }}
        div[data-testid="stAlert"] {{ border-radius:10px; }}
        div[data-testid="stMetricValue"], .stDataFrame {{ font-variant-numeric: tabular-nums; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(icon: str, eyebrow: str, titre: str, sous_titre: str = ""):
    """Affiche l'en-tête stylisé d'une page (icône Material, eyebrow, titre, sous-titre).

    `icon` est le nom d'une icône Material Symbols (ex: "dashboard"), sans les `:material/...:`.
    """
    sous_titre_html = f'<p class="hero-subtitle">{sous_titre}</p>' if sous_titre else ""
    st.markdown(
        f"""
        <div class="hero">
            <p class="hero-eyebrow">{eyebrow}</p>
            <h1 class="hero-title"><span class="material-symbols-outlined">{icon}</span> {titre}</h1>
            {sous_titre_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_brand():
    """Affiche le logo KTech Solutions (version blanche) + titre de marque dans la sidebar."""
    logo_svg = (ASSETS_DIR / "logo_white.svg").read_text()
    st.sidebar.markdown(
        f"""
        <div class="sidebar-emblem-wrap">
            <div class="sidebar-emblem">{logo_svg}</div>
            <div>
                <p class="sidebar-brand-title">Détection de fraude</p>
                <p class="sidebar-brand-sub">KTech Solutions</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def masquer_sidebar():
    """Masque entièrement la sidebar (panneau + flèche d'ouverture), à appeler tant que
    l'utilisateur n'est pas authentifié — sinon son panneau vide (navy, cf.
    `.streamlit/config.toml`) reste visible sur l'écran de login."""
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] { display: none; }
        div[data-testid="stExpandSidebarButton"] { display: none; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def badge_html(statut: str, theme_actif: str = "light") -> str:
    """Retourne le HTML d'un badge coloré pour un statut (Normal / Suspect / Fraude).

    `statut` est échappé avant insertion dans le HTML — défense en profondeur si cette
    valeur venait un jour d'une donnée non contrôlée (ex : ré-entraînement sur un CSV tiers)."""
    couleur = couleurs_target(theme_actif).get(statut, couleurs_target(theme_actif)["Suspect"])
    classe = CLASSE_CSS.get(statut, "badge-suspect")
    return (
        f'<span class="badge {classe}" style="background:{couleur}22; color:{couleur};">'
        f'<span class="badge-dot" style="background:{couleur};"></span>{html.escape(str(statut))}</span>'
    )


def render_verdict(statut: str, proba_fraude: float, seuil: float, theme_actif: str = "light"):
    """Affiche la carte de verdict stylisée (résultat de la classification d'une transaction)."""
    couleur = couleurs_target(theme_actif).get(statut, couleurs_target(theme_actif)["Suspect"])
    icone = ICONE_VERDICT.get(statut, "warning")
    libelle = "FRAUDE POTENTIELLE" if statut == "Fraude" else html.escape(str(statut).upper())
    st.markdown(
        f"""
        <div class="verdict-card" style="border-left-color:{couleur};">
            <p class="verdict-title"><span class="material-symbols-outlined" style="color:{couleur};">{icone}</span> Transaction classée : {libelle} {badge_html(statut, theme_actif)}</p>
            <p class="verdict-detail">Probabilité de fraude = {proba_fraude * 100:.1f} %  |  Seuil utilisé = {seuil:.2f}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def colorer_statut(val, theme_actif: str = "light"):
    """Fonction de style pandas : colore le texte d'une cellule selon le statut
    (Normal / Suspect / Fraude), pour prolonger la palette dans les tableaux natifs.
    Utiliser via `functools.partial(colorer_statut, theme_actif=...)` avec `.style.map(...)`."""
    couleur = couleurs_target(theme_actif).get(val)
    if couleur:
        return f"color: {couleur}; font-weight: 600;"
    return ""


def colorer_statut_fn(theme_actif: str):
    """Raccourci pour obtenir une fonction de style pandas liée au thème actif."""
    return partial(colorer_statut, theme_actif=theme_actif)
