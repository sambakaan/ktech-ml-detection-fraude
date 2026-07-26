"""Configuration Plotly centralisée et adaptative au thème actif (clair/sombre)."""
from common.theme import PALETTES, couleurs_target

PLOTLY_TEMPLATE = {"light": "plotly_white", "dark": "plotly_dark"}


def styliser_figure(fig, theme_actif: str, **layout_kwargs):
    """Applique le template Plotly et les couleurs KTech adaptés au thème actif, en un
    point unique (remplace la répétition de `template=`/`paper_bgcolor=` dans chaque page)."""
    p = PALETTES.get(theme_actif, PALETTES["light"])
    fig.update_layout(
        template=PLOTLY_TEMPLATE.get(theme_actif, "plotly_white"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=p["ink"],
        **layout_kwargs,
    )
    return fig


def couleur_primaire(theme_actif: str) -> str:
    return PALETTES.get(theme_actif, PALETTES["light"])["primary"]


def couleur_accent(theme_actif: str) -> str:
    return PALETTES.get(theme_actif, PALETTES["light"])["accent"]


def hex_vers_rgba(couleur_hex: str, alpha: float = 1.0) -> str:
    """Convertit une couleur '#RRGGBB' en chaîne 'rgba(r,g,b,a)' — Plotly n'accepte pas la
    notation hexadécimale à 8 chiffres (avec canal alpha) sur certaines propriétés (ex :
    les steps d'un indicator.gauge)."""
    h = couleur_hex.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


__all__ = [
    "PLOTLY_TEMPLATE", "styliser_figure", "couleur_primaire", "couleur_accent",
    "couleurs_target", "hex_vers_rgba",
]
