"""
Pont JS pour la détection du thème système (clair/sombre) en mode "Système".

Streamlit exécute tout le script Python à chaque interaction ; Python peut donc choisir la
palette à injecter à chaque run, à condition de connaître la préférence de l'OS. Ce petit
composant, invisible (hauteur nulle), lit `prefers-color-scheme` côté navigateur et, si la
résolution diffère de celle déjà connue de Python (transmise via le paramètre d'URL
`resolved`), met à jour l'URL et recharge la page une fois — le seul moyen fiable de faire
remonter l'information côté serveur sans dépendance ni cookie additionnel.
"""
import streamlit.components.v1 as components


def injecter_pont_js(mode_systeme: bool):
    """À appeler uniquement lorsque le choix utilisateur est "Système". `mode_systeme`
    n'est utilisé que pour éviter d'injecter le composant inutilement en mode Clair/Sombre
    explicite (appelant responsable de ce filtrage)."""
    if not mode_systeme:
        return

    components.html(
        """
        <script>
        (function() {
            const resolved = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
            const params = new URLSearchParams(window.parent.location.search);
            if (params.get('resolved') !== resolved) {
                params.set('resolved', resolved);
                window.parent.location.search = params.toString();
            }
        })();
        </script>
        """,
        height=0,
    )
