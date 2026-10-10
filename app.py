import math
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="PronoScore IA Pro", page_icon="⚽", layout="wide")

API_BASE = "https://api.football-data.org/v4"
DEFAULT_COMPETITIONS = {
    "Premier League (Angleterre)": "PL",
    "Ligue 1 (France)": "FL1",
    "La Liga (Espagne)": "PD",
    "Serie A (Italie)": "SA",
    "Bundesliga (Allemagne)": "BL1",
    "Ligue des champions": "CL",
}

st.markdown("""
<style>
.stApp {background: linear-gradient(180deg,#0d1828 0%,#09111d 100%); color:#eef5ff}
.block-container {max-width:1200px;padding-top:2rem}
div[data-testid="stMetric"] {background:#132238;border:1px solid #29405d;padding:16px;border-radius:14px}
section[data-testid="stSidebar"] {background:#101d30}
h1,h2,h3 {letter-spacing:-.02em}
div.stButton>button {border-radius:10px;font-weight:700}
</style>
""", unsafe_allow_html=True)

st.title("⚽ PronoScore IA Pro")
st.caption("Analyse statistique du football · Scores exacts probables · Données de matchs en ligne")

with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Clé API football-data.org", type="password", help="Crée un compte sur football-data.org et colle ta clé API ici. Ne la partage pas.")
    comp_label = st.selectbox("Championnat", list(DEFAULT_COMPETITIONS.keys()))
    comp_code = DEFAULT_COMPETITIONS[comp_label]
    days_ahead = st.slider("Jours de matchs à venir", 1, 14, 7)
    history_n = st.slider("Matchs récents par équipe", 5, 20, 10)
    home_advantage = st.slider("Ajustement domicile", 0.90, 1.25, 1.08, 0.01)
    run = st.button("Actualiser les matchs et prédictions", type="primary", use_container_width=True)
    st.divider()
    st.caption("Les données disponibles dépendent du forfait et des limites de l’API.")

def api_get(path: str, key: str, params: dict | None = None) -> Any:
    response = requests.get(
        f"{API_BASE}{path}",
        headers={"X-Auth-Token": key},
        params=params,
        timeout=20,
    )
    if response.status_code == 401:
        raise RuntimeError("Clé API invalide ou non reconnue (HTTP 401).")
    if response.status_code == 403:
        raise RuntimeError("Cette compétition ou cette période n’est pas incluse dans ton forfait API (HTTP 403).")
    if response.status_code == 429:
        raise RuntimeError("Limite de requêtes atteinte. Attends un peu avant de réessayer (HTTP 429).")
    response.raise_for_status()
    return response.json()

@st.cache_data(ttl=900, show_spinner=False)
def load_fixtures(comp: str, key: str, days: int) -> list[dict]:
    today = datetime.now(timezone.utc).date()
    end = today + timedelta(days=days)
    data = api_get(f"/competitions/{comp}/matches", key, {
        "dateFrom": today.isoformat(),
        "dateTo": end.isoformat(),
        "status": "SCHEDULED",
    })
    return data.get("matches", [])

@st.cache_data(ttl=3600, show_spinner=False)
def load_recent_matches(comp: str, key: str, team_id: int, limit: int) -> list[dict]:
    data = api_get(f"/teams/{team_id}/matches", key, {
        "competitions": comp,
        "status": "FINISHED",
        "limit": limit,
    })
    return data.get("matches", [])

def poisson_prob(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1))

def team_form(matches: list[dict], team_id: int) -> dict | None:
    scored, conceded, home_scored, home_conceded, away_scored, away_conceded = [], [], [], [], [], []
    for m in matches:
        home = m.get("homeTeam", {})
        away = m.get("awayTeam", {})
        score = m.get("score", {}).get("fullTime", {})
        hg, ag = score.get("home"), score.get("away")
        if hg is None or ag is None:
            continue
        if home.get("id") == team_id:
            scored.append(hg); conceded.append(ag)
            home_scored.append(hg); home_conceded.append(ag)
        elif away.get("id") == team_id:
            scored.append(ag); conceded.append(hg)
            away_scored.append(ag); away_conceded.append(hg)
    if not scored:
        return None
    return {
        "n": len(scored),
        "gf": sum(scored) / len(scored),
        "ga": sum(conceded) / len(conceded),
        "home_gf": sum(home_scored) / len(home_scored) if home_scored else None,
        "home_ga": sum(home_conceded) / len(home_conceded) if home_conceded else None,
        "away_gf": sum(away_scored) / len(away_scored) if away_scored else None,
        "away_ga": sum(away_conceded) / len(away_conceded) if away_conceded else None,
    }

def predict(home_form: dict, away_form: dict, advantage: float) -> dict:
    # Simple blended scoring rates; not a trained ML model.
    home_attack = (home_form["home_gf"] if home_form["home_gf"] is not None else home_form["gf"])
    home_def = (home_form["home_ga"] if home_form["home_ga"] is not None else home_form["ga"])
    away_attack = (away_form["away_gf"] if away_form["away_gf"] is not None else away_form["gf"])
    away_def = (away_form["away_ga"] if away_form["away_ga"] is not None else away_form["ga"])
    lam_home = max(0.15, ((home_attack + away_def) / 2) * advantage)
    lam_away = max(0.15, (away_attack + home_def) / 2)
    cells = []
    home_win = draw = away_win = over25 = total = 0.0
    max_goals = 9
    for h in range(max_goals + 1):
        ph = poisson_prob(h, lam_home)
        for a in range(max_goals + 1):
            p = ph * poisson_prob(a, lam_away)
            cells.append((h, a, p))
            total += p
            if h > a: home_win += p
            elif h == a: draw += p
            else: away_win += p
            if h + a >= 3: over25 += p
    cells = [(h, a, p / total) for h, a, p in cells]
    cells.sort(key=lambda x: x[2], reverse=True)
    return {
        "home_lambda": lam_home, "away_lambda": lam_away, "scores": cells[:6],
        "home_win": home_win / total, "draw": draw / total,
        "away_win": away_win / total, "over25": over25 / total,
    }

def pct(x: float) -> str:
    return f"{100*x:.1f}%"

if not api_key:
    st.info("Pour récupérer automatiquement les matchs, ajoute une clé API gratuite ou compatible dans le panneau de gauche.")
    st.markdown("""
### Démarrage rapide
1. Crée un compte sur [football-data.org](https://www.football-data.org/client/register).
2. Récupère ta clé API dans ton compte.
3. Colle-la dans **Clé API football-data.org** à gauche.
4. Choisis un championnat puis clique sur **Actualiser les matchs et prédictions**.

La clé est saisie localement dans l’interface et n’est pas écrite dans le code. Les données historiques disponibles et les compétitions autorisées dépendent de l’offre du fournisseur.
""")
    st.stop()

if run or "fixtures" not in st.session_state:
    try:
        with st.spinner("Récupération des matchs programmés…"):
            st.session_state.fixtures = load_fixtures(comp_code, api_key, days_ahead)
            st.session_state.comp_code = comp_code
            st.session_state.api_key_fingerprint = len(api_key)
    except Exception as exc:
        st.error(f"Impossible de récupérer les matchs : {exc}")
        st.stop()

fixtures = st.session_state.get("fixtures", [])
if not fixtures:
    st.warning("Aucun match programmé trouvé pour cette compétition dans la période sélectionnée. Essaie une autre compétition ou augmente la période.")
    st.stop()

rows = []
progress = st.progress(0, text="Analyse des équipes…")
errors = []
for idx, match in enumerate(fixtures):
    home = match.get("homeTeam", {})
    away = match.get("awayTeam", {})
    try:
        hm = load_recent_matches(comp_code, api_key, home["id"], history_n)
        am = load_recent_matches(comp_code, api_key, away["id"], history_n)
        hf, af = team_form(hm, home["id"]), team_form(am, away["id"])
        if not hf or not af:
            errors.append(f"{home.get('name','Domicile')} / {away.get('name','Extérieur')}: historique insuffisant.")
            continue
        pred = predict(hf, af, home_advantage)
        rows.append({
            "Date": match.get("utcDate", "")[:16].replace("T", " ") + " UTC",
            "Domicile": home.get("name", "Domicile"),
            "Extérieur": away.get("name", "Extérieur"),
            "Score le plus probable": f"{pred['scores'][0][0]}–{pred['scores'][0][1]}",
            "Probabilité du score": pct(pred["scores"][0][2]),
            "Victoire dom.": pct(pred["home_win"]),
            "Nul": pct(pred["draw"]),
            "Victoire ext.": pct(pred["away_win"]),
            "_pred": pred,
            "_home_form": hf,
            "_away_form": af,
            "_id": match.get("id"),
        })
    except Exception as exc:
        errors.append(f"{home.get('name','Domicile')} / {away.get('name','Extérieur')}: {exc}")
    progress.progress((idx + 1) / len(fixtures), text=f"Analyse des équipes… {idx+1}/{len(fixtures)}")
progress.empty()

st.subheader(f"Matchs à venir · {comp_label}")
st.caption(f"{len(rows)} match(s) analysé(s). Données récupérées automatiquement ; cache de 15 minutes pour les calendriers et 1 heure pour l’historique.")
if not rows:
    st.error("Aucun match n’a pu être prédit avec les données disponibles.")
    if errors:
        with st.expander("Détails des problèmes"):
            st.write("\n".join(errors))
    st.stop()

df = pd.DataFrame([{k:v for k,v in r.items() if not k.startswith("_")} for r in rows])
st.dataframe(df, use_container_width=True, hide_index=True)

options = [f"{r['Domicile']} — {r['Extérieur']}" for r in rows]
chosen = st.selectbox("Examiner un match en détail", options)
selected = rows[options.index(chosen)]
pred = selected["_pred"]
st.divider()
st.subheader(f"{selected['Domicile']} vs {selected['Extérieur']}")
a,b,c = st.columns(3)
a.metric("Score exact le plus probable", selected["Score le plus probable"], selected["Probabilité du score"])
b.metric("Buts attendus domicile", f"{pred['home_lambda']:.2f}")
c.metric("Buts attendus extérieur", f"{pred['away_lambda']:.2f}")

left, right = st.columns([1,1])
with left:
    st.markdown("#### Top 6 des scores exacts")
    score_df = pd.DataFrame([
        {"Score": f"{h}–{a}", "Probabilité estimée": pct(p)}
        for h,a,p in pred["scores"]
    ])
    st.dataframe(score_df, use_container_width=True, hide_index=True)
with right:
    st.markdown("#### Probabilités du résultat")
    outcome_df = pd.DataFrame({
        "Issue": ["Victoire domicile", "Match nul", "Victoire extérieur", "Plus de 2,5 buts"],
        "Probabilité estimée": [pred["home_win"], pred["draw"], pred["away_win"], pred["over25"]],
    })
    st.bar_chart(outcome_df.set_index("Issue"))
    st.caption("Le total des probabilités de score exact est approximé en limitant les buts à 0–9.")

st.markdown("#### Forme récente utilisée")
f1, f2 = st.columns(2)
with f1:
    st.write(f"**{selected['Domicile']}** · {selected['_home_form']['n']} matchs utilisables")
    st.write(f"Buts marqués/match : {selected['_home_form']['gf']:.2f}")
    st.write(f"Buts encaissés/match : {selected['_home_form']['ga']:.2f}")
with f2:
    st.write(f"**{selected['Extérieur']}** · {selected['_away_form']['n']} matchs utilisables")
    st.write(f"Buts marqués/match : {selected['_away_form']['gf']:.2f}")
    st.write(f"Buts encaissés/match : {selected['_away_form']['ga']:.2f}")

st.warning("""
**Limites importantes :** ce prototype utilise un modèle statistique simple de Poisson basé sur les buts des matchs récents. Ce n’est pas encore un modèle d’apprentissage automatique entraîné et validé. Il ne prend pas automatiquement en compte les blessures, les compositions, les xG, les suspensions, les changements d’entraîneur, la météo ou les cotes. Les données et quotas dépendent du fournisseur. Un score affiché comme « le plus probable » reste loin d’être certain ; compare les probabilités et teste le modèle sur des saisons passées avant de lui faire confiance.
""")
if errors:
    with st.expander(f"Matchs ignorés ou problèmes ({len(errors)})"):
        st.write("\n".join(errors))
