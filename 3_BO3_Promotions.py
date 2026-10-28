import numpy as np
import pandas as pd
import streamlit as st

import bo3_utils as U

st.set_page_config(page_title="BO3 · Planifier les promotions", page_icon="📈", layout="wide")


@st.cache_resource
def load_model():
    return U.load_model(U.MODEL)


@st.cache_data
def load_data():
    return U.load_prepared(U.DATA)


art, df = load_model(), load_data()
FEATS = art['features']

st.title("📈 BO3 · Planifier les promotions")
st.caption("Estimer le délai avant la prochaine promotion (DSO) · AFT log-logistique · Employee Attrition")

st.sidebar.header("À propos")
st.sidebar.markdown(f"**Modèle retenu** : AFT log-logistique  \n**C-index test** : {U.CINDEX['Modèle 2 : AFT log-logistique']:.3f}")
st.sidebar.info("Outil d'aide à la décision RH : un délai estimé n'est pas une promesse de promotion.")


def show_prediction(row):
    """Affiche le délai médian, les probabilités et la courbe pour un profil (DataFrame d'une ligne)."""
    med = U.median_wait(art, row)[0]
    st.metric("Délai médian avant promotion", f"{med:.1f} ans")
    probs = U.prob_promoted_within(art, row)
    cols = st.columns(len(probs))
    for c, (y, p) in zip(cols, probs.items()):
        c.metric(f"Promu d'ici {y} an{'s' if y > 1 else ''}", f"{p:.0%}")
    st.line_chart(U.survival_curve(art, row), x_label="Années", y_label="Probabilité de ne pas encore être promu")


t1, t2, t3, t4 = st.tabs(["👤 Employé existant", "🆕 Nouveau profil", "📄 Lot (CSV)", "ℹ️ À propos du modèle"])

with t1:
    idx = st.selectbox("Employé (numéro de ligne du jeu préparé)", list(df.index))
    row = df.loc[[idx]]
    show_prediction(row)
    status = "promu" if int(row[U.E].iloc[0]) == 1 else "censuré (pas encore promu)"
    st.caption(f"Dans les données : durée observée = {row[U.T].iloc[0]:.1f} an(s), statut = {status}.")
    st.dataframe(row[FEATS].T.rename(columns={idx: 'valeur'}).astype(str))

with t2:
    st.write("Renseignez le profil (les valeurs par défaut sont les valeurs typiques du jeu de données).")
    vals = {}
    cols = st.columns(3)
    for i, c in enumerate(FEATS):
        s = df[c]
        box = cols[i % 3]
        if s.nunique() <= 2:
            vals[c] = int(box.checkbox(c, value=bool(round(s.mean())), key=f"f_{c}"))
        else:
            lo, hi = float(s.min()), float(s.max())
            vals[c] = box.number_input(c, lo, hi, float(s.median()), step=1.0, key=f"f_{c}")
    if st.button("Estimer le délai", type="primary"):
        show_prediction(pd.DataFrame([vals]))

with t3:
    st.write("Déposez un CSV contenant au minimum les colonnes suivantes : " + ", ".join(f"`{c}`" for c in FEATS) + ".")
    up = st.file_uploader("Fichier CSV", type="csv")
    if up is not None:
        d = pd.read_csv(up, encoding='utf-8-sig')
        miss = [c for c in FEATS if c not in d.columns]
        if miss:
            st.error(f"Colonnes manquantes : {miss}")
        else:
            out = d.copy()
            out['delai_median_ans'] = U.median_wait(art, d).round(2)
            st.metric("Délai médian moyen", f"{out['delai_median_ans'].mean():.1f} ans")
            st.dataframe(out.sort_values('delai_median_ans').head(50))
            st.download_button("Télécharger les résultats", out.to_csv(index=False).encode('utf-8'), "delais_promotion.csv")

with t4:
    st.metric("C-index test (AFT log-logistique)", f"{U.CINDEX['Modèle 2 : AFT log-logistique']:.3f}")
    st.markdown(f"""
**Modèle retenu** : AFT log-logistique, entraîné sur {art['n_train']} employés du jeu d'entraînement ({art['event_rate']:.0%} promus, le reste est censuré), décalage des durées de {art['shift']} an.

**Pourquoi l'AFT plutôt que Cox**

1. Meilleur C-index de test, mais l'écart reste faible face à l'écart-type de validation croisée (0,02 à 0,03).
2. Pas d'hypothèse de proportionnalité des risques, violée dans Cox pour `JobLevel` et `YearsInCurrentRole`.
3. Il prédit directement un temps d'attente, alors que Cox donne des paliers entiers.
""")
    st.subheader("Comparaison des modèles (jeu de test)")
    st.bar_chart(pd.Series(U.CINDEX, name='C-index test'), horizontal=True)
    st.subheader("Effet de chaque variable (time ratio)")
    st.caption("Un time ratio supérieur à 1 allonge l'attente avant la promotion, inférieur à 1 la raccourcit.")
    tr = U.time_ratios(art)
    st.dataframe(tr.style.format({'time ratio': '{:.2f}', 'p-value': '{:.3f}'}))
    st.markdown("""
**Limites à connaître**

- L'événement « promu » est construit : il repose sur l'hypothèse faite lors de la préparation des données.
- `YearsInCurrentRole` est liée mécaniquement à la durée : à interpréter avec prudence.
- Seulement 12 % d'employés censurés, et un seul découpage entraînement/test.
- Données d'exemple (jeu pédagogique IBM) : ne pas utiliser pour des décisions réelles sans validation sur les données de l'entreprise.
""")
