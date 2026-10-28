"""BO3 · Planifier les promotions : prédictions avec le modèle AFT log-logistique exporté par le notebook.
Aucune dépendance à lifelines : seulement numpy et pandas."""
import json
import numpy as np
import pandas as pd

DATA = 'data/BO3_prepared.csv'
MODEL = 'data/model_bo3.json'
T, E = 'duration', 'event_promoted'

# Scores du notebook (jeu de test)
CINDEX = {
    'Modèle 1 : Cox': 0.740,
    'Modèle 1 bis : Cox stratifié sur JobLevel': 0.748,
    'Modèle 2 : AFT log-logistique': 0.755,
}


def load_prepared(path=DATA):
    df = pd.read_csv(path, encoding='utf-8-sig')
    for c in df.columns:
        if df[c].dtype == bool:
            df[c] = df[c].astype(int)
    return df


def load_model(path=MODEL):
    with open(path, encoding='utf-8') as f:
        m = json.load(f)
    m['coefs'] = np.array([m['alpha_coef'][c] for c in m['features']], dtype=float)
    m['beta'] = float(np.exp(m['beta_intercept']))
    return m


def _alpha(art, X):
    X = X[art['features']].astype(float).to_numpy()
    return np.exp(art['alpha_coef']['Intercept'] + X @ art['coefs'])


def median_wait(art, X):
    """Délai médian avant promotion, en années (décalage retiré)."""
    return np.clip(_alpha(art, X) - art['shift'], 0, None)


def survival_curve(art, X, horizon=15, n=100):
    """Probabilité de ne pas encore être promu, en fonction du temps (première ligne de X)."""
    a = _alpha(art, X.iloc[[0]])[0]
    t = np.linspace(0, horizon, n)
    s = 1.0 / (1.0 + ((t + art['shift']) / a) ** art['beta'])
    return pd.Series(s, index=np.round(t, 2), name='Pas encore promu')


def prob_promoted_within(art, X, years=(1, 2, 3, 5)):
    a = _alpha(art, X.iloc[[0]])[0]
    return {y: float(1 - 1.0 / (1.0 + ((y + art['shift']) / a) ** art['beta'])) for y in years}


def time_ratios(art):
    """Time ratio = exp(coef) (>1 : attente plus longue)."""
    rows = {c: {'time ratio': float(np.exp(art['alpha_coef'][c])), 'p-value': art['pvalues'].get(c, np.nan)}
            for c in art['features']}
    return pd.DataFrame(rows).T
