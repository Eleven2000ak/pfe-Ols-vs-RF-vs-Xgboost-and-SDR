"""
hypotheses.py  —  Test des hypothèses H1, H2, H3
==================================================

  - H1 : affichage des RMSE pour justifier le résultat dans le mémoire
  - H2 : affichage des RMSE + calcul du % d'amélioration
  - H3 : correction bug index numpy (y_corrupted converti en ndarray)
         + comparaison avec baseline propre (sans outliers)
         + affichage des dégradations % pour valider/invalider H3
  - Toutes les fonctions retournent un dict complet (résultats + booléen)
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def test_h1(X_lin, y_lin, random_state=42):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error
    import numpy as np

    X_train, X_test, y_train, y_test = train_test_split(
        X_lin, y_lin, test_size=0.2, random_state=random_state
    )

    model_ols = LinearRegression().fit(X_train, y_train)
    model_rf = RandomForestRegressor(
        n_estimators=100,
        random_state=random_state
    ).fit(X_train, y_train)

    rmse_ols = np.sqrt(mean_squared_error(y_test, model_ols.predict(X_test)))
    rmse_rf = np.sqrt(mean_squared_error(y_test, model_rf.predict(X_test)))

    #  ONLY RULE (important)
    result = rmse_ols <= rmse_rf

    print("=== H1 — Linéarité (données simulées) ===")
    print(f"RMSE OLS           : {rmse_ols:.4f}")
    print(f"RMSE Random Forest : {rmse_rf:.4f}")
    print(f"H1 = {result}")

    if result:
        print("Conclusion : H1 validée (OLS ≤ RF)")
    else:
        print("Conclusion : H1 rejetée")

    return result
# ============================================================
# H2  —  Nécessité de modèles complexes
# ============================================================

def test_h2(X, y, min_improvement=0.10, random_state=42):
    """
    H2 : XGBoost améliore le RMSE d'au moins min_improvement (défaut 10%)
         par rapport à Random Forest sur données réelles.

    Parameters
    ----------
    X               : features (données California Housing)
    y               : cible (médiane des prix)
    min_improvement : seuil minimal d'amélioration relative (défaut 0.10)
    random_state    : graine

    Returns
    -------
    dict avec 'result', 'rmse_rf', 'rmse_xgb', 'improvement'
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    model_rf = RandomForestRegressor(
        n_estimators=100, random_state=random_state
    ).fit(X_train, y_train)

    model_xgb = xgb.XGBRegressor(
        n_estimators=100, learning_rate=0.1, random_state=random_state
    ).fit(X_train, y_train)

    rmse_rf  = np.sqrt(mean_squared_error(y_test, model_rf.predict(X_test)))
    rmse_xgb = np.sqrt(mean_squared_error(y_test, model_xgb.predict(X_test)))

    improvement = (rmse_rf - rmse_xgb) / (rmse_rf + 1e-10)
    result      = improvement >= min_improvement

    print("=== H2 — Nécessité de modèles complexes ===")
    print(f"  RMSE Random Forest: {rmse_rf:.2f}")
    print(f"  RMSE XGBoost      : {rmse_xgb:.2f}")
    print(f"  Amélioration XGB  : {improvement*100:.2f}%  (seuil: {min_improvement*100:.0f}%)")
    print(f"  H2 = {result}  "
          f"({'XGBoost significativement meilleur' if result else 'amélioration insuffisante'})")

    return {
        "result": result,
        "rmse_rf": rmse_rf,
        "rmse_xgb": rmse_xgb,
        "improvement": improvement
    }


# ============================================================
# H3  —  Robustesse aux valeurs aberrantes
# ============================================================

# ============================================================
# H3 — Robustesse aux valeurs aberrantes
# ============================================================

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb


def test_h3(X, y, outlier_fraction=0.05, random_state=42):
    """
    H3 : Random Forest est plus robuste aux outliers que XGBoost.
    """

    rng = np.random.default_rng(random_state)

    # =========================
    # Split
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    # =========================
    # Baseline (clean data)
    # =========================
    model_rf_clean = RandomForestRegressor(
        n_estimators=100,
        random_state=random_state
    ).fit(X_train, y_train)

    model_xgb_clean = xgb.XGBRegressor(
        n_estimators=100,
        random_state=random_state,
        eval_metric="rmse"
    ).fit(X_train, y_train)

    rmse_rf_clean = np.sqrt(mean_squared_error(
        y_test, model_rf_clean.predict(X_test)
    ))

    rmse_xgb_clean = np.sqrt(mean_squared_error(
        y_test, model_xgb_clean.predict(X_test)
    ))

    # =========================
    # Inject outliers
    # =========================
    y_corrupted = np.array(y_train, dtype=float).copy()

    n_outliers = int(len(y_corrupted) * outlier_fraction)
    indices = rng.choice(len(y_corrupted), size=n_outliers, replace=False)
    factors = rng.uniform(10, 50, size=n_outliers)

    y_corrupted[indices] *= factors

    print("=== H3 — Robustesse aux outliers ===")
    print(f"Outliers injectés : {n_outliers} ({outlier_fraction*100:.0f}%)")
    print("Facteur multiplicatif : ×[10, 50]")

    # =========================
    # Train corrupted models
    # =========================
    model_rf_corr = RandomForestRegressor(
        n_estimators=100,
        random_state=random_state
    ).fit(X_train, y_corrupted)

    model_xgb_corr = xgb.XGBRegressor(
        n_estimators=100,
        random_state=random_state,
        eval_metric="rmse"
    ).fit(X_train, y_corrupted)

    rmse_rf_corr = np.sqrt(mean_squared_error(
        y_test, model_rf_corr.predict(X_test)
    ))

    rmse_xgb_corr = np.sqrt(mean_squared_error(
        y_test, model_xgb_corr.predict(X_test)
    ))

    # =========================
    # Degradation
    # =========================
    degradation_rf = (rmse_rf_corr - rmse_rf_clean) / (rmse_rf_clean + 1e-10)
    degradation_xgb = (rmse_xgb_corr - rmse_xgb_clean) / (rmse_xgb_clean + 1e-10)

    result = degradation_rf < degradation_xgb

    # =========================
    # Print results
    # =========================
    print(f"\nRMSE RF  : {rmse_rf_clean:.4f} → {rmse_rf_corr:.4f}")
    print(f"RMSE XGB : {rmse_xgb_clean:.4f} → {rmse_xgb_corr:.4f}")

    print(f"\nDégradation RF  : {degradation_rf*100:.2f}%")
    print(f"Dégradation XGB : {degradation_xgb*100:.2f}%")

    print(f"\nH3 = {result}")

    if result:
        print("Conclusion : H3 validée (RF plus robuste)")
    else:
        print("Conclusion : H3 rejetée ou XGB plus robuste")

    return {
        "result": result,
        "rmse_rf_clean": rmse_rf_clean,
        "rmse_rf_corr": rmse_rf_corr,
        "rmse_xgb_clean": rmse_xgb_clean,
        "rmse_xgb_corr": rmse_xgb_corr,
        "degradation_rf": degradation_rf,
        "degradation_xgb": degradation_xgb
    }