"""
evaluation.py  —  Métriques d'évaluation
==========================================
Corrections apportées (Prof. K. ELFASSI) :
  - Toutes les métriques précisent explicitement l'ensemble de calcul
  - Ajout de compute_metrics_full() pour diagnostiquer l'overfitting
    (métriques train ET test en un seul appel)
  - Ajout compute_bootstrap_ci() : intervalles de confiance bootstrap
    sur RMSE (B=1000) pour tester la significativité des différences
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ------------------------------------------------------------------
# Métriques individuelles
# ------------------------------------------------------------------

def compute_rmse(y_true, y_pred):
    """RMSE sur l'ensemble fourni (train ou test)."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def compute_mae(y_true, y_pred):
    """MAE sur l'ensemble fourni."""
    return mean_absolute_error(y_true, y_pred)


def compute_r2(y_true, y_pred):
    """R² sur l'ensemble fourni."""
    return r2_score(y_true, y_pred)


def compute_mape(y_true, y_pred):
    """MAPE (%) sur l'ensemble fourni."""
    return 100 * np.mean(np.abs((np.array(y_true) - np.array(y_pred))
                                / (np.array(y_true) + 1e-8)))


# ------------------------------------------------------------------
# Affichage simple
# ------------------------------------------------------------------

def display_metrics(y_true, y_pred, model_name, split="test"):
    """
    Affiche les métriques pour un ensemble donné.

    Parameters
    ----------
    split : str
        Nom de l'ensemble ('train' ou 'test') — affiché dans le header
    """
    rmse_val = compute_rmse(y_true, y_pred)
    mae_val  = compute_mae(y_true, y_pred)
    r2_val   = compute_r2(y_true, y_pred)
    mape_val = compute_mape(y_true, y_pred)

    print(f"{model_name} [{split}] → "
          f"RMSE={rmse_val:.2f} | MAE={mae_val:.2f} | "
          f"R²={r2_val:.4f} | MAPE={mape_val:.2f}%")

    return {"RMSE": rmse_val, "MAE": mae_val, "R2": r2_val, "MAPE": mape_val}


# ------------------------------------------------------------------
# Diagnostic overfitting  (CORRECTION PROF. ELFASSI)
# ------------------------------------------------------------------

def compute_metrics_full(model, X_train, y_train, X_test, y_test, model_name):
    """
    Calcule et affiche les métriques sur TRAIN et TEST.
    Permet de diagnostiquer le surapprentissage.

    Returns
    -------
    dict avec clés 'train' et 'test', chacune contenant RMSE, MAE, R², MAPE
    """
    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)

    metrics_train = display_metrics(y_train, y_pred_train, model_name, split="train")
    metrics_test  = display_metrics(y_test,  y_pred_test,  model_name, split="test")

    gap_r2   = metrics_train["R2"]   - metrics_test["R2"]
    gap_rmse = metrics_test["RMSE"]  - metrics_train["RMSE"]

    print(f"  └─ Écart R²  (train-test) : {gap_r2:+.4f}  "
          f"{'⚠ surapprentissage' if gap_r2 > 0.05 else '✓ stable'}")
    print(f"  └─ Écart RMSE (test-train): {gap_rmse:+.2f}")

    return {"train": metrics_train, "test": metrics_test}


# ------------------------------------------------------------------
# Intervalles de confiance bootstrap  (CORRECTION PROF. ELFASSI)
# ------------------------------------------------------------------

def compute_bootstrap_ci(y_true, y_pred, B=1000, alpha=0.05, random_state=42):
    """
    Calcule un intervalle de confiance à (1-alpha)*100 % sur le RMSE
    par bootstrap (B tirages avec remise sur l'ensemble de test).

    Parameters
    ----------
    B      : int   — nombre d'itérations bootstrap (défaut 1000)
    alpha  : float — niveau de risque (défaut 0.05 → IC 95 %)

    Returns
    -------
    (rmse_obs, lower, upper) : tuple de floats
    """
    rng = np.random.default_rng(random_state)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n = len(y_true)

    boot_rmse = np.zeros(B)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        boot_rmse[b] = np.sqrt(mean_squared_error(y_true[idx], y_pred[idx]))

    rmse_obs = compute_rmse(y_true, y_pred)
    lower    = np.percentile(boot_rmse, 100 * alpha / 2)
    upper    = np.percentile(boot_rmse, 100 * (1 - alpha / 2))

    print(f"RMSE observé : {rmse_obs:.2f}  |  "
          f"IC {int((1-alpha)*100)}% bootstrap : [{lower:.2f}, {upper:.2f}]")

    return rmse_obs, lower, upper


def compare_models_bootstrap(y_test, preds_dict, B=1000, random_state=42):
    """
    Compare plusieurs modèles via bootstrap RMSE avec IC 95%.

    Parameters
    ----------
    preds_dict : dict  {model_name: y_pred_array}

    Returns
    -------
    DataFrame avec RMSE, IC_lower, IC_upper pour chaque modèle
    """
    import pandas as pd

    results = {}
    for name, y_pred in preds_dict.items():
        rmse, lo, hi = compute_bootstrap_ci(
            y_test, y_pred, B=B, random_state=random_state
        )
        results[name] = {"RMSE": rmse, "IC_lower_95": lo, "IC_upper_95": hi}
        print(f"  {name:20s} : RMSE={rmse:.2f}  IC95%=[{lo:.2f}, {hi:.2f}]")

    return pd.DataFrame(results).T
