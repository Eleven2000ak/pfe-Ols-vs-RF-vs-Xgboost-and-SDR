import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Root Mean Squared Error
def compute_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Mean Absolute Error
def compute_mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

# Coefficient de détermination R²
def compute_r2(y_true, y_pred):
    return r2_score(y_true, y_pred)

# Mean Absolute Percentage Error
def compute_mape(y_true, y_pred):
    return 100 * np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8)))

# Affichage des métriques
def display_metrics(y_true, y_pred, model_name):
    rmse_val = compute_rmse(y_true, y_pred)
    mae_val = compute_mae(y_true, y_pred)
    r2_val = compute_r2(y_true, y_pred)
    mape_val = compute_mape(y_true, y_pred)

    print(f"{model_name} → RMSE={rmse_val:.4f} | MAE={mae_val:.4f} | R²={r2_val:.4f} | MAPE={mape_val:.2f}%")