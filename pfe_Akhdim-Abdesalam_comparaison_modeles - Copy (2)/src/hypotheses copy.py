import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Hypothèse 1
def test_h1(X_lin, y_lin):
    """
    H1: La différence de RMSE entre RF et OLS est inférieure à 5% sur données linéaires
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X_lin, y_lin, test_size=0.2, random_state=42
    )

    model_ols = LinearRegression().fit(X_train, y_train)
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)

    rmse_ols = np.sqrt(mean_squared_error(y_test, model_ols.predict(X_test)))
    rmse_rf = np.sqrt(mean_squared_error(y_test, model_rf.predict(X_test)))

    return abs(rmse_rf - rmse_ols) / rmse_ols < 0.05


# Hypothèse 2
def test_h2(X, y):
    """
    H2: XGBoost améliore la performance (RMSE) d'au moins 10% par rapport à RF
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model_rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
    model_xgb = xgb.XGBRegressor(
        n_estimators=100, learning_rate=0.1, random_state=42
    ).fit(X_train, y_train)

    rmse_rf = np.sqrt(mean_squared_error(y_test, model_rf.predict(X_test)))
    rmse_xgb = np.sqrt(mean_squared_error(y_test, model_xgb.predict(X_test)))

    return (rmse_rf - rmse_xgb) / rmse_rf >= 0.10


# Hypothèse 3
def test_h3(X, y, outlier_fraction=0.05):
    """
    H3: Random Forest est plus robuste aux valeurs aberrantes que XGBoost
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Injection d'outliers
    y_corrupted = y_train.copy()
    n_outliers = int(len(y_train) * outlier_fraction)
    indices = np.random.choice(len(y_train), n_outliers, replace=False)

    y_corrupted[indices] = y_corrupted[indices] * np.random.uniform(10, 50, n_outliers)

    #y_corrupted.iloc[indices] = y_corrupted.iloc[indices] * np.random.uniform(10, 50, n_outliers)

    model_rf = RandomForestRegressor(random_state=42).fit(X_train, y_corrupted)
    model_xgb = xgb.XGBRegressor(random_state=42).fit(X_train, y_corrupted)

    rmse_rf = np.sqrt(mean_squared_error(y_test, model_rf.predict(X_test)))
    rmse_xgb = np.sqrt(mean_squared_error(y_test, model_xgb.predict(X_test)))

    return rmse_rf < rmse_xgb