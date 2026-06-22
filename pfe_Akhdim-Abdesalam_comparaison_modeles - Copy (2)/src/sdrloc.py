def compute_SDRloc(
    model,
    X_train,
    y_train,
    x_point,
    k=50,
    B=300,
    lam=1.0,
    background_size=100,
    random_state=42
):
    import numpy as np
    import pandas as pd
    from sklearn.neighbors import NearestNeighbors
    from sklearn.metrics import mean_squared_error
    from sklearn.impute import SimpleImputer
    from scipy.stats import entropy
    import shap

    np.random.seed(random_state)

    # =========================
    # 1. ALIGN
    # =========================
    X_train = X_train.reindex(columns=model.feature_names_in_, fill_value=0)
    x_point = x_point.reindex(columns=model.feature_names_in_, fill_value=0)

    y_train = np.array(y_train).reshape(-1)

    # =========================
    # 2. IMPUTE
    # =========================
    imputer = SimpleImputer(strategy="median")
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    x_point = pd.DataFrame(imputer.transform(x_point), columns=X_train.columns)

    # =========================
    # 3. NEIGHBORS
    # =========================
    nbrs = NearestNeighbors(n_neighbors=k).fit(X_train)
    idx = nbrs.kneighbors(x_point, return_distance=False)[0]

    X_neighbors = X_train.iloc[idx]
    y_neighbors = y_train[idx]

    # =========================
    # 4. ACCURACY (PDF)
    # =========================
    preds = model.predict(X_neighbors)
    mse = mean_squared_error(y_neighbors, preds)

    A = 1.0 / (1.0 + mse)

    # =========================
    # 5. SHAP
    # =========================
    background = X_train.sample(
        n=min(background_size, len(X_train)),
        random_state=random_state
    )

    explainer = shap.TreeExplainer(model, data=background)

    shap_local = explainer.shap_values(X_neighbors, check_additivity=False)
    shap_global = explainer.shap_values(background, check_additivity=False)

    if isinstance(shap_local, list):
        shap_local = shap_local[0]
        shap_global = shap_global[0]

    # =========================
    # 6. STABILITY (PDF)
    # =========================
    shap_boot = []

    for _ in range(B):
        idx_boot = np.random.choice(len(X_neighbors), len(X_neighbors), replace=True)
        X_boot = X_neighbors.iloc[idx_boot]

        shap_vals = explainer.shap_values(X_boot, check_additivity=False)

        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]

        shap_boot.append(np.mean(shap_vals, axis=0))

    shap_boot = np.array(shap_boot)

    var_j = np.var(shap_boot, axis=0)
    V = np.sum(var_j)

    S = np.exp(-V)

    # =========================
    # 7. DIVERGENCE (PDF)
    # =========================
    imp_local = np.abs(shap_local).mean(axis=0)

    p_local = imp_local / (imp_local.sum() + 1e-10)

    d = len(p_local)
    q_uniform = np.ones(d) / d

    kl = entropy(p_local + 1e-10, q_uniform + 1e-10)

    K = kl

    # =========================
    # 8. FINAL
    # =========================
    SDR = (A * S) / (1 + lam * K)

    return SDR, A, S, K, kl