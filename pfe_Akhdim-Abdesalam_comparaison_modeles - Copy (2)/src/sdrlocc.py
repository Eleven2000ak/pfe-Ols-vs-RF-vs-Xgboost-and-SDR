def compute_SDRloc(
    model,
    X_train,
    y_train,
    x_point,
    k=50,
    B=30,
    lam=1,
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

    rng = np.random.default_rng(random_state)

    # =========================
    # 1. ALIGN
    # =========================
    cols = list(model.feature_names_in_)
    X_train = X_train.reindex(columns=cols, fill_value=0)
    x_point = x_point.reindex(columns=cols, fill_value=0)

    y_train = np.array(y_train).reshape(-1)

    # =========================
    # 2. IMPUTATION
    # =========================
    imputer = SimpleImputer(strategy="median")
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=cols)
    x_point = pd.DataFrame(imputer.transform(x_point), columns=cols)

    # =========================
    # 3. NEIGHBORS
    # =========================
    nbrs = NearestNeighbors(n_neighbors=k).fit(X_train)
    idx = nbrs.kneighbors(x_point, return_distance=False)[0]

    X_neighbors = X_train.iloc[idx]
    y_neighbors = y_train[idx]

    # =========================
    # 4. ACCURACY (normalized)
    # =========================
    preds = model.predict(X_neighbors)

    rmse = np.sqrt(mean_squared_error(y_neighbors, preds))
    sigma_y = np.std(y_train) + 1e-10

    A = np.exp(-(rmse / sigma_y) ** 2)

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
    # 6. STABILITY (robust)
    # =========================
    shap_boot = []

    for _ in range(B):
        idx_boot = rng.choice(len(X_neighbors), len(X_neighbors), replace=True)
        X_boot = X_neighbors.iloc[idx_boot]

        shap_vals = explainer.shap_values(X_boot, check_additivity=False)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]

        # 🔥 normalization per feature
        shap_vals = shap_vals / (np.std(shap_vals, axis=0) + 1e-10)

        shap_boot.append(np.abs(shap_vals).mean(axis=0))

    shap_boot = np.array(shap_boot)

    var_mean = np.mean(np.var(shap_boot, axis=0))
    var_mean = np.clip(var_mean, 0, 3)

    S = np.exp(-var_mean)

    # =========================
    # 7. DIVERGENCE (stable)
    # =========================
    imp_local = np.abs(shap_local).mean(axis=0)
    imp_global = np.abs(shap_global).mean(axis=0)

    p_local = imp_local / (imp_local.sum() + 1e-10)
    p_global = imp_global / (imp_global.sum() + 1e-10)

    kl = entropy(p_local + 1e-10, p_global + 1e-10)
    kl = np.clip(kl, 0, 3)

    K = 1 - np.exp(-kl)

    # =========================
    # 8. FINAL SDRloc
    # =========================
    SDR = (A * S) / (1 + lam * K)

    return SDR, A, S, K, kl