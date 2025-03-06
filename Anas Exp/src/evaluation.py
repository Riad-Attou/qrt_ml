from sksurv.metrics import concordance_index_censored


def evaluate_model(model, X_test, y_test):
    """Calcule le C-index pour évaluer le modèle."""

    risk_scores = model.predict(X_test)
    c_index = concordance_index_censored(
        y_test["OS_STATUS"] == 1, y_test["OS_YEARS"], risk_scores
    )[0]

    return c_index
