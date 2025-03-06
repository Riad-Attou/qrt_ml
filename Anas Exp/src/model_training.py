import numpy as np
from sksurv.linear_model import CoxPHSurvivalAnalysis


def train_cox_model(X, y):
    """Entraîne un modèle de Cox."""

    y_survival = [
        (bool(status), time)
        for status, time in zip(y["OS_STATUS"], y["OS_YEARS"])
    ]
    y_survival = np.array(y_survival, dtype=[("event", bool), ("time", float)])

    cox_model = CoxPHSurvivalAnalysis()
    cox_model.fit(X, y_survival)

    return cox_model
