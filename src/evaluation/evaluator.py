# src/evaluation/evaluator.py

import numpy as np
import pandas as pd
from sksurv.metrics import concordance_index_ipcw
from sksurv.util import Surv


class Evaluator:
    def __init__(self, tau: float = 7.0):
        """
        Initialise l'évaluateur avec une valeur tau (par défaut 7 ans).
        """
        self.tau = tau

    def evaluate(self, y_true, predictions: np.array) -> float:
        """
        Calcule l'IPCW-C-index pour un ensemble de données.

        Parameters:
            y_true: soit un DataFrame contenant 'OS_STATUS' et 'OS_YEARS',
                    soit un tableau structuré déjà obtenu via Surv.from_dataframe.
            predictions (np.array): Tableau des scores de risque prédits.

        Returns:
            float: L'IPCW-C-index.
        """
        if isinstance(y_true, pd.DataFrame):
            y_true = y_true.dropna(subset=["OS_YEARS", "OS_STATUS"])
            survival_data = Surv.from_dataframe(
                "OS_STATUS", "OS_YEARS", y_true
            )
        else:
            survival_data = y_true

        c_index, _ = concordance_index_ipcw(
            survival_data, survival_data, predictions, self.tau
        )
        return c_index

    def evaluate_both(
        self, y_train, pred_train: np.array, y_test, pred_test: np.array
    ) -> tuple:
        # Traitement de y_train
        if isinstance(y_train, pd.DataFrame):
            y_train = y_train.dropna(subset=["OS_YEARS", "OS_STATUS"])
            surv_train = Surv.from_dataframe("OS_STATUS", "OS_YEARS", y_train)
        else:
            surv_train = y_train
            if np.any(np.isnan(surv_train["OS_YEARS"])) or np.any(
                np.isnan(surv_train["OS_STATUS"])
            ):
                raise ValueError("Surv object for training contains NaN")

        # Traitement de y_test
        if isinstance(y_test, pd.DataFrame):
            y_test = y_test.dropna(subset=["OS_YEARS", "OS_STATUS"])
            surv_test = Surv.from_dataframe("OS_STATUS", "OS_YEARS", y_test)
        else:
            surv_test = y_test
            if np.any(np.isnan(surv_test["OS_YEARS"])) or np.any(
                np.isnan(surv_test["OS_STATUS"])
            ):
                raise ValueError("Surv object for test contains NaN")

        train_cindex = concordance_index_ipcw(
            surv_train, surv_train, pred_train, self.tau
        )[0]
        test_cindex = concordance_index_ipcw(
            surv_test, surv_test, pred_test, self.tau
        )[0]
        return train_cindex, test_cindex
