import numpy as np
import pandas as pd
from sksurv.metrics import concordance_index_ipcw
from sksurv.util import Surv


class Evaluator:
    def __init__(self, tau: float = 7.0):
        """
        Initialise l'évaluateur avec une valeur tau, par exemple pour tronquer le suivi à 7 ans.
        """
        self.tau = tau

    def prepare_survival_data(self, y: pd.DataFrame) -> np.array:
        """
        Transforme un DataFrame contenant 'OS_STATUS' (booléen) et 'OS_YEARS' en un tableau structuré.
        """
        # On utilise Surv.from_dataframe pour créer la structure attendue
        survival_data = Surv.from_dataframe("OS_STATUS", "OS_YEARS", y)
        return survival_data

    def evaluate(self, y_true: pd.DataFrame, predictions: np.array) -> float:
        """
        Calcule le IPCW-C-index pour évaluer l'ordre des prédictions en fonction des temps de survie réels.

        Parameters:
          - y_true : DataFrame contenant 'OS_STATUS' et 'OS_YEARS'.
          - predictions : Array des scores de risque prédits par le modèle.

        Returns:
          - Le IPCW-C-index (valeur entre 0 et 1).
        """
        # Préparer les données de survie
        survival_data = self.prepare_survival_data(y_true)

        # Pour calculer le IPCW-C-index, on passe les données de survie deux fois (une fois pour les temps de suivi)
        c_index, _ = concordance_index_ipcw(
            training_survival_data=survival_data,  # dans un contexte réel, ce pourrait être le jeu d'entraînement
            test_survival_data=survival_data,  # ici, on évalue sur le même ensemble pour l'exemple
            estimate=predictions,
            tau=self.tau,
        )
        return c_index
