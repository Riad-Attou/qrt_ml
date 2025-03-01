import lightgbm as lgb
import numpy as np
import pandas as pd


class LightGBMModel:
    def __init__(
        self,
        params: dict = None,
        num_boost_round: int = 1000,
        early_stopping_rounds: int = 50,
        verbose_eval: int = 50,
    ):
        """
        Initialise le modèle LightGBM avec les paramètres souhaités.
        """
        # Paramètres par défaut si aucun n'est fourni
        self.params = params or {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "learning_rate": 0.01,
            "num_leaves": 31,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
        }
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose_eval = verbose_eval
        self.model = None

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: np.array,
        X_valid: pd.DataFrame = None,
        y_valid: np.array = None,
    ):
        """
        Entraîne le modèle sur les données d'entraînement, avec option de validation.
        """
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_sets = [train_data]
        valid_names = ["train"]

        if X_valid is not None and y_valid is not None:
            valid_data = lgb.Dataset(X_valid, label=y_valid)
            valid_sets.append(valid_data)
            valid_names.append("valid")

        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=self.num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=self.verbose_eval,
        )
        return self.model

    def predict(self, X: pd.DataFrame) -> np.array:
        """
        Prédit avec le modèle entraîné.
        """
        if self.model is None:
            raise ValueError("Le modèle n'est pas encore entraîné.")
        return self.model.predict(X, num_iteration=self.model.best_iteration)
