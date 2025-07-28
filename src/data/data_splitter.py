# src/data/data_splitter.py

import numpy as np
import pandas as pd
from scipy.stats import lognorm, weibull_min


class DataSplitter:
    def __init__(self, nb_classes: int, ignore_os_status_zero: bool = True):
        self.nb_classes = nb_classes
        self.ignore_os_status_zero = ignore_os_status_zero

    def split_by_distribution(
        self, df: pd.DataFrame, distribution: str = "lognorm"
    ) -> list[list[str]]:
        # Filtrer les patients en fonction d'OS_STATUS si besoin
        if self.ignore_os_status_zero:
            df = df[df["OS_STATUS"] != 0]
        df = df[np.isfinite(df["OS_YEARS"])]

        os_years = df["OS_YEARS"].values

        # Choix de la distribution pour la transformation
        if distribution == "lognorm":
            shape, loc, scale = lognorm.fit(os_years, floc=0)
            df["cdf"] = lognorm.cdf(os_years, shape, loc=loc, scale=scale)
        elif distribution == "weibull":
            shape, loc, scale = weibull_min.fit(os_years, floc=0)
            df["cdf"] = weibull_min.cdf(os_years, shape, loc=loc, scale=scale)
        else:
            # Utiliser un tri simple si aucun modèle n'est spécifié
            df["cdf"] = df["OS_YEARS"]

        # Trier par la transformation
        df_sorted = df.sort_values(by="cdf")
        patient_ids = df_sorted["ID"].tolist()

        # Diviser en nb_classes groupes de taille égale (ou presque)
        n = len(patient_ids)
        classes = []
        base_size = n // self.nb_classes
        remainder = n % self.nb_classes
        start = 0
        for i in range(self.nb_classes):
            size = base_size + (1 if i < remainder else 0)
            classes.append(patient_ids[start : start + size])
            start += size

        return classes
