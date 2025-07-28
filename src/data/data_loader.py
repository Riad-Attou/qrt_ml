import pandas as pd


class DataLoader:
    def __init__(
        self,
        clinical_path_train: str,
        clinical_path_test: str,
        molecular_path_train: str,
        molecular_path_test: str,
        target_path_train: str,
    ):
        self.clinical_path_train = clinical_path_train
        self.clinical_path_test = clinical_path_test
        self.molecular_path_train = molecular_path_train
        self.molecular_path_test = molecular_path_test
        self.target_path_train = target_path_train

    def load_clinical_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        # Charger les données cliniques
        df_train = pd.read_csv(self.clinical_path_train, sep=",")
        df_train["ID"] = df_train["ID"].astype(str)
        df_test = pd.read_csv(self.clinical_path_test, sep=",")
        df_test["ID"] = df_test["ID"].astype(str)

        return df_train, df_test

    def load_molecular_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        # Charger les données moléculaires
        df_train = pd.read_csv(self.molecular_path_train, sep=",")
        df_train["ID"] = df_train["ID"].astype(str)
        df_test = pd.read_csv(self.molecular_path_test, sep=",")
        df_test["ID"] = df_test["ID"].astype(str)

        return df_train, df_test

    def load_target_data(self) -> pd.DataFrame:
        # Charger les targets
        df_train = pd.read_csv(self.target_path_train, sep=",")
        df_train = df_train[df_train["ID"].isin(df_train["ID"])]
        df_train["OS_YEARS"] = pd.to_numeric(
            df_train["OS_YEARS"], errors="coerce"
        )
        df_train["OS_STATUS"] = df_train["OS_STATUS"].astype(bool)

        return df_train

    def load_all(self):
        # Charger toutes les données
        clinical_train, clinical_test = self.load_clinical_data()
        molecular_train, molecular_test = self.load_clinical_data()
        target = self.load_target_data()
        return (
            clinical_train,
            clinical_test,
            molecular_train,
            molecular_test,
            target,
        )
