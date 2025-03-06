import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def load_data():
    """Charge les fichiers CSV et retourne les DataFrames."""
    clinical_train = pd.read_csv("../data/X_train/clinical_train.csv")
    molecular_train = pd.read_csv("../data/X_train/molecular_train.csv")
    target_train = pd.read_csv("../data/X_train/target_train.csv")

    clinical_test = pd.read_csv("../data/X_test/clinical_test.csv")
    molecular_test = pd.read_csv("../data/X_test/molecular_test.csv")

    return (
        clinical_train,
        molecular_train,
        target_train,
        clinical_test,
        molecular_test,
    )


def clean_data(clinical_train, clinical_test):
    """Nettoie les valeurs manquantes et encode les variables catégorielles."""
    num_cols = ["BM_BLAST", "WBC", "ANC", "MONOCYTES", "HB", "PLT"]
    for col in num_cols:
        median_value = clinical_train[col].median()
        clinical_train[col].fillna(median_value, inplace=True)
        clinical_test[col].fillna(median_value, inplace=True)

    clinical_train["CYTOGENETICS"].fillna("UNKNOWN", inplace=True)
    clinical_test["CYTOGENETICS"].fillna("UNKNOWN", inplace=True)

    # Encodage One-Hot de CENTER
    encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
    center_encoded_train = encoder.fit_transform(clinical_train[["CENTER"]])
    center_encoded_test = encoder.transform(clinical_test[["CENTER"]])

    center_columns = [f"CENTER_{cat}" for cat in encoder.categories_[0]]
    center_df_train = pd.DataFrame(
        center_encoded_train,
        columns=center_columns,
        index=clinical_train.index,
    )
    center_df_test = pd.DataFrame(
        center_encoded_test, columns=center_columns, index=clinical_test.index
    )

    clinical_train = pd.concat(
        [clinical_train.drop(columns=["CENTER"]), center_df_train], axis=1
    )
    clinical_test = pd.concat(
        [clinical_test.drop(columns=["CENTER"]), center_df_test], axis=1
    )

    return clinical_train, clinical_test
