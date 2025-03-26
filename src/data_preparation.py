import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from data_preprocessing import clean_data, load_data
from feature_engineering import aggregate_molecular_features, merge_features

DATA_PROCESSED_PATH = "data/processed/"


def prepare_data():
    print(
        "Préparation des données (approche simplifiée basée sur le benchmark)..."
    )

    # Chargement des données
    (
        clinical_train,
        molecular_train,
        target_train,
        clinical_test,
        molecular_test,
    ) = load_data()

    # Nettoyage des données cliniques
    clinical_train, clinical_test = clean_data(clinical_train, clinical_test)

    # Agrégation des caractéristiques moléculaires (réduite à Nmut uniquement)
    molecular_features = aggregate_molecular_features(molecular_train)

    # Fusion des caractéristiques
    train_merged, test_merged = merge_features(
        clinical_train,
        molecular_features,
        target_train,
        clinical_test,
        molecular_test,
    )

    # Définir l'index à partir de l'ID
    if "ID" in train_merged.columns:
        train_merged = train_merged.set_index("ID")
    if "ID" in test_merged.columns:
        test_merged = test_merged.set_index("ID")
    if "ID" in target_train.columns:
        target_train = target_train.set_index("ID")

    # Filtrer les indices communs entre les données et les cibles
    common_index = target_train.index.intersection(train_merged.index)
    print(
        f"Nombre de patients avec données cliniques et cibles: {len(common_index)}"
    )

    X = train_merged.loc[common_index]
    y = target_train.loc[common_index]

    # Vérifier et traiter les valeurs manquantes dans X et y
    print(f"Valeurs manquantes dans X: {X.isnull().sum().sum()}")
    print(f"Valeurs manquantes dans y: {y.isnull().sum().sum()}")

    # Supprimer les lignes avec des valeurs manquantes dans y
    valid_rows = ~y.isnull().any(axis=1)
    X = X.loc[valid_rows]
    y = y.loc[valid_rows]

    print(
        f"Dimensions après filtrage des valeurs manquantes - X: {X.shape}, y: {y.shape}"
    )

    # Sélectionner toutes les colonnes numériques disponibles
    X = X.select_dtypes(include=["number"])
    test_merged = test_merged.select_dtypes(include=["number"])

    # Renommage si nécessaire (NUM_MUTATIONS -> Nmut)
    if "NUM_MUTATIONS" in X.columns and "Nmut" not in X.columns:
        X["Nmut"] = X["NUM_MUTATIONS"]
        test_merged["Nmut"] = test_merged["NUM_MUTATIONS"]

    # Limiter strictement aux 4 caractéristiques principales du benchmark
    benchmark_features = ["BM_BLAST", "HB", "PLT", "Nmut"]
    available_features = [f for f in benchmark_features if f in X.columns]

    # Si une caractéristique manque, vérifier les alternatives
    if "Nmut" not in available_features and "NUM_MUTATIONS" in X.columns:
        X["Nmut"] = X["NUM_MUTATIONS"]
        test_merged["Nmut"] = test_merged["NUM_MUTATIONS"]
        available_features.append("Nmut")

    # Ne garder que les caractéristiques disponibles du benchmark
    X = X[available_features]
    test_merged = test_merged[available_features]

    print(f"Caractéristiques du benchmark retenues: {available_features}")

    # Division en ensembles d'entraînement et de validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Imputer les valeurs manquantes (comme dans le benchmark)
    imputer = SimpleImputer(strategy="median")
    X_train_values = imputer.fit_transform(X_train)
    X_val_values = imputer.transform(X_val)
    X_test_values = imputer.transform(test_merged)

    # Reconstruire les DataFrames
    X_train = pd.DataFrame(
        X_train_values, index=X_train.index, columns=X_train.columns
    )
    X_val = pd.DataFrame(
        X_val_values, index=X_val.index, columns=X_val.columns
    )
    test_merged = pd.DataFrame(
        X_test_values, index=test_merged.index, columns=test_merged.columns
    )

    # Sauvegarde des données
    X_train.index.name = "ID"
    X_val.index.name = "ID"
    y_train.index.name = "ID"
    y_val.index.name = "ID"
    test_merged.index.name = "ID"

    X_train.to_csv(DATA_PROCESSED_PATH + "X_train_prepared.csv", index=True)
    y_train.to_csv(DATA_PROCESSED_PATH + "y_train_prepared.csv", index=True)
    X_val.to_csv(DATA_PROCESSED_PATH + "X_val_prepared.csv", index=True)
    y_val.to_csv(DATA_PROCESSED_PATH + "y_val_prepared.csv", index=True)
    test_merged.to_csv(DATA_PROCESSED_PATH + "X_test_prepared.csv", index=True)

    print("Données préparées et sauvegardées dans data/processed !")
    print(
        f"Dimensions finales - X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {test_merged.shape}"
    )


if __name__ == "__main__":
    prepare_data()
