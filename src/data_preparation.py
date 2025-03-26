import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_preprocessing import clean_data, impute_missing_values, load_data
from feature_engineering import aggregate_molecular_features, merge_features

DATA_PROCESSED_PATH = "data/processed/"


def prepare_data():
    print("Préparation des données avec techniques avancées...")

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

    # Agrégation des caractéristiques moléculaires avancées
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

    # Sélectionner uniquement les colonnes numériques
    X = X.select_dtypes(include=["number"])
    test_merged = test_merged.select_dtypes(include=["number"])

    # S'assurer que les mêmes colonnes sont présentes dans les deux ensembles
    common_cols = set(X.columns).intersection(set(test_merged.columns))
    X = X[list(common_cols)]
    test_merged = test_merged[list(common_cols)]

    # Assurer que Nmut est présent (pour le benchmark)
    if "Nmut" in common_cols:
        print("Nmut est déjà présent dans les caractéristiques.")
    elif "NUM_MUTATIONS" in common_cols:
        # Renommer NUM_MUTATIONS en Nmut
        X = X.rename(columns={"NUM_MUTATIONS": "Nmut"})
        test_merged = test_merged.rename(columns={"NUM_MUTATIONS": "Nmut"})
        common_cols = list(common_cols - {"NUM_MUTATIONS"}) + ["Nmut"]
        print("NUM_MUTATIONS renommé en Nmut.")

    # Limiter aux caractéristiques du benchmark si elles sont toutes disponibles
    benchmark_features = ["BM_BLAST", "HB", "PLT", "Nmut"]
    if all(f in common_cols for f in benchmark_features):
        print(
            f"Utilisation des caractéristiques du benchmark: {benchmark_features}"
        )
        X = X[benchmark_features]
        test_merged = test_merged[benchmark_features]
    else:
        print(
            f"Caractéristiques du benchmark non disponibles. Utilisation de toutes les caractéristiques: {len(common_cols)}"
        )
        X = X[list(common_cols)]
        test_merged = test_merged[list(common_cols)]

    # Division en ensembles d'entraînement et de validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Normalisation des données
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        index=X_train.index,
        columns=X_train.columns,
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val), index=X_val.index, columns=X_val.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(test_merged),
        index=test_merged.index,
        columns=test_merged.columns,
    )

    # Sauvegarde des données préparées
    X_train_scaled.index.name = "ID"
    X_val_scaled.index.name = "ID"
    y_train.index.name = "ID"
    y_val.index.name = "ID"
    X_test_scaled.index.name = "ID"

    X_train_scaled.to_csv(
        DATA_PROCESSED_PATH + "X_train_prepared.csv", index=True
    )
    y_train.to_csv(DATA_PROCESSED_PATH + "y_train_prepared.csv", index=True)
    X_val_scaled.to_csv(DATA_PROCESSED_PATH + "X_val_prepared.csv", index=True)
    y_val.to_csv(DATA_PROCESSED_PATH + "y_val_prepared.csv", index=True)
    X_test_scaled.to_csv(
        DATA_PROCESSED_PATH + "X_test_prepared.csv", index=True
    )

    # Sauvegarder le scaler
    import joblib

    joblib.dump(scaler, DATA_PROCESSED_PATH + "scaler.pkl")

    print("Données préparées et sauvegardées dans data/processed !")


if __name__ == "__main__":
    prepare_data()
