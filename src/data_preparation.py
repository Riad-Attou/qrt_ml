import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data_preprocessing import clean_data, load_data
from src.feature_engineering import (
    aggregate_molecular_features,
    merge_features,
)

DATA_PROCESSED_PATH = "data/processed/"


def prepare_data():
    print("Préparation des données...")

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

    # Agrégation des caractéristiques moléculaires
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

    # Normalisation des données
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X), index=X.index, columns=X.columns
    )

    # Sélection des caractéristiques les plus importantes
    # Utiliser f_regression car nous travaillons avec des données de survie
    selector = SelectKBest(f_regression, k=min(30, X.shape[1]))

    # Créer une version numérique de y pour la sélection de caractéristiques
    y_numeric = y["OS_YEARS"]

    # Ajuster le sélecteur
    selector.fit(X_scaled, y_numeric)

    # Obtenir les colonnes sélectionnées
    cols_selected = X.columns[selector.get_support()]

    # Filtrer les caractéristiques
    X_selected = X_scaled[cols_selected]

    print(f"Nombre de caractéristiques sélectionnées: {len(cols_selected)}")
    print("Top 10 caractéristiques:")
    scores = selector.scores_
    for i in range(min(10, len(cols_selected))):
        feature_idx = np.argsort(scores[selector.get_support()])[-i - 1]
        feature_name = cols_selected[feature_idx]
        print(
            f"- {feature_name}: {scores[selector.get_support()][feature_idx]:.4f}"
        )

    # Appliquer la même transformation au jeu de test
    test_scaled = pd.DataFrame(
        scaler.transform(test_merged.select_dtypes(include=["number"])),
        index=test_merged.index,
        columns=test_merged.select_dtypes(include=["number"]).columns,
    )

    # Sélectionner les mêmes colonnes que dans l'ensemble d'entraînement
    # Vérifier si toutes les colonnes existent dans le jeu de test
    common_cols = [col for col in cols_selected if col in test_scaled.columns]
    test_selected = test_scaled[common_cols]

    if len(common_cols) < len(cols_selected):
        print(
            f"Attention: {len(cols_selected) - len(common_cols)} caractéristiques manquantes dans le jeu de test"
        )

    # Division en ensembles d'entraînement et de validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_selected, y, test_size=0.2, random_state=42
    )

    # Sauvegarde des données préparées
    X_train = pd.DataFrame(
        X_train, index=X_train.index, columns=X_selected.columns
    )
    X_val = pd.DataFrame(X_val, index=X_val.index, columns=X_selected.columns)
    test_selected = pd.DataFrame(
        test_selected, index=test_selected.index, columns=common_cols
    )

    # S'assurer que les indices sont bien présents
    X_train.index.name = "ID"
    X_val.index.name = "ID"
    y_train.index.name = "ID"
    y_val.index.name = "ID"
    test_selected.index.name = "ID"

    # Sauvegarde des données
    X_train.to_csv(DATA_PROCESSED_PATH + "X_train_prepared.csv", index=True)
    y_train.to_csv(DATA_PROCESSED_PATH + "y_train_prepared.csv", index=True)
    X_val.to_csv(DATA_PROCESSED_PATH + "X_val_prepared.csv", index=True)
    y_val.to_csv(DATA_PROCESSED_PATH + "y_val_prepared.csv", index=True)
    test_selected.to_csv(
        DATA_PROCESSED_PATH + "X_test_prepared.csv", index=True
    )

    print("Données préparées !")
    print(
        f"Dimensions finales - X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {test_selected.shape}"
    )


if __name__ == "__main__":
    prepare_data()
