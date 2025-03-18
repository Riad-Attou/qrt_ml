import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def load_data():
    """Charge les fichiers CSV et retourne les DataFrames."""
    clinical_train = pd.read_csv("data/raw/clinical_train.csv")
    molecular_train = pd.read_csv("data/raw/molecular_train.csv")
    target_train = pd.read_csv("data/raw/target_train.csv")
    clinical_test = pd.read_csv("data/raw/clinical_test.csv")
    molecular_test = pd.read_csv("data/raw/molecular_test.csv")

    print(
        f"Données chargées : \n"
        f"- clinical_train: {clinical_train.shape} lignes\n"
        f"- molecular_train: {molecular_train.shape} lignes\n"
        f"- target_train: {target_train.shape} lignes\n"
        f"- clinical_test: {clinical_test.shape} lignes\n"
        f"- molecular_test: {molecular_test.shape} lignes"
    )

    return (
        clinical_train,
        molecular_train,
        target_train,
        clinical_test,
        molecular_test,
    )


def clean_data(clinical_train, clinical_test):
    """Nettoie les valeurs manquantes et encode les variables catégorielles."""

    print("Nettoyage des données cliniques...")

    # Vérification des valeurs manquantes avant nettoyage
    print("\nValeurs manquantes avant nettoyage (clinical_train):")
    print(clinical_train.isnull().sum())
    print("\nValeurs manquantes avant nettoyage (clinical_test):")
    print(clinical_test.isnull().sum())

    # Remplissage des valeurs manquantes pour les variables numériques
    num_cols = ["BM_BLAST", "WBC", "ANC", "MONOCYTES", "HB", "PLT"]
    for col in num_cols:
        # Utiliser la médiane de l'ensemble d'entraînement pour les deux datasets
        median_value = clinical_train[col].median()
        print(f"Médiane pour {col}: {median_value}")
        clinical_train[col] = clinical_train[col].fillna(median_value)
        clinical_test[col] = clinical_test[col].fillna(median_value)

    # Gestion des valeurs extrêmes (outliers)
    for col in num_cols:
        # Calculer les percentiles sur l'ensemble d'entraînement
        q1 = clinical_train[col].quantile(0.01)
        q3 = clinical_train[col].quantile(0.99)

        # Seuils pour les valeurs extrêmes
        lower_bound = max(
            0, q1
        )  # Ne pas aller en dessous de zéro pour les valeurs cliniques
        upper_bound = q3 * 1.5

        # Appliquer les limites aux deux ensembles
        clinical_train[col] = clinical_train[col].clip(
            lower_bound, upper_bound
        )
        clinical_test[col] = clinical_test[col].clip(lower_bound, upper_bound)

        # Log transformation for skewed variables
        if col in ["WBC", "PLT", "ANC"]:
            clinical_train[col] = np.log1p(clinical_train[col])
            clinical_test[col] = np.log1p(clinical_test[col])
            print(f"Transformation logarithmique appliquée sur {col}")

    # Remplacement des valeurs manquantes pour la variable catégorielle CYTOGENETICS
    most_common = clinical_train["CYTOGENETICS"].value_counts().index[0]
    print(f"Valeur la plus fréquente pour CYTOGENETICS: {most_common}")

    clinical_train["CYTOGENETICS"] = clinical_train["CYTOGENETICS"].fillna(
        most_common
    )
    clinical_test["CYTOGENETICS"] = clinical_test["CYTOGENETICS"].fillna(
        most_common
    )

    # Encodage One-Hot de CENTER avec gestion des nouveaux centres dans les données de test
    # Liste de tous les centres possibles (train + test)
    all_centers = set(clinical_train["CENTER"].unique()).union(
        set(clinical_test["CENTER"].unique())
    )
    print(f"Nombre total de centres distincts: {len(all_centers)}")

    # Encodage manuel pour éviter les problèmes avec les nouveaux centres
    # Préparer des dictionnaires pour créer les colonnes en une seule fois (pour éviter la fragmentation)
    center_cols_train = {}
    center_cols_test = {}

    for center in all_centers:
        center_col = f"CENTER_{center}"
        center_cols_train[center_col] = (
            clinical_train["CENTER"] == center
        ).astype(int)
        center_cols_test[center_col] = (
            clinical_test["CENTER"] == center
        ).astype(int)

    # Créer des DataFrames et les fusionner avec les DataFrames originaux
    center_df_train = pd.DataFrame(
        center_cols_train, index=clinical_train.index
    )
    center_df_test = pd.DataFrame(center_cols_test, index=clinical_test.index)

    # Encodage One-Hot de CYTOGENETICS avec gestion des nouvelles catégories
    all_cytogenetics = set(clinical_train["CYTOGENETICS"].unique()).union(
        set(clinical_test["CYTOGENETICS"].unique())
    )
    print(f"Nombre total de catégories CYTOGENETICS: {len(all_cytogenetics)}")

    # Préparer des dictionnaires pour créer les colonnes en une seule fois (pour éviter la fragmentation)
    cyto_cols_train = {}
    cyto_cols_test = {}

    for cyto in all_cytogenetics:
        cyto_col = f"CYTO_{cyto}"
        cyto_cols_train[cyto_col] = (
            clinical_train["CYTOGENETICS"] == cyto
        ).astype(int)
        cyto_cols_test[cyto_col] = (
            clinical_test["CYTOGENETICS"] == cyto
        ).astype(int)

    # Créer des DataFrames et les fusionner avec les DataFrames originaux
    cyto_df_train = pd.DataFrame(cyto_cols_train, index=clinical_train.index)
    cyto_df_test = pd.DataFrame(cyto_cols_test, index=clinical_test.index)

    clinical_train = pd.concat(
        [clinical_train.drop(columns=["CYTOGENETICS"]), cyto_df_train], axis=1
    )
    clinical_test = pd.concat(
        [clinical_test.drop(columns=["CYTOGENETICS"]), cyto_df_test], axis=1
    )

    # Suppression des colonnes originales et intégration des nouvelles colonnes
    clinical_train = pd.concat(
        [clinical_train.drop(columns=["CENTER"]), center_df_train], axis=1
    )
    clinical_test = pd.concat(
        [clinical_test.drop(columns=["CENTER"]), center_df_test], axis=1
    )

    # Vérification des valeurs manquantes après nettoyage
    print("\nValeurs manquantes après nettoyage (clinical_train):")
    print(clinical_train.isnull().sum().sum())
    print("\nValeurs manquantes après nettoyage (clinical_test):")
    print(clinical_test.isnull().sum().sum())

    print(
        f"Dimensions après nettoyage - clinical_train: {clinical_train.shape}, clinical_test: {clinical_test.shape}"
    )

    return clinical_train, clinical_test
