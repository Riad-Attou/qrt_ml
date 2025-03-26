import numpy as np
import pandas as pd


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
    """Nettoie les valeurs manquantes dans les données cliniques avec des approches avancées."""
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

        # Transformation logarithmique pour les variables asymétriques
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

    # Conserver uniquement les colonnes essentielles
    cols_to_keep = ["ID", "BM_BLAST", "HB", "PLT", "CYTOGENETICS"]

    # Ajouter d'autres colonnes si disponibles
    for col in ["WBC", "ANC", "MONOCYTES", "CENTER"]:
        if col in clinical_train.columns:
            cols_to_keep.append(col)

    clinical_train_clean = clinical_train[cols_to_keep].copy()
    clinical_test_clean = clinical_test[cols_to_keep].copy()

    # Vérification des valeurs manquantes après nettoyage
    print("\nValeurs manquantes après nettoyage (clinical_train):")
    print(clinical_train_clean.isnull().sum().sum())
    print("\nValeurs manquantes après nettoyage (clinical_test):")
    print(clinical_test_clean.isnull().sum().sum())

    print(
        f"Dimensions après nettoyage - clinical_train: {clinical_train_clean.shape}, clinical_test: {clinical_test_clean.shape}"
    )

    return clinical_train_clean, clinical_test_clean


def find_nearest_patient(target_patient, patients, features):
    """
    Trouve le patient le plus proche en utilisant la distance euclidienne.

    Args:
        target_patient: Patient avec des valeurs manquantes
        patients: DataFrame des patients pour la recherche
        features: Liste des caractéristiques à considérer

    Returns:
        Le patient le plus proche
    """
    # Filtrer les caractéristiques disponibles pour le patient cible
    available_features = [
        f for f in features if not pd.isna(target_patient[f])
    ]

    if not available_features:
        return None

    # Calculer les distances euclidiennes
    distances = []
    for _, patient in patients.iterrows():
        # Ignorer les patients avec des valeurs manquantes dans les caractéristiques disponibles
        if any(pd.isna(patient[f]) for f in available_features):
            distances.append(float("inf"))
            continue

        # Calculer la distance euclidienne
        squared_diff = sum(
            (target_patient[f] - patient[f]) ** 2 for f in available_features
        )
        distances.append(np.sqrt(squared_diff))

    # Trouver l'indice du patient le plus proche
    if not distances or min(distances) == float("inf"):
        return None

    closest_idx = np.argmin(distances)
    return patients.iloc[closest_idx]


def impute_missing_values(df, features):
    """
    Impute les valeurs manquantes en utilisant le patient le plus proche.

    Args:
        df: DataFrame des patients
        features: Liste des caractéristiques à considérer

    Returns:
        DataFrame avec valeurs imputées
    """
    df_imputed = df.copy()

    # Pour chaque patient avec des valeurs manquantes
    for idx, patient in df.iterrows():
        missing_features = [f for f in features if pd.isna(patient[f])]

        if not missing_features:
            continue

        # Trouver les patients sans valeurs manquantes dans les caractéristiques concernées
        complete_patients = df.dropna(subset=features)

        # Si pas de patients complets, utiliser la médiane
        if complete_patients.empty:
            for feature in missing_features:
                df_imputed.loc[idx, feature] = df[feature].median()
            continue

        # Trouver le patient le plus proche
        closest_patient = find_nearest_patient(
            patient, complete_patients, features
        )

        if closest_patient is None:
            # Utiliser la médiane si aucun patient proche n'est trouvé
            for feature in missing_features:
                df_imputed.loc[idx, feature] = df[feature].median()
        else:
            # Imputer avec les valeurs du patient le plus proche
            for feature in missing_features:
                df_imputed.loc[idx, feature] = closest_patient[feature]

    return df_imputed
