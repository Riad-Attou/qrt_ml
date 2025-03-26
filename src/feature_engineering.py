import numpy as np
import pandas as pd


def aggregate_molecular_features(molecular_train):
    """Crée de nouvelles variables basées sur les mutations moléculaires."""
    print("Agrégation des caractéristiques moléculaires...")

    # Vérifier les valeurs manquantes
    print("\nValeurs manquantes avant nettoyage :")
    print(molecular_train.isnull().sum())

    # Créer une copie pour éviter les modifications indésirables
    molecular_data = molecular_train.copy()

    # Remplacer les valeurs manquantes dans VAF et DEPTH par 0
    # Ces colonnes sont importantes pour l'analyse
    molecular_data["VAF"] = molecular_data["VAF"].fillna(0)
    molecular_data["DEPTH"] = molecular_data["DEPTH"].fillna(0)

    # Remplacer les valeurs manquantes dans GENE
    molecular_data["GENE"] = molecular_data["GENE"].fillna("UNKNOWN")

    print("\nValeurs manquantes après remplacement :")
    print(molecular_data.isnull().sum())

    # Nombre total de mutations par patient (caractéristique principale du benchmark)
    mutation_count = (
        molecular_data.groupby("ID").size().reset_index(name="NUM_MUTATIONS")
    )

    # Renommer pour être cohérent avec le benchmark
    mutation_count = mutation_count.rename(columns={"NUM_MUTATIONS": "Nmut"})

    # On ne garde que cette caractéristique clé pour suivre le benchmark
    molecular_features = mutation_count.copy()

    print(
        f"Features moléculaires créées: {molecular_features.shape[0]} patients, {molecular_features.shape[1]} caractéristiques"
    )
    return molecular_features


def merge_features(
    clinical_train,
    molecular_features,
    target_train,
    clinical_test,
    molecular_test=None,
):
    """Fusionne les données cliniques avec les nouvelles variables des mutations."""
    print("Fusion des caractéristiques cliniques et moléculaires...")

    # Si molecular_test est fourni, l'agréger de la même manière que molecular_train
    if molecular_test is not None:
        molecular_features_test = aggregate_molecular_features(molecular_test)
        # Fusionner avec les données cliniques de test
        test_merged = clinical_test.merge(
            molecular_features_test, on="ID", how="left"
        )
    else:
        # Sinon, utiliser les mêmes features moléculaires pour l'ensemble de test
        # Cela suppose que les patients de test peuvent être présents dans l'ensemble d'entraînement moléculaire
        test_merged = clinical_test.merge(
            molecular_features, on="ID", how="left"
        )

    # Fusion pour l'ensemble d'entraînement
    train_merged = clinical_train.merge(
        molecular_features, on="ID", how="left"
    )

    # Vérifier les valeurs manquantes après la fusion
    print(
        f"\nValeurs manquantes après fusion (train): {train_merged.isnull().sum().sum()}"
    )
    print(
        f"Valeurs manquantes après fusion (test): {test_merged.isnull().sum().sum()}"
    )

    # Remplacer les valeurs manquantes par 0 pour les caractéristiques moléculaires
    # Identifier les colonnes moléculaires (celles qui ne sont pas dans clinical_train)
    mol_columns = [
        col
        for col in train_merged.columns
        if col not in clinical_train.columns and col != "ID"
    ]

    for col in mol_columns:
        if col in train_merged.columns:
            train_merged[col] = train_merged[col].fillna(0)
        if col in test_merged.columns:
            test_merged[col] = test_merged[col].fillna(0)

    # Vérifier si TOP_MUTATED_GENE existe avant de le traiter
    if (
        "TOP_MUTATED_GENE" in train_merged.columns
        and "TOP_MUTATED_GENE" in test_merged.columns
    ):
        # Convertir en chaîne de caractères
        train_merged["TOP_MUTATED_GENE"] = train_merged[
            "TOP_MUTATED_GENE"
        ].astype(str)
        test_merged["TOP_MUTATED_GENE"] = test_merged[
            "TOP_MUTATED_GENE"
        ].astype(str)

        # One-hot encoding pour TOP_MUTATED_GENE
        # Prendre uniquement les gènes les plus fréquents pour éviter trop de colonnes
        top_genes = (
            train_merged["TOP_MUTATED_GENE"].value_counts().head(10).index
        )

        # Créer les colonnes en une seule fois pour éviter la fragmentation
        top_gene_cols_train = {}
        top_gene_cols_test = {}

        for gene in top_genes:
            top_gene_cols_train[f"TOP_GENE_{gene}"] = (
                train_merged["TOP_MUTATED_GENE"] == gene
            ).astype(int)
            top_gene_cols_test[f"TOP_GENE_{gene}"] = (
                test_merged["TOP_MUTATED_GENE"] == gene
            ).astype(int)

        # Créer des DataFrames et les fusionner avec les DataFrames originaux
        top_gene_df_train = pd.DataFrame(
            top_gene_cols_train, index=train_merged.index
        )
        top_gene_df_test = pd.DataFrame(
            top_gene_cols_test, index=test_merged.index
        )

        # Supprimer et ajouter les colonnes avec sécurité
        train_merged = pd.concat(
            [
                train_merged.drop(
                    columns=["TOP_MUTATED_GENE"], errors="ignore"
                ),
                top_gene_df_train,
            ],
            axis=1,
        )
        test_merged = pd.concat(
            [
                test_merged.drop(
                    columns=["TOP_MUTATED_GENE"], errors="ignore"
                ),
                top_gene_df_test,
            ],
            axis=1,
        )

    # Vérifier les dimensions finales
    print(
        f"Dimensions finales - train_merged: {train_merged.shape}, test_merged: {test_merged.shape}"
    )

    return train_merged, test_merged
