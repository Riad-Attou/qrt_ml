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

    # 1. Nombre total de mutations par patient
    mutation_count = (
        molecular_data.groupby("ID").size().reset_index(name="NUM_MUTATIONS")
    )

    # 2. Nombre de gènes affectés par patient
    unique_genes_per_patient = (
        molecular_data.groupby("ID")["GENE"]
        .nunique()
        .reset_index(name="NUM_GENES_AFFECTED")
    )

    # 3. Mutation ayant le plus grand VAF par patient
    # S'assurer qu'il y a des données valides avant de chercher un maximum
    molecular_data_with_vaf = molecular_data[molecular_data["VAF"] > 0]

    if not molecular_data_with_vaf.empty:
        top_mutation = molecular_data_with_vaf.loc[
            molecular_data_with_vaf.groupby("ID")["VAF"].idxmax(),
            ["ID", "GENE", "VAF"],
        ]
        top_mutation.rename(
            columns={"GENE": "TOP_MUTATED_GENE", "VAF": "TOP_MUTATION_VAF"},
            inplace=True,
        )
    else:
        # Créer un DataFrame vide avec les bonnes colonnes si pas de données VAF
        top_mutation = pd.DataFrame(
            columns=["ID", "TOP_MUTATED_GENE", "TOP_MUTATION_VAF"]
        )

    # 4. Moyenne des VAF par patient (indicateur de la charge mutationnelle)
    avg_vaf = (
        molecular_data.groupby("ID")["VAF"].mean().reset_index(name="AVG_VAF")
    )

    # 5. Écart-type des VAF par patient (hétérogénéité clonale)
    std_vaf = (
        molecular_data.groupby("ID")["VAF"].std().reset_index(name="STD_VAF")
    )
    # Remplacer NaN par 0 dans std_vaf (si un patient n'a qu'une mutation)
    std_vaf["STD_VAF"] = std_vaf["STD_VAF"].fillna(0)

    # 6. Nombre de chromosomes différents affectés
    chromosomes_affected = (
        molecular_data.groupby("ID")["CHR"]
        .nunique()
        .reset_index(name="NUM_CHROMOSOMES_AFFECTED")
    )

    # 7. Nombre de mutations par effet (fonctionnel, silencieux, etc.)
    # Identifier les mutations fonctionnelles importantes
    molecular_data["IS_FUNCTIONAL"] = (
        molecular_data["EFFECT"]
        .isin(["missense", "frameshift", "nonsense", "splice_site"])
        .astype(int)
    )

    functional_mutations = (
        molecular_data.groupby("ID")["IS_FUNCTIONAL"]
        .sum()
        .reset_index(name="NUM_FUNCTIONAL_MUTATIONS")
    )

    # 8. Gènes récurrents - vérifier la présence de mutations dans des gènes spécifiques importants pour la leucémie
    important_genes = [
        "TP53",
        "FLT3",
        "NPM1",
        "DNMT3A",
        "IDH1",
        "IDH2",
        "RUNX1",
        "ASXL1",
        "TET2",
    ]
    gene_features = pd.DataFrame({"ID": mutation_count["ID"]})

    for gene in important_genes:
        # Pour chaque ID, vérifier si le gène est muté
        gene_data = (
            molecular_data[molecular_data["GENE"] == gene]
            .groupby("ID")
            .size()
            .reset_index(name=f"HAS_{gene}")
        )
        gene_data[f"HAS_{gene}"] = 1  # Marquer comme présent
        gene_features = gene_features.merge(gene_data, on="ID", how="left")
        gene_features[f"HAS_{gene}"] = (
            gene_features[f"HAS_{gene}"].fillna(0).astype(int)
        )

    # 9. Calculer le ratio de mutations fonctionnelles
    functional_ratio = pd.merge(
        functional_mutations, mutation_count, on="ID", how="right"
    )
    functional_ratio["FUNCTIONAL_RATIO"] = (
        functional_ratio["NUM_FUNCTIONAL_MUTATIONS"]
        / functional_ratio["NUM_MUTATIONS"]
    )
    functional_ratio["FUNCTIONAL_RATIO"] = functional_ratio[
        "FUNCTIONAL_RATIO"
    ].fillna(0)
    functional_ratio = functional_ratio[["ID", "FUNCTIONAL_RATIO"]]

    # Combiner toutes les caractéristiques - commencer par mutation_count
    molecular_features = mutation_count.copy()

    # Liste des DataFrames à fusionner
    feature_dfs = [
        unique_genes_per_patient,
        top_mutation,
        avg_vaf,
        std_vaf,
        chromosomes_affected,
        functional_mutations,
        gene_features,
        functional_ratio,
    ]

    # Effectuer les fusions de manière sécurisée
    for df in feature_dfs:
        if not df.empty and "ID" in df.columns:
            molecular_features = molecular_features.merge(
                df, on="ID", how="left"
            )

    # S'assurer qu'il n'y a pas de valeurs manquantes
    molecular_features = molecular_features.fillna(
        {
            "NUM_MUTATIONS": 0,
            "NUM_GENES_AFFECTED": 0,
            "TOP_MUTATION_VAF": 0,
            "TOP_MUTATED_GENE": "NONE",
            "AVG_VAF": 0,
            "STD_VAF": 0,
            "NUM_CHROMOSOMES_AFFECTED": 0,
            "NUM_FUNCTIONAL_MUTATIONS": 0,
            "FUNCTIONAL_RATIO": 0,
        }
    )

    # Pour les colonnes des gènes importants, remplacer NaN par 0
    for gene in important_genes:
        col_name = f"HAS_{gene}"
        if col_name in molecular_features.columns:
            molecular_features[col_name] = (
                molecular_features[col_name].fillna(0).astype(int)
            )

    print(
        f"Features moléculaires créées: {molecular_features.shape[0]} patients, {molecular_features.shape[1]} caractéristiques"
    )
    print(f"Top features moléculaires par importance:")

    # Affichage des caractéristiques de manière sécurisée
    for col in molecular_features.columns:
        if col == "ID":
            continue

        # Traitement spécifique par type de colonne
        if col == "TOP_MUTATED_GENE":
            if "TOP_MUTATED_GENE" in molecular_features.columns:
                non_empty = (molecular_features[col] != "NONE").sum()
                print(
                    f"- {col}: {non_empty} patients ({non_empty/molecular_features.shape[0]*100:.1f}%)"
                )
        elif (
            col.startswith("HAS_")
            or "NUM_" in col
            or col.endswith("_VAF")
            or col.endswith("_RATIO")
        ):
            if col in molecular_features.columns and molecular_features[
                col
            ].dtype in [np.float64, np.int64, np.float32, np.int32, np.bool_]:
                non_zero = (molecular_features[col] > 0).sum()
                print(
                    f"- {col}: {non_zero} patients ({non_zero/molecular_features.shape[0]*100:.1f}%)"
                )
        # Pour les autres types de colonnes, pas d'affichage

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
