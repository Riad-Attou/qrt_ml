def aggregate_molecular_features(molecular_train, molecular_test):
    """Crée de nouvelles variables basées sur les mutations moléculaires."""

    mutation_count = (
        molecular_train.groupby("ID").size().reset_index(name="NUM_MUTATIONS")
    )
    unique_genes_per_patient = (
        molecular_train.groupby("ID")["GENE"]
        .nunique()
        .reset_index(name="NUM_GENES_AFFECTED")
    )

    top_mutation = molecular_train.loc[
        molecular_train.groupby("ID")["VAF"].idxmax(), ["ID", "GENE", "VAF"]
    ]
    top_mutation.rename(
        columns={"GENE": "TOP_MUTATED_GENE", "VAF": "TOP_MUTATION_VAF"},
        inplace=True,
    )

    molecular_features = mutation_count.merge(
        unique_genes_per_patient, on="ID", how="left"
    )
    molecular_features = molecular_features.merge(
        top_mutation, on="ID", how="left"
    )

    molecular_features.fillna(
        {
            "NUM_MUTATIONS": 0,
            "NUM_GENES_AFFECTED": 0,
            "TOP_MUTATION_VAF": 0,
            "TOP_MUTATED_GENE": "NONE",
        },
        inplace=True,
    )

    return molecular_features


def merge_features(
    clinical_train,
    molecular_features,
    target_train,
    clinical_test,
    molecular_test,
):
    """Fusionne les données cliniques avec les nouvelles variables des mutations."""

    train_merged = clinical_train.merge(
        molecular_features, on="ID", how="left"
    )
    test_merged = clinical_test.merge(molecular_features, on="ID", how="left")

    train_merged.fillna(0, inplace=True)
    test_merged.fillna(0, inplace=True)

    return train_merged, test_merged
