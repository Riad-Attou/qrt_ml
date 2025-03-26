import re

import numpy as np
import pandas as pd


def aggregate_molecular_features(molecular_train):
    """Crée de nouvelles variables basées sur les mutations moléculaires."""
    print("Agrégation des caractéristiques moléculaires avancées...")

    # Vérifier les valeurs manquantes
    print("\nValeurs manquantes avant nettoyage :")
    print(molecular_train.isnull().sum())

    # Créer une copie pour éviter les modifications indésirables
    molecular_data = molecular_train.copy()

    # Remplacer les valeurs manquantes dans VAF et DEPTH par 0
    molecular_data["VAF"] = molecular_data["VAF"].fillna(0)
    molecular_data["DEPTH"] = molecular_data["DEPTH"].fillna(0)
    molecular_data["GENE"] = molecular_data["GENE"].fillna("UNKNOWN")

    print("\nValeurs manquantes après remplacement :")
    print(molecular_data.isnull().sum())

    # 1. Nombre total de mutations par patient (caractéristique de base)
    mutation_count = (
        molecular_data.groupby("ID").size().reset_index(name="Nmut")
    )

    # 2. Nombre de gènes affectés par patient
    unique_genes_per_patient = (
        molecular_data.groupby("ID")["GENE"]
        .nunique()
        .reset_index(name="NUM_GENES_AFFECTED")
    )

    # 3. Moyenne des VAF par patient (indicateur de la charge mutationnelle)
    avg_vaf = (
        molecular_data.groupby("ID")["VAF"].mean().reset_index(name="AVG_VAF")
    )

    # 4. Nombre de mutations par effet (fonctionnel, silencieux, etc.)
    # Identifier les mutations fonctionnelles importantes
    effect_mapping = {
        "frameshift": "functional",
        "missense": "functional",
        "nonsense": "functional",
        "stop_gained": "functional",
        "splice_site": "functional",
        "non_synonymous": "functional",
        "splice": "functional",
        "synonym": "silent",
        "silent": "silent",
    }

    # Créer une colonne pour l'effet fonctionnel basée sur les mots-clés
    molecular_data["EFFECT_TYPE"] = "other"
    for effect, effect_type in effect_mapping.items():
        mask = molecular_data["EFFECT"].str.contains(
            effect, case=False, na=False
        )
        molecular_data.loc[mask, "EFFECT_TYPE"] = effect_type

    # Compter les mutations fonctionnelles et silencieuses par patient
    functional_mutations = (
        molecular_data[molecular_data["EFFECT_TYPE"] == "functional"]
        .groupby("ID")
        .size()
        .reset_index(name="FUNCTIONAL_MUTATIONS")
    )

    silent_mutations = (
        molecular_data[molecular_data["EFFECT_TYPE"] == "silent"]
        .groupby("ID")
        .size()
        .reset_index(name="SILENT_MUTATIONS")
    )

    # 5. Identifier les mutations dans des gènes spécifiques importants pour la leucémie
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
        "SF3B1",
        "SRSF2",
        "U2AF1",
        "STAG2",
        "BCOR",
        "KIT",
        "CEBPA",
        "JAK2",
        "MPL",
    ]

    gene_features = pd.DataFrame({"ID": mutation_count["ID"]})

    for gene in important_genes:
        gene_data = (
            molecular_data[molecular_data["GENE"] == gene]
            .groupby("ID")
            .size()
            .reset_index(name=f"{gene}_MUTATION")
        )
        gene_features = gene_features.merge(gene_data, on="ID", how="left")

    # 6. Classification du risque génétique
    favorable_genes = ["NPM1", "CEBPA"]
    unfavorable_genes = [
        "TP53",
        "ASXL1",
        "RUNX1",
        "FLT3",
        "DNMT3A",
        "IDH1",
        "IDH2",
    ]

    def gene_risk(row):
        for gene in unfavorable_genes:
            if row.get(f"{gene}_MUTATION", 0) > 0:
                return "unfavorable"
        for gene in favorable_genes:
            if row.get(f"{gene}_MUTATION", 0) > 0:
                return "favorable"
        return "intermediate"

    # Fusionner les caractéristiques et remplir les valeurs manquantes
    molecular_features = mutation_count.merge(
        unique_genes_per_patient, on="ID", how="left"
    )
    molecular_features = molecular_features.merge(avg_vaf, on="ID", how="left")
    molecular_features = molecular_features.merge(
        functional_mutations, on="ID", how="left"
    )
    molecular_features = molecular_features.merge(
        silent_mutations, on="ID", how="left"
    )
    molecular_features = molecular_features.merge(
        gene_features, on="ID", how="left"
    )

    # Remplir les valeurs manquantes avec 0
    molecular_features = molecular_features.fillna(0)

    # Calculer le risque génétique
    molecular_features["GENE_RISK"] = molecular_features.apply(
        gene_risk, axis=1
    )
    molecular_features["GENE_RISK_UNFAVORABLE"] = (
        molecular_features["GENE_RISK"] == "unfavorable"
    ).astype(int)
    molecular_features["GENE_RISK_FAVORABLE"] = (
        molecular_features["GENE_RISK"] == "favorable"
    ).astype(int)

    print(
        f"Features moléculaires créées: {molecular_features.shape[0]} patients, {molecular_features.shape[1]} caractéristiques"
    )

    return molecular_features


def classify_cytogenetic_risk(cyto_str):
    """Classifie le risque cytogénétique en se basant sur des motifs connus."""
    if not isinstance(cyto_str, str):
        return "unknown"
    s = cyto_str.lower()

    # Cas favorables
    if any(pattern in s for pattern in ["t(15;17)", "t(8;21)", "inv(16)"]):
        return "favorable"

    # Cas défavorables
    if any(
        pattern in s
        for pattern in [
            "del(5)",
            "del(7)",
            "complex",
            "t(3;3)",
            "inv(3)",
            "t(11;19)",
            "del(5q)",
            "-7",
            "-17",
        ]
    ):
        return "unfavorable"

    # Caryotype complexe (indicateur de mauvais pronostic)
    if "complex" in s:
        return "unfavorable"

    # Monosomies (perte d'un chromosome entier)
    monosomies = re.findall(r"-(\d+)", s)
    if (
        len(monosomies) > 2
    ):  # Caryotype monosomal (indicateur de très mauvais pronostic)
        return "unfavorable"

    # Sinon, c'est plutôt intermédiaire (ex : caryotype normal)
    return "intermediate"


def extract_cytogenetic_features(cytogenetic_data):
    """Extrait des caractéristiques à partir des données cytogénétiques."""
    features = []

    for cyto in cytogenetic_data:
        if not isinstance(cyto, str):
            features.append(
                {
                    "nb_del": 0,
                    "nb_t": 0,  # nombre de translocations
                    "nb_inv": 0,  # nombre d'inversions
                    "complex_karyotype": 0,
                    "nb_mono": 0,  # nombre de monosomies
                    "cyto_risk": "unknown",
                }
            )
            continue

        s = cyto.lower()

        # Compter les caractéristiques de base
        nb_del = s.count("del")
        nb_t = s.count("t(")
        nb_inv = s.count("inv")
        complex_karyotype = 1 if "complex" in s else 0

        # Compter les monosomies
        monosomies = re.findall(r"-(\d+)", s)
        nb_mono = len(monosomies)

        # Classifier le risque
        cyto_risk = classify_cytogenetic_risk(s)

        features.append(
            {
                "nb_del": nb_del,
                "nb_t": nb_t,
                "nb_inv": nb_inv,
                "complex_karyotype": complex_karyotype,
                "nb_mono": nb_mono,
                "cyto_risk": cyto_risk,
            }
        )

    return pd.DataFrame(features)


def merge_features(
    clinical_train,
    molecular_features,
    target_train=None,
    clinical_test=None,
    molecular_test=None,
):
    """Fusionne les données cliniques avec les nouvelles variables des mutations."""
    print("Fusion des caractéristiques cliniques et moléculaires...")

    # Préparation des données cliniques d'entraînement
    clinical_data_train = clinical_train.copy()

    # Extraction des caractéristiques cytogénétiques
    if "CYTOGENETICS" in clinical_data_train.columns:
        cyto_features_train = extract_cytogenetic_features(
            clinical_data_train["CYTOGENETICS"]
        )
        clinical_data_train = pd.concat(
            [clinical_data_train, cyto_features_train], axis=1
        )

        # Création de dummy variables pour le risque cytogénétique
        cyto_risk_dummies = pd.get_dummies(
            clinical_data_train["cyto_risk"], prefix="CYTO_RISK"
        )
        clinical_data_train = pd.concat(
            [clinical_data_train, cyto_risk_dummies], axis=1
        )

    # Garder uniquement les colonnes intéressantes
    cols_to_keep = ["ID", "BM_BLAST", "HB", "PLT"]
    if "WBC" in clinical_data_train.columns:
        cols_to_keep.append("WBC")
    if "ANC" in clinical_data_train.columns:
        cols_to_keep.append("ANC")
    if "MONOCYTES" in clinical_data_train.columns:
        cols_to_keep.append("MONOCYTES")

    # Ajouter les nouvelles caractéristiques si elles existent
    for col in [
        "nb_del",
        "nb_t",
        "nb_inv",
        "complex_karyotype",
        "nb_mono",
        "CYTO_RISK_favorable",
        "CYTO_RISK_intermediate",
        "CYTO_RISK_unfavorable",
    ]:
        if col in clinical_data_train.columns:
            cols_to_keep.append(col)

    # Sélectionner les colonnes à conserver
    clinical_train_selected = clinical_data_train[cols_to_keep]

    # Fusion avec les caractéristiques moléculaires
    train_merged = clinical_train_selected.merge(
        molecular_features, on="ID", how="left"
    )

    # Traitement similaire pour les données de test si fournies
    if clinical_test is not None:
        clinical_data_test = clinical_test.copy()

        if "CYTOGENETICS" in clinical_data_test.columns:
            cyto_features_test = extract_cytogenetic_features(
                clinical_data_test["CYTOGENETICS"]
            )
            clinical_data_test = pd.concat(
                [clinical_data_test, cyto_features_test], axis=1
            )

            # Création de dummy variables pour le risque cytogénétique
            cyto_risk_dummies = pd.get_dummies(
                clinical_data_test["cyto_risk"], prefix="CYTO_RISK"
            )
            clinical_data_test = pd.concat(
                [clinical_data_test, cyto_risk_dummies], axis=1
            )

        # Sélectionner les mêmes colonnes que pour l'entraînement
        test_cols = [
            col for col in cols_to_keep if col in clinical_data_test.columns
        ]
        clinical_test_selected = clinical_data_test[test_cols]

        # Si molecular_test est fourni, l'agréger et fusionner
        if molecular_test is not None:
            molecular_features_test = aggregate_molecular_features(
                molecular_test
            )
            test_merged = clinical_test_selected.merge(
                molecular_features_test, on="ID", how="left"
            )
        else:
            # Sinon, fusionner avec les caractéristiques moléculaires d'entraînement
            test_merged = clinical_test_selected.merge(
                molecular_features, on="ID", how="left"
            )
    else:
        test_merged = None

    # Remplir les valeurs manquantes
    if train_merged is not None:
        train_merged = train_merged.fillna(0)
    if test_merged is not None:
        test_merged = test_merged.fillna(0)

    print("Dimensions après fusion:")
    print(f"- train_merged: {train_merged.shape}")
    if test_merged is not None:
        print(f"- test_merged: {test_merged.shape}")

    return train_merged, test_merged
