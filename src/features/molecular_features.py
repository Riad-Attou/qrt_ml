from typing import Any, Dict, List, Set

import numpy as np
import pandas as pd

from database import Database


def quantitatif_en_qualitatif1(mat: np.ndarray, nbclasses: int) -> np.ndarray:
    """
    Convertit les données quantitatives en classes avec intervalles égaux.
    """
    mmin = np.min(mat)
    mmax = np.max(mat)
    # Si tous les éléments sont identiques, on retourne un vecteur de 0
    if mmin == mmax:
        return np.zeros(len(mat), dtype=int)
    bins = np.linspace(mmin, mmax, nbclasses + 1)
    # np.digitize renvoie des indices de 1 à nbclasses+1, on soustrait 1 pour obtenir 0 à nbclasses-1
    matqual = np.digitize(mat, bins, right=True) - 1
    return matqual


def classes_mutations(
    nbclasses: int,
    target_df: pd.DataFrame,
    df_train: pd.DataFrame,
    mol_df: pd.DataFrame,
) -> List[Set[Any]]:
    """
    Renvoie la liste des ensembles de mutations pour chaque classe.

    Paramètres :
      - nbclasses : nombre de classes à définir.
      - target_df : DataFrame contenant la variable cible ('OS_YEARS').
      - df_train : DataFrame clinique d'entraînement.
      - mol_df : DataFrame moléculaire d'entraînement.
    """
    # Utilisation de la deuxième colonne pour le temps de survie
    target_df_classe = target_df.iloc[:, 1].copy()
    target_df_classe.fillna(target_df_classe.mean(), inplace=True)
    target_array = target_df_classe.to_numpy()
    classes_temps_survie = quantitatif_en_qualitatif1(target_array, nbclasses)
    df_train = df_train.copy()
    df_train["classe_years"] = classes_temps_survie

    mol_df_classes = mol_df[["ID", "GENE"]].copy()
    clinical_df_classes = df_train[["ID", "classe_years"]].copy()
    merged_df_classes = mol_df_classes.merge(clinical_df_classes, on="ID")
    mutations_by_class: Dict[int, np.ndarray] = (
        merged_df_classes.groupby("classe_years")["GENE"].unique().to_dict()
    )
    # On s'assure que chaque classe (de 0 à nbclasses-1) est présente dans la liste
    mutations_par_classe = [
        set(mutations_by_class.get(i, [])) for i in range(nbclasses)
    ]

    mutations: List[Set[Any]] = []
    for i in range(nbclasses):
        mutations_diff = mutations_par_classe[i].copy()
        for j in range(i + 1, nbclasses):
            mutations_diff -= mutations_par_classe[j]
        mutations.append(mutations_diff)
    return mutations


def classify_gene_risk(gene_str: Any, mutations: List[Set[Any]]) -> float:
    """
    Classifie le risque génétique en fonction de la présence de la mutation dans les classes.
    Retourne une valeur numérique (l'indice de la classe) ou np.nan si la donnée n'est pas exploitable.
    """
    if not isinstance(gene_str, str):
        return (
            np.nan
        )  # Utilisation de np.nan pour indiquer une valeur manquante
    for i in range(len(mutations) - 1, -1, -1):
        if gene_str in mutations[i]:
            return float(i)
    return 0  # Retourne np.nan si aucune mutation ne correspond


def classify_gene_risk_bis(gene_str: Any) -> str:
    """
    Classifie le risque génique selon des critères prédéfinis.
    """
    if not isinstance(gene_str, str):
        return "inconnu"
    # Cas favorables
    if "NPM1" in gene_str or "CEBPA" in gene_str:
        return "favorable"
    # Cas défavorables
    if any(
        x in gene_str
        for x in [
            "ASXL1",
            "BCOR",
            "EZH2",
            # "SF3B1",
            "RUNX1",
            "SRSF2",
            "STAG2",
            "U2AF1",
            "ZRSR2",
            "ABL1",
            "CREBBP",
            "GATA2",
            "DNMT3A",
            "NFE2",
            "DDX41",
            "CSNK1A1",
            "SH2B3",
            "TP53",
        ]
    ):
        return "defavorable"
    # Sinon
    return "intermediaire"


def calc_weighted_gene_risk(df: pd.DataFrame, mutations):
    """
    Calcule la somme pondérée de gene_risk pour chaque ID dans le DataFrame.
    """
    # Calcul de beta pour chaque classe
    p = 5
    sum_k_i = sum(k**p for k in range(len(mutations)))  # Total des puissances
    beta = 5 * np.array([j**p / sum_k_i for j in range(len(mutations))])

    # Calcul de la pondération pour chaque ID
    weighted_gene_risk = (
        df.groupby("ID")["gene_risk"]
        .apply(lambda risks: sum(risk * beta[int(risk)] for risk in risks))
        .rename("gene_risk_pondere")
    )
    return pd.concat(
        [
            weighted_gene_risk,
        ],
        axis=1,
    )


def create_features_mol_df(
    df: pd.DataFrame, db: Database, mutations
) -> pd.DataFrame:
    """
    Ajoute des colonnes de classification du risque génétique et des variables dummies.
    De plus, ajoute une colonne 'genetic_score' qui correspond au score pondéré
    calculé précédemment (disponible dans self.patient_test_combinations).

    Parameters:
      - df : DataFrame contenant les données moléculaires (avec au moins les colonnes "ID" et "GENE")
      - mutations : liste ou ensemble de mutations (selon votre usage)

    Returns:
      df modifié avec les colonnes 'gene_risk', 'gene_risk2', les variables dummies associées
      et 'genetic_score'.
    """
    df = df.copy()

    # Ajout des colonnes de classification de risque basées sur le gène
    df["gene_risk"] = df["GENE"].apply(
        lambda gene: classify_gene_risk(gene, mutations)
    )
    df["gene_risk2"] = df["GENE"].apply(classify_gene_risk_bis)
    gene_risk_dummies = pd.get_dummies(df["gene_risk2"], prefix="gene_risk2")
    df = pd.concat([df, gene_risk_dummies], axis=1)

    return df


def gene_risk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrège les indicateurs de risque génétique par patient.
    """
    gene_risk_favorable = (
        df.groupby("ID")["gene_risk2_favorable"]
        .max()
        .rename("gene_risk_favorable")
    )
    gene_risk_defavorable = (
        df.groupby("ID")["gene_risk2_defavorable"]
        .max()
        .rename("gene_risk_defavorable")
    )
    gene_risk_intermediaire = (
        df.groupby("ID")["gene_risk2_intermediaire"]
        .max()
        .rename("gene_risk_intermediaire")
    )
    gene_risk_val = df.groupby("ID")["gene_risk"].max()
    return pd.concat(
        [
            gene_risk_defavorable,
            gene_risk_favorable,
            gene_risk_intermediaire,
            gene_risk_val,
        ],
        axis=1,
    )


def extract_features_mol(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrait quelques statistiques des données moléculaires par patient.
    """
    total_mutations = df.groupby("ID").size().rename("total_mutations")
    max_vaf = df.groupby("ID")["VAF"].max().rename("max_VAF")
    min_depth = df.groupby("ID")["DEPTH"].min().rename("min_depth")
    return pd.concat([total_mutations, max_vaf, min_depth], axis=1)


def extract_features_effect(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrait des features en fonction de l'effet des mutations.
    """
    weighted_stop_gained = (
        df[df["EFFECT"] == "stop_gained"]
        .groupby("ID")["VAF"]
        .sum()
        .rename("weighted_stop_gained")
    )
    weighted_frameshift = (
        df[df["EFFECT"].str.contains("frameshift", na=False)]
        .groupby("ID")["VAF"]
        .sum()
        .rename("weighted_frameshift")
    )
    weighted_non_synonymous = (
        df[df["EFFECT"].str.contains("non_synonymous", na=False)]
        .groupby("ID")["VAF"]
        .sum()
        .rename("weighted_non_synonymous")
    )
    stop_gained = (
        df[df["EFFECT"] == "stop_gained"]
        .groupby("ID")
        .size()
        .rename("stop_gained_count")
    )
    frameshift_variant = (
        df[df["EFFECT"].str.contains("frameshift", na=False)]
        .groupby("ID")
        .size()
        .rename("frameshift_variant_count")
    )
    non_synonymous = (
        df[df["EFFECT"].str.contains("non_synonymous", na=False)]
        .groupby("ID")
        .size()
        .rename("non_synonymous_count")
    )
    return pd.concat(
        [
            weighted_frameshift,
            weighted_non_synonymous,
            weighted_stop_gained,
            stop_gained,
            frameshift_variant,
            non_synonymous,
        ],
        axis=1,
    )


def extract_features_gene(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrait des indicateurs pour une liste de gènes clés.
    """
    genes = [
        "NPM1",
        "FLT3",
        "GATA2",
        "EZH2",
        "CREBBP",
        "TP53",
        "ASXL1",
        "ZRSR2",
        "SF3B1",
        "DNMT3A",
        "BCOR",
        "RUNX1",
        "STAG2",
        "ABL1",
        "NFE2",
        "DDX41",
        "CSNK1A1",
        "SH2B3",
        "SRSF2",
        "U2AF1",
        "NRAS",
        "GNB1",
        "CSF3R",
        "MPL",
        "HAX1",
        "RIT1",
        "SMC3",
        "WT1",
        "ATM",
        "CBL",
        "ETV6",
        "ETNK1",
        "KRAS",
        "ARID2",
        "PTPN11",
        "BRCA2",
        "PDS5B",
        "IDH2",
        "NF1",
        "PPM1D",
        "CEBPA",
        "IDH1",
        "MYD88",
        "KIT",
        "PHF6",
        "BCORL1",
        "JAK2",
        "CUX1",
        "VEGFA",
    ]
    features = []
    for gene in genes:
        gene_mut = (
            df[df["GENE"].str.upper() == gene]
            .groupby("ID")
            .size()
            .rename(f"{gene.lower()}_mutated")
        )
        features.append(gene_mut)
    return pd.concat(features, axis=1)


def extract_features_ref(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrait un indicateur basé sur la colonne 'REF'.
    """
    tca_ref = (
        df[df["REF"].str.upper() == "TCACCACTGCCATAGAGAGGCGGC"]
        .groupby("ID")
        .size()
        .rename("tca_ref")
    )
    return pd.concat([tca_ref], axis=1)


def extract_all_features_mol(df: pd.DataFrame, mutations) -> pd.DataFrame:
    """
    Concatène toutes les features moléculaires extraites en un seul DataFrame.
    """
    return pd.concat(
        [
            extract_features_mol(df),
            extract_features_effect(df),
            extract_features_gene(df),
            extract_features_ref(df),
            gene_risk(df),
            calc_weighted_gene_risk(df, mutations),
        ],
        axis=1,
    ).reset_index()
