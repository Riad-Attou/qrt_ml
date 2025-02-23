import re
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sksurv.metrics import concordance_index_ipcw
from sksurv.util import Surv

import lightgbm as lgb

######################################################################
# 1. Définition des nouvelles features à partir des données cliniques
######################################################################


def monosomie(cyto_str: Any) -> Any:
    """
    Renvoie le nombre de monosomies que présente un patient.
    Si l'entrée n'est pas une chaîne, retourne "inconnu".
    """
    if not isinstance(cyto_str, str):
        return "inconnu"
    s = cyto_str.lower()
    monosomies = re.findall(r"-(\d+)", s)
    return len(monosomies)


def classify_cytogenetic_risk(cyto_str: Any) -> str:
    """
    Classifie le risque cytogénétique en se basant sur des motifs connus.
    Renvoie une chaîne ("favorable", "intermediaire", "defavorable") ou "inconnu".
    """
    if not isinstance(cyto_str, str):
        return "inconnu"
    s = cyto_str.lower()

    # Cas favorables
    if ("t(15;17)" in s) or ("t(8;21)" in s) or ("inv(16)" in s):
        return "favorable"

    # Cas défavorables
    if (
        ("del(5)" in s)
        or ("del(7)" in s)
        or ("trisomy 8" in s)
        or ("complex" in s)
        or ("t(3;3)" in s)
        or ("inv(3)" in s)
        or ("t(11;19)" in s)
        or ("del(5q)" in s)
        or ("-7" in s)
        or ("7" in s)
        or ("-17" in s)
        or (monosomie(s) > 2)
    ):
        return "defavorable"

    # Sinon, c'est plutôt intermediaire (ex : caryotype normal)
    return "intermediaire"


def extract_features(cyto_str: Any) -> pd.Series:
    """
    Extrait plusieurs caractéristiques à partir de la chaîne cytogénétique.
    Renvoie un Series avec :
      - nb_del : nombre de délétion
      - nb_t : nombre de translocations hors t(9;22)
      - nb_philadelphia : indicateur (1/0) de présence de la translocation t(9;22)
      - nb_dup : nombre de duplications
      - nb_inv : nombre d'inversions
      - complex_karyotype : indicateur (1/0) si la chaîne contient "complex"
    """
    if not isinstance(cyto_str, str):
        return pd.Series(
            {
                "nb_del": 0,
                "nb_t": 0,
                "nb_philadelphia": 0,
                "nb_dup": 0,
                "nb_inv": 0,
                "complex_karyotype": 0,
            }
        )
    s = cyto_str.lower().strip()
    complex_karyotype = 1 if "complex" in s else 0
    nb_del = s.count("del")
    nb_t_total = s.count("t")
    nb_philadelphia = 1 if "t(9;22)" in s else 0
    nb_t = max(nb_t_total - nb_philadelphia, 0)
    nb_dup = s.count("dup")
    nb_inv = s.count("inv")
    return pd.Series(
        {
            "nb_del": nb_del,
            "nb_t": nb_t,
            "nb_philadelphia": nb_philadelphia,
            "nb_dup": nb_dup,
            "nb_inv": nb_inv,
            "complex_karyotype": complex_karyotype,
        }
    )


def create_features_clinical_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrichit le DataFrame clinique en ajoutant :
      - Les features extraites depuis la colonne 'CYTOGENETICS'
      - La classification du risque cytogénétique
      - Le nombre de monosomies
      - Des variables dummies à partir de la classification
    """
    features_df = df["CYTOGENETICS"].apply(extract_features)
    df = pd.concat([df, features_df], axis=1)
    df["cyto_risk"] = df["CYTOGENETICS"].apply(classify_cytogenetic_risk)
    df["nbr_mono"] = df["CYTOGENETICS"].apply(monosomie)
    cyto_risk_dummies = pd.get_dummies(df["cyto_risk"], prefix="cyto_risk")
    df = pd.concat([df, cyto_risk_dummies], axis=1)
    return df


########################################################################
# 2. Définition des nouvelles features à partir des données moléculaires
########################################################################


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
            "SF3B1",
            "RUNX1",
            "SRSF2",
            "STAG2",
            "U2AF1",
            "ZRSR2",
            "ABL1",
            "CREBBP",
            "GATA2",
            "TETA2",
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
    df: pd.DataFrame, mutations: List[Set[Any]]
) -> pd.DataFrame:
    """
    Ajoute des colonnes de classification du risque génétique et des variables dummies.
    """
    df = df.copy()
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
        "TET2",
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


pire_mutations = ["SF3B1", "TP53"]
meilleures_mutations = ["NPM1"]


def pire_meilleure_mutations(
    df: pd.DataFrame,
    pires_mutations: List[str],
    meilleures_mutations: List[str],
) -> pd.DataFrame:
    """
    Calcule plusieurs indicateurs basés sur des mutations considérées comme pires ou meilleures.
    """
    pire_mutation = (
        df[df["GENE"].str.upper().isin(piresh for piresh in pires_mutations)]
        .groupby("ID")
        .size()
        .rename("pire_mutations")
    )
    meilleure_mutation = (
        df[~df["GENE"].str.upper().isin(["FLT3"])]
        .loc[df["GENE"].str.upper().isin(meilleures_mutations)]
        .groupby("ID")
        .size()
        .rename("meilleure_mutations")
    )
    pire_mutation_vaf = (
        df[df["GENE"].str.upper().isin(piresh for piresh in pires_mutations)]
        .groupby("ID")["VAF"]
        .max()
        .rename("pire_mutation_vaf_max")
    )
    return pd.concat(
        [meilleure_mutation, pire_mutation, pire_mutation_vaf], axis=1
    )


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
            pire_meilleure_mutations(df, pire_mutations, meilleures_mutations),
            calc_weighted_gene_risk(df, mutations),
        ],
        axis=1,
    ).reset_index()


###################################################
# 3. Fusion des données cliniques et moléculaires
###################################################


def merge_df(
    clin_df: pd.DataFrame, target_df: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Fusionne les données cliniques en conservant uniquement certaines colonnes.
    Si target_df est fourni et que clin_df contient plus de 2000 lignes, effectue une jointure.
    """
    clinical_cols = [
        "ID",
        "nb_del",
        "nb_t",
        "nb_philadelphia",
        "nb_dup",
        "nb_inv",
        "complex_karyotype",
        "cyto_risk_intermediaire",
        "cyto_risk_defavorable",
        "CLASSE_CYT",
        "BM_BLAST",
        "PLT",
        "HB",
        "MONOCYTES",
        "WBC",
        "ANC",
        "nbr_mono",
    ]
    merged_df = clin_df[clinical_cols].copy()
    if len(clin_df) > 2000 and target_df is not None:
        merged_df = pd.merge(target_df, merged_df, on="ID", how="left")
    merged_df[
        [
            "nb_del",
            "nb_t",
            "nb_philadelphia",
            "nb_dup",
            "nb_inv",
            "complex_karyotype",
        ]
    ] = merged_df[
        [
            "nb_del",
            "nb_t",
            "nb_philadelphia",
            "nb_dup",
            "nb_inv",
            "complex_karyotype",
        ]
    ].fillna(
        0
    )
    return merged_df


def fusion_df(
    clin_df: pd.DataFrame,
    mol_df: pd.DataFrame,
    mutations,
    target_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Fusionne les données cliniques et les features moléculaires.
    """
    mol_agg = extract_all_features_mol(mol_df, mutations)
    merged = merge_df(clin_df, target_df=None)
    merged = pd.merge(merged, mol_agg, on="ID", how="left")
    merged.fillna(0, inplace=True)
    return merged


def trad_ris_int(merged_df: pd.DataFrame) -> None:
    """
    Convertit certaines colonnes en type numérique.
    """
    for col in ["cyto_risk_intermediaire", "cyto_risk_defavorable"]:
        merged_df[col] = pd.to_numeric(merged_df[col], errors="coerce")
    for col in [
        "gene_risk_favorable",
        "gene_risk_defavorable",
        "gene_risk_intermediaire",
    ]:
        merged_df[col] = pd.to_numeric(merged_df[col], errors="coerce")


#########################
# 4. Lecture des données
#########################


def charger_donnees() -> (
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
):
    # Charger les données cliniques
    df_train = pd.read_csv(
        "databases/X_train/new_df_train.csv", sep=",", quotechar='"'
    )
    df_train["ID"] = df_train["ID"].astype(str)
    df_eval = pd.read_csv("databases/X_test/new_df_test.csv")
    df_eval["ID"] = df_eval["ID"].astype(str)

    # Charger les données moléculaires
    mol_df = pd.read_csv(
        "databases/X_train/molecular_train.csv", sep=",", quotechar='"'
    )
    mol_df["ID"] = mol_df["ID"].astype(str)
    mol_eval = pd.read_csv(
        "databases/X_test/molecular_test.csv", sep=",", quotechar='"'
    )
    mol_eval["ID"] = mol_eval["ID"].astype(str)

    # Charger les données des prédictions
    target_df = pd.read_csv("databases/X_train/target_train.csv")
    target_df = target_df[target_df["ID"].isin(df_train["ID"])]
    target_df["OS_YEARS"] = pd.to_numeric(
        target_df["OS_YEARS"], errors="coerce"
    )
    target_df["OS_STATUS"] = target_df["OS_STATUS"].astype(bool)

    return df_train, df_eval, mol_df, mol_eval, target_df


#############################
# 5. Préparation des données
#############################


def traitement_donnees(
    nbclasses: int,
    df_train: pd.DataFrame,
    df_eval: pd.DataFrame,
    mol_df: pd.DataFrame,
    mol_eval: pd.DataFrame,
    target_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    mutations = classes_mutations(nbclasses, target_df, df_train, mol_df)
    df_train_enrichi = create_features_clinical_df(df_train)
    df_eval_enrichi = create_features_clinical_df(df_eval)
    mol_df_enrichi = create_features_mol_df(mol_df, mutations)
    mol_eval_enrichi = create_features_mol_df(mol_eval, mutations)
    merged_train = fusion_df(
        df_train_enrichi, mol_df_enrichi, mutations, target_df
    )
    merged_test = fusion_df(
        df_eval_enrichi, mol_eval_enrichi, mutations, target_df
    )
    trad_ris_int(merged_train)
    trad_ris_int(merged_test)
    return merged_train, merged_test


#############################
# 6. Modélisation et prédictions
#############################


def modele_survival(
    cas,
    merged_train: pd.DataFrame,
    merged_test: pd.DataFrame,
    target_df: pd.DataFrame,
) -> None:
    features = [
        "BM_BLAST",
        "HB",
        "PLT",
        "WBC",
        "ANC",
        "nb_del",
        "nb_t",
        "nb_dup",
        "nb_inv",
        "complex_karyotype",
        "nb_philadelphia",
        "cyto_risk_defavorable",
        "cyto_risk_intermediaire",
        "CLASSE_CYT",
        "gene_risk_favorable",
        "gene_risk_defavorable",
        "max_VAF",
        "stop_gained_count",
        "frameshift_variant_count",
        "non_synonymous_count",
        "weighted_stop_gained",
        "weighted_frameshift",
        "weighted_non_synonymous",
        "min_depth",
        "flt3_mutated",
        "npm1_mutated",
        "pire_mutations",
        "dnmt3a_mutated",
        "asxl1_mutated",
        "total_mutations",
        "tca_ref",
        "gata2_mutated",
        "ezh2_mutated",
        "crebbp_mutated",
        "zrsr2_mutated",
        "nbr_mono",
        "bcor_mutated",
        "runx1_mutated",
        "stag2_mutated",
        "abl1_mutated",
        "nfe2_mutated",
        "ddx41_mutated",
        "csnk1a1_mutated",
        "sh2b3_mutated",
        "gene_risk_pondere",
        "tp53_mutated",
        "srsf2_mutated",
        "u2af1_mutated",
        "nras_mutated",
        "gnb1_mutated",
        "csf3r_mutated",
        "mpl_mutated",
        "hax1_mutated",
        "rit1_mutated",
        "smc3_mutated",
        "wt1_mutated",
        "atm_mutated",
        "cbl_mutated",
        "etv6_mutated",
        "etnk1_mutated",
        "kras_mutated",
        "arid2_mutated",
        "ptpn11_mutated",
        "brca2_mutated",
        "pds5b_mutated",
        "idh2_mutated",
        "nf1_mutated",
        "ppm1d_mutated",
        "cebpa_mutated",
        "idh1_mutated",
        "myd88_mutated",
        "kit_mutated",
        "phf6_mutated",
        "bcorl1_mutated",
        "jak2_mutated",
        "cux1_mutated",
        "vegfa_mutated",
    ]
    # Préparation de la donnée cible
    target_df = target_df.dropna(subset=["OS_YEARS", "OS_STATUS"])
    X = merged_train.loc[merged_train["ID"].isin(target_df["ID"]), features]
    y = Surv.from_dataframe("OS_STATUS", "OS_YEARS", target_df)
    if cas == 1:
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.01, random_state=42
        )
    if cas == 2:
        X_train = X
        y_train = y
        X_test = merged_test[features]
        y_test = None

    # Paramètres LightGBM
    lgbm_params = {"max_depth": 3, "learning_rate": 0.038, "verbose": -1}
    y_train_transformed = y_train["OS_YEARS"]

    train_dataset = lgb.Dataset(X_train, label=y_train_transformed)
    model = lgb.train(params=lgbm_params, train_set=train_dataset)

    pred_train = -model.predict(X_train)
    pred_test = -model.predict(X_test)
    risk_score = 1 / (np.abs(pred_test) + 1e-6)
    df_risk_score = pd.DataFrame(risk_score, columns=["risk_score"])
    # On suppose que merged_test contient la colonne "ID"
    df_risk_score.insert(0, "ID", merged_test["ID"])
    df_risk_score.to_csv("pred_test.csv", index=False)

    train_ci_ipcw = concordance_index_ipcw(
        y_train, y_train, pred_train, tau=7
    )[0]
    print(
        f"LightGBM Survival Model Concordance Index IPCW on train: {train_ci_ipcw:.8f}"
    )
    if y_test is not None:
        test_ci_ipcw = concordance_index_ipcw(
            y_test, y_test, pred_test, tau=7
        )[0]
        print(
            f"LightGBM Survival Model Concordance Index IPCW on test: {test_ci_ipcw:.4f}"
        )
    return train_ci_ipcw


if __name__ == "__main__":
    nbclasses = 100
    df_train, df_eval, mol_df, mol_eval, target_df = charger_donnees()
    merged_train, merged_test = traitement_donnees(
        nbclasses, df_train, df_eval, mol_df, mol_eval, target_df
    )
    modele_survival(2, merged_train, merged_test, target_df)
