import pandas as pd

from features.molecular_features import extract_all_features_mol


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
    db,  # Peut être None dans le nouveau pipeline
    mutations,
    target_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Fusionne les données cliniques et les features moléculaires.

    Ici, on réutilise le DataFrame complété par create_features_mol_df qui contient déjà
    des features moléculaires. Si un objet db est fourni, on utilise ses attributs pour ajouter
    des scores ; sinon, on assigne des valeurs par défaut (par exemple, None).
    """
    mol_agg = extract_all_features_mol(mol_df, mutations)

    # Si db est fourni, utiliser ses scores, sinon définir des valeurs par défaut
    if db is not None:
        mol_agg["genetic_score"] = mol_agg["ID"].apply(
            lambda pid: db.patient_test_combinations.get(pid, None)
        )
        mol_agg["cytogene_score"] = mol_agg["ID"].apply(
            lambda pid: db.cyto_test_combinations.get(pid, None)
        )
    else:
        mol_agg["genetic_score"] = None
        mol_agg["cytogene_score"] = None

    merged = merge_df(clin_df, target_df=target_df)
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
