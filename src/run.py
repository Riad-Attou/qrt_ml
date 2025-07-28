import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

from sklearn.model_selection import train_test_split
from sksurv.util import Surv

from data.data_preparation import traitement_donnees
from evaluation.evaluator import Evaluator
from models.lightgbm_model import LightGBMModel


def main():
    # Paramètres d'exécution
    nb_classes = 56  # 39 pour opti score_train, 56 pour opti score_test
    max_depth = 15
    max_depth_cyto = 10

    print(
        f"Exécution pour nb_classes = {nb_classes}, max_depth = {max_depth}, max_depth_cyto = {max_depth_cyto}"
    )

    # Préparer les données via le nouveau pipeline
    merged_train, merged_test, target_df = traitement_donnees(
        nb_classes, max_depth, max_depth_cyto
    )

    target_df = target_df.dropna(subset=["OS_STATUS", "OS_YEARS"])

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
        # "gnb1_mutated",
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
        "genetic_score",
        "cytogene_score",
    ]

    # Filtrer pour garder uniquement les IDs communs
    common_ids = set(merged_train["ID"]).intersection(set(target_df["ID"]))
    merged_train = merged_train[merged_train["ID"].isin(common_ids)]
    target_df = target_df[target_df["ID"].isin(common_ids)]

    # Trier pour garantir un ordre identique
    merged_train = merged_train.sort_values("ID").reset_index(drop=True)
    target_df = target_df.sort_values("ID").reset_index(drop=True)

    # Préparer X et y à partir de merged_train et target_df
    X = merged_train[features]
    y_df = target_df[["OS_STATUS", "OS_YEARS"]]

    # Réaliser un split train/test
    X_train, X_test, y_train_df, y_test_df = train_test_split(
        X, y_df, test_size=0.01, random_state=42
    )

    # Convertir les DataFrames de cibles en structures de survie
    y_train = Surv.from_dataframe("OS_STATUS", "OS_YEARS", y_train_df)
    y_test = Surv.from_dataframe("OS_STATUS", "OS_YEARS", y_test_df)

    # Entraîner le modèle LightGBM
    model = LightGBMModel(
        params={"max_depth": 3, "learning_rate": 0.038, "verbose": -1}
    )
    model.train(X_train, y_train_df["OS_YEARS"])

    # Obtenir les prédictions sur train et test
    pred_train = -model.predict(X_train)
    pred_test = -model.predict(X_test)

    # Utiliser l'évaluateur pour obtenir les scores sur train et test
    evaluator = Evaluator(tau=7)
    train_ci, test_ci = evaluator.evaluate_both(
        y_train, pred_train, y_test, pred_test
    )

    print(f"Concordance Index IPCW on train: {train_ci:.8f}")
    print(f"Concordance Index IPCW on test:  {test_ci:.4f}")


if __name__ == "__main__":
    main()
