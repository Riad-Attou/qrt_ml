import pandas as pd

# Ajuster l'option pour afficher toutes les lignes et toutes les colonnes
pd.set_option("display.max_rows", None)  # Afficher toutes les lignes
pd.set_option("display.max_columns", None)  # Afficher toutes les colonnes
pd.set_option("display.width", None)  # Pas de limite de largeur
pd.set_option("display.max_colwidth", None)  # Pas de limite sur la largeur des colonnes


def load_data():
    target_train = pd.read_csv("databases/X_train/target_train.csv")
    target_test = pd.read_csv("databases/X_test/pred_test_riad.csv")
    mol_train = pd.read_csv("databases/X_train/molecular_train.csv")
    mol_test = pd.read_csv("databases/X_test/molecular_test.csv")
    return target_train, target_test, mol_train, mol_test


# Fonction pour filtrer et joindre les données de mol_train
def process_mol_train(mol_train, target_train, survival_time_limit):
    mol_train_risk_max = mol_train[
        mol_train["ID"].isin(
            target_train[target_train["OS_YEARS"] > survival_time_limit]["ID"]
        )
    ]
    mol_train_risk_max = mol_train_risk_max.merge(
        target_train[["ID", "OS_YEARS"]], on="ID", how="left"
    )
    mol_train_risk_max = mol_train_risk_max.sort_values(by="OS_YEARS", ascending=True)
    return mol_train_risk_max


# Fonction pour filtrer et joindre les données de mol_test
def process_mol_test(mol_test, target_test, risk_score_limit):
    mol_test_risk_max = mol_test[
        mol_test["ID"].isin(
            target_test[target_test["risk_score"] > risk_score_limit]["ID"]
        )
    ]
    mol_test_risk_max = mol_test_risk_max.merge(
        target_test[["ID", "risk_score"]], on="ID", how="left"
    )
    mol_test_risk_max = mol_test_risk_max.sort_values(by="risk_score", ascending=True)
    return mol_test_risk_max


def display_data(mol_train_risk_max, mol_test_risk_max):
    print(
        mol_train_risk_max[
            ["ID", "GENE", "PROTEIN_CHANGE", "EFFECT", "VAF", "DEPTH", "OS_YEARS"]
        ]
    )
    print(
        mol_test_risk_max[
            ["ID", "GENE", "PROTEIN_CHANGE", "EFFECT", "VAF", "DEPTH", "risk_score"]
        ]
    )


def tp53_vaf(df, cas, target_train, target_test):
    mol_tp53_vaf = df[df["GENE"] == "TP53"].groupby("ID")["VAF"].sum().reset_index()
    if cas == "train":
        score = "OS_YEARS"
        mol_tp53_vaf = mol_tp53_vaf.merge(
            target_train[["ID", score]], on="ID", how="left"
        )
        mol_tp53_vaf = mol_tp53_vaf.sort_values(by=score, ascending=True)
    elif cas == "test":
        score = "risk_score"
        mol_tp53_vaf = mol_tp53_vaf.merge(
            target_test[["ID", score]], on="ID", how="left"
        )
        mol_tp53_vaf = mol_tp53_vaf.sort_values(by=score, ascending=True)
    return mol_tp53_vaf


def sf3b1_vaf(df, cas, target_train, target_test):
    mol_sf3b1_vaf = df[df["GENE"] == "SF3B1"].groupby("ID")["VAF"].sum().reset_index()
    if cas == "train":
        score = "OS_YEARS"
        mol_sf3b1_vaf = mol_sf3b1_vaf.merge(
            target_train[["ID", score]], on="ID", how="left"
        )
        mol_sf3b1_vaf = mol_sf3b1_vaf.sort_values(by=score, ascending=True)
    elif cas == "test":
        score = "risk_score"
        mol_sf3b1_vaf = mol_sf3b1_vaf.merge(
            target_test[["ID", score]], on="ID", how="left"
        )
        mol_sf3b1_vaf = mol_sf3b1_vaf.sort_values(by=score, ascending=True)
    return mol_sf3b1_vaf


def main():
    survival_time_limit = 11
    risk_score_limit = 1
    target_train, target_test, mol_train, mol_test = load_data()
    mol_train_risk_max = process_mol_train(mol_train, target_train, survival_time_limit)
    mol_test_risk_max = process_mol_test(mol_test, target_test, risk_score_limit)
    display_data(mol_train_risk_max, mol_test_risk_max)
    tp53_vaf_test = tp53_vaf(mol_test_risk_max, "test", target_train, target_test)
    tp53_vaf_train = tp53_vaf(mol_train_risk_max, "train", target_train, target_test)
    print(tp53_vaf_test)
    print(tp53_vaf_train)
    sf3b1_vaf_test = sf3b1_vaf(mol_test_risk_max, "test", target_train, target_test)
    sf3b1_vaf_train = sf3b1_vaf(mol_train_risk_max, "train", target_train, target_test)
    print(sf3b1_vaf_test)
    print(sf3b1_vaf_train)


main()
