import pandas as pd

target_train = pd.read_csv("databases/X_train/target_train.csv")
target_test = pd.read_csv("pred_test.csv")
mol_train = pd.read_csv("databases/X_train/molecular_train.csv")
mol_test = pd.read_csv("databases/X_test/molecular_test.csv")


mol_train_risk_max = mol_train[
    mol_train["ID"].isin(target_train[target_train["OS_YEARS"] > 10]["ID"])
]
mol_test_risk_max = mol_test[
    mol_test["ID"].isin(target_test[target_test["risk_score"]])
]
print(len(mol_train_risk_max))
