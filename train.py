import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored

# Créer les répertoires s'ils n'existent pas
os.makedirs("models", exist_ok=True)
os.makedirs("submissions", exist_ok=True)
os.makedirs("figures", exist_ok=True)

print("Chargement des données...")
# Charger les données
X_train = pd.read_csv("data/processed/X_train_prepared.csv", index_col=0)
y_train = pd.read_csv("data/processed/y_train_prepared.csv", index_col=0)
X_val = pd.read_csv("data/processed/X_val_prepared.csv", index_col=0)
y_val = pd.read_csv("data/processed/y_val_prepared.csv", index_col=0)
X_test = pd.read_csv("data/processed/X_test_prepared.csv", index_col=0)

print(
    f"Dimensions des données - X_train: {X_train.shape}, y_train: {y_train.shape}"
)
print(f"X_val: {X_val.shape}, y_val: {y_val.shape}, X_test: {X_test.shape}")

# Vérification des valeurs manquantes
print(f"Valeurs manquantes dans X_train: {X_train.isnull().sum().sum()}")
print(f"Valeurs manquantes dans y_train: {y_train.isnull().sum().sum()}")
print(f"Valeurs manquantes dans X_val: {X_val.isnull().sum().sum()}")
print(f"Valeurs manquantes dans y_val: {y_val.isnull().sum().sum()}")

# Supprimer les lignes avec des valeurs manquantes
y_train = y_train.dropna()
y_val = y_val.dropna()

# S'assurer que les indices sont communs entre X et y
common_index_train = X_train.index.intersection(y_train.index)
common_index_val = X_val.index.intersection(y_val.index)

X_train = X_train.loc[common_index_train]
y_train = y_train.loc[common_index_train]
X_val = X_val.loc[common_index_val]
y_val = y_val.loc[common_index_val]

print(
    f"Après filtrage des valeurs manquantes - X_train: {X_train.shape}, y_train: {y_train.shape}"
)
print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")

# Convertir les colonnes de y en types appropriés
y_train["OS_STATUS"] = y_train["OS_STATUS"].astype(bool)
y_val["OS_STATUS"] = y_val["OS_STATUS"].astype(bool)

# Transformer y en format compatible avec scikit-survival
y_train_survival = np.array(
    [
        (status, time)
        for status, time in zip(y_train["OS_STATUS"], y_train["OS_YEARS"])
    ],
    dtype=[("event", bool), ("time", float)],
)

y_val_survival = np.array(
    [
        (status, time)
        for status, time in zip(y_val["OS_STATUS"], y_val["OS_YEARS"])
    ],
    dtype=[("event", bool), ("time", float)],
)

# Vérification des types et suppression des colonnes non numériques
print("Types de données dans X_train:")
print(X_train.dtypes.value_counts())

X_train = X_train.select_dtypes(include=["number"])
X_val = X_val.select_dtypes(include=["number"])
X_test = X_test.select_dtypes(include=["number"])

# S'assurer que les mêmes colonnes sont présentes dans tous les ensembles
common_cols = set(X_train.columns) & set(X_val.columns) & set(X_test.columns)
X_train = X_train[list(common_cols)]
X_val = X_val[list(common_cols)]
X_test = X_test[list(common_cols)]

print(f"Utilisation de {len(common_cols)} caractéristiques communes")

# Initialiser et entraîner le modèle de Cox avec régularisation L2
print("Entraînement du modèle de Cox...")
cox_model = CoxPHSurvivalAnalysis(alpha=0.1)  # L2 regularization
print(f"Dimensions de X_train : {X_train.shape}")
print(f"Dimensions de y_train_survival : {y_train_survival.shape}")

cox_model.fit(X_train, y_train_survival)

# Évaluer le modèle sur l'ensemble de validation
risk_scores_val = cox_model.predict(X_val)
c_index_val = concordance_index_censored(
    y_val["OS_STATUS"], y_val["OS_YEARS"], risk_scores_val
)[0]
print(f"📊 C-index du modèle de Cox sur validation : {c_index_val:.4f}")

# Examiner les coefficients pour comprendre l'importance des caractéristiques
coef_df = pd.DataFrame(
    {"Feature": X_train.columns, "Coefficient": cox_model.coef_}
)
coef_df["Abs_Coefficient"] = abs(coef_df["Coefficient"])
coef_df = coef_df.sort_values("Abs_Coefficient", ascending=False)

print("\nTop 10 caractéristiques les plus importantes :")
for i, row in coef_df.head(10).iterrows():
    print(f"- {row['Feature']}: {row['Coefficient']:.4f}")


# Entraîner également un Random Survival Forest pour comparaison
print("\nEntraînement du modèle Random Survival Forest pour comparaison...")
rsf = RandomSurvivalForest(
    n_estimators=100,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features="sqrt",
    n_jobs=-1,
    random_state=42,
)
rsf.fit(X_train, y_train_survival)

# Évaluer le modèle RSF
risk_scores_rsf = rsf.predict(X_val)
c_index_rsf = concordance_index_censored(
    y_val["OS_STATUS"], y_val["OS_YEARS"], risk_scores_rsf
)[0]
print(
    f" C-index du modèle Random Survival Forest sur validation : {c_index_rsf:.4f}"
)


# Générer les prédictions sur les données de test
risk_scores_test_cox = cox_model.predict(X_test)
risk_scores_test_rsf = rsf.predict(X_test)

# Créer le fichier de soumission pour le modèle Cox
submission_cox = pd.DataFrame(
    {"ID": X_test.index, "risk_score": risk_scores_test_cox}
)
submission_cox.to_csv("submissions/cox_submission.csv", index=False)

# Créer le fichier de soumission pour le RSF
submission_rsf = pd.DataFrame(
    {"ID": X_test.index, "risk_score": risk_scores_test_rsf}
)
submission_rsf.to_csv("submissions/rsf_submission.csv", index=False)


print(" Fichiers de soumission générés :")
print("   - submissions/cox_submission.csv")
print("   - submissions/rsf_submission.csv")


print(" Analyse de survie terminée avec succès !")
