import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored

# Créer les répertoires s'ils n'existent pas
os.makedirs("models", exist_ok=True)
os.makedirs("submissions", exist_ok=True)

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

# Transformer y en format compatible avec scikit-survival
y_train_survival = np.array(
    [
        (bool(status), time)
        for status, time in zip(y_train["OS_STATUS"], y_train["OS_YEARS"])
    ],
    dtype=[("event", bool), ("time", float)],
)

y_val_survival = np.array(
    [
        (bool(status), time)
        for status, time in zip(y_val["OS_STATUS"], y_val["OS_YEARS"])
    ],
    dtype=[("event", bool), ("time", float)],
)

# Initialiser et entraîner le modèle de Cox avec régularisation
print("Entraînement du modèle de Cox avec régularisation...")
cox_model = CoxPHSurvivalAnalysis(
    alpha=1.0
)  # Forte régularisation L2 pour éviter les problèmes numériques
print(f"Dimensions de X_train : {X_train.shape}")
print(f"Dimensions de y_train_survival : {y_train_survival.shape}")

# Vérifier la présence de multicolinéarité
print("\nVérification de la multicolinéarité...")
correlation_matrix = X_train.corr().abs()
upper_tri = correlation_matrix.where(
    np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
)
high_correlation = [
    column for column in upper_tri.columns if any(upper_tri[column] > 0.95)
]
print(f"Caractéristiques hautement corrélées (r > 0.95): {high_correlation}")

if high_correlation:
    print(f"Suppression des caractéristiques hautement corrélées...")
    X_train = X_train.drop(columns=high_correlation)
    X_val = X_val.drop(columns=high_correlation)
    X_test = X_test.drop(columns=high_correlation)
    print(f"Nouvelles dimensions - X_train: {X_train.shape}")

# Vérifier les valeurs extrêmes
print("\nVérification des valeurs extrêmes...")
for col in X_train.columns:
    q1 = X_train[col].quantile(0.01)
    q3 = X_train[col].quantile(0.99)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Limiter les valeurs extrêmes
    X_train[col] = X_train[col].clip(lower_bound, upper_bound)
    X_val[col] = X_val[col].clip(lower_bound, upper_bound)
    X_test[col] = X_test[col].clip(lower_bound, upper_bound)

# Réduire le nombre de caractéristiques si nécessaire
if X_train.shape[1] > 10:  # Si plus de 10 caractéristiques
    print("\nRéduction du nombre de caractéristiques...")
    # Garder les caractéristiques du benchmark si elles existent
    benchmark_features = ["BM_BLAST", "HB", "PLT", "Nmut"]
    # Filtrer pour ne garder que les caractéristiques disponibles
    available_benchmark_features = [
        f for f in benchmark_features if f in X_train.columns
    ]

    # Si moins de 4 caractéristiques du benchmark sont disponibles, sélectionner les plus importantes
    if len(available_benchmark_features) < 4:
        print(
            f"Caractéristiques du benchmark disponibles: {available_benchmark_features}"
        )
        print("Sélection des caractéristiques les plus importantes...")

        # Ajuster un modèle avec une forte régularisation pour sélectionner les caractéristiques
        temp_model = CoxPHSurvivalAnalysis(alpha=10.0)
        temp_model.fit(X_train, y_train_survival)

        # Créer un DataFrame avec les coefficients et leur importance
        coefs = pd.DataFrame(
            {"feature": X_train.columns, "importance": abs(temp_model.coef_)}
        ).sort_values("importance", ascending=False)

        # Sélectionner les 4 caractéristiques les plus importantes
        top_features = coefs.head(4)["feature"].tolist()
        print(f"Caractéristiques les plus importantes: {top_features}")

        # Combiner avec les caractéristiques du benchmark
        selected_features = list(
            set(available_benchmark_features + top_features)
        )[:4]
    else:
        selected_features = available_benchmark_features

    print(f"Caractéristiques finales sélectionnées: {selected_features}")
    X_train = X_train[selected_features]
    X_val = X_val[selected_features]
    X_test = X_test[selected_features]

# Réentraîner le modèle avec les caractéristiques sélectionnées
cox_model = CoxPHSurvivalAnalysis(alpha=1.0)
cox_model.fit(X_train, y_train_survival)

# Évaluer le modèle sur l'ensemble de validation
risk_scores_val = cox_model.predict(X_val)
c_index_val = concordance_index_censored(
    y_val["OS_STATUS"].astype(bool), y_val["OS_YEARS"], risk_scores_val
)[0]
print(f"📊 C-index du modèle de Cox sur validation : {c_index_val:.4f}")

# Examiner les coefficients pour comprendre l'importance des caractéristiques
coef_df = pd.DataFrame(
    {"Feature": X_train.columns, "Coefficient": cox_model.coef_}
)
coef_df["Abs_Coefficient"] = abs(coef_df["Coefficient"])
coef_df = coef_df.sort_values("Abs_Coefficient", ascending=False)

print("\nCoefficients du modèle Cox (importance des variables) :")
for i, row in coef_df.iterrows():
    print(f"- {row['Feature']}: {row['Coefficient']:.4f}")

# Sauvegarder le modèle
joblib.dump(cox_model, "models/cox_model.pkl")
print("✅ Modèle de Cox sauvegardé !")

# Prédictions sur les données de test (risque)
risk_scores_test = cox_model.predict(X_test)

# Créer le fichier de soumission au format attendu
submission = pd.DataFrame({"ID": X_test.index, "risk_score": risk_scores_test})
submission.to_csv("submissions/cox_submission.csv", index=False)

print("✅ Fichier de soumission généré : submissions/cox_submission.csv")

# Statistiques des scores de risque
print("\nStatistiques des scores de risque sur le jeu de test:")
print(f"Min: {risk_scores_test.min():.4f}, Max: {risk_scores_test.max():.4f}")
print(
    f"Moyenne: {risk_scores_test.mean():.4f}, Médiane: {np.median(risk_scores_test):.4f}"
)
print(f"Écart-type: {risk_scores_test.std():.4f}")

print("🎉 Analyse de survie terminée avec succès !")
os.makedirs("models", exist_ok=True)
os.makedirs("submissions", exist_ok=True)

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

# Transformer y en format compatible avec scikit-survival
y_train_survival = np.array(
    [
        (bool(status), time)
        for status, time in zip(y_train["OS_STATUS"], y_train["OS_YEARS"])
    ],
    dtype=[("event", bool), ("time", float)],
)

y_val_survival = np.array(
    [
        (bool(status), time)
        for status, time in zip(y_val["OS_STATUS"], y_val["OS_YEARS"])
    ],
    dtype=[("event", bool), ("time", float)],
)

# Initialiser et entraîner le modèle de Cox sans régularisation (comme dans le benchmark)
print("Entraînement du modèle de Cox...")
cox_model = CoxPHSurvivalAnalysis()
print(f"Dimensions de X_train : {X_train.shape}")
print(f"Dimensions de y_train_survival : {y_train_survival.shape}")

cox_model.fit(X_train, y_train_survival)

# Évaluer le modèle sur l'ensemble de validation
risk_scores_val = cox_model.predict(X_val)
c_index_val = concordance_index_censored(
    y_val["OS_STATUS"].astype(bool), y_val["OS_YEARS"], risk_scores_val
)[0]
print(f"📊 C-index du modèle de Cox sur validation : {c_index_val:.4f}")

# Examiner les coefficients pour comprendre l'importance des caractéristiques
coef_df = pd.DataFrame(
    {"Feature": X_train.columns, "Coefficient": cox_model.coef_}
)
coef_df["Abs_Coefficient"] = abs(coef_df["Coefficient"])
coef_df = coef_df.sort_values("Abs_Coefficient", ascending=False)

print("\nCoefficients du modèle Cox (importance des variables) :")
for i, row in coef_df.iterrows():
    print(f"- {row['Feature']}: {row['Coefficient']:.4f}")

# Sauvegarder le modèle
joblib.dump(cox_model, "models/cox_model.pkl")
print("✅ Modèle de Cox sauvegardé !")

# Prédictions sur les données de test (risque)
risk_scores_test = cox_model.predict(X_test)

# Créer le fichier de soumission au format attendu
submission = pd.DataFrame({"ID": X_test.index, "risk_score": risk_scores_test})
submission.to_csv("submissions/cox_submission.csv", index=False)

print("✅ Fichier de soumission généré : submissions/cox_submission.csv")

# Statistiques des scores de risque
print("\nStatistiques des scores de risque sur le jeu de test:")
print(f"Min: {risk_scores_test.min():.4f}, Max: {risk_scores_test.max():.4f}")
print(
    f"Moyenne: {risk_scores_test.mean():.4f}, Médiane: {np.median(risk_scores_test):.4f}"
)
print(f"Écart-type: {risk_scores_test.std():.4f}")

print("🎉 Analyse de survie terminée avec succès !")
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

# Visualiser les coefficients
plt.figure(figsize=(12, 8))
plt.barh(coef_df.head(15)["Feature"], coef_df.head(15)["Coefficient"])
plt.xlabel("Coefficient")
plt.ylabel("Feature")
plt.title("Top 15 caractéristiques importantes dans le modèle Cox")
plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("figures/cox_feature_importance.png")


# Générer les prédictions sur les données de test
risk_scores_test_cox = cox_model.predict(X_test)


# Créer le fichier de soumission pour le modèle Cox
submission_cox = pd.DataFrame(
    {"ID": X_test.index, "risk_score": risk_scores_test_cox}
)
submission_cox.to_csv("submissions/cox_submission.csv", index=False)


print("✅ Fichiers de soumission générés :")
print("   - submissions/cox_submission.csv")


# Bonus: Visualiser la distribution des scores de risque
plt.figure(figsize=(10, 6))
plt.hist(risk_scores_test_cox, bins=30, alpha=0.5, label="Cox")
plt.title("Distribution des scores de risque sur les données de test")
plt.xlabel("Score de risque")
plt.ylabel("Nombre de patients")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("figures/risk_scores_distribution.png")

print("🎉 Analyse de survie terminée avec succès !")
