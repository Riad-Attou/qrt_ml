import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw

# Créer les répertoires nécessaires
os.makedirs("models", exist_ok=True)
os.makedirs("submissions", exist_ok=True)
os.makedirs("figures", exist_ok=True)


def load_prepared_data():
    """Charge les données préparées et retourne les DataFrames."""
    print("Chargement des données préparées...")
    X_train = pd.read_csv("data/processed/X_train_prepared.csv", index_col=0)
    y_train = pd.read_csv("data/processed/y_train_prepared.csv", index_col=0)
    X_val = pd.read_csv("data/processed/X_val_prepared.csv", index_col=0)
    y_val = pd.read_csv("data/processed/y_val_prepared.csv", index_col=0)
    X_test = pd.read_csv("data/processed/X_test_prepared.csv", index_col=0)

    print(
        f"Dimensions des données: \n"
        f"- X_train: {X_train.shape}\n"
        f"- y_train: {y_train.shape}\n"
        f"- X_val: {X_val.shape}\n"
        f"- y_val: {y_val.shape}\n"
        f"- X_test: {X_test.shape}"
    )

    # Vérification des valeurs manquantes
    print(f"Valeurs manquantes dans X_train: {X_train.isnull().sum().sum()}")
    print(f"Valeurs manquantes dans y_train: {y_train.isnull().sum().sum()}")

    return X_train, y_train, X_val, y_val, X_test


def prepare_survival_data(y_df):
    """Convertit les données de survie en format compatible avec scikit-survival."""
    # Assurez-vous que OS_STATUS est booléen
    y_df["OS_STATUS"] = y_df["OS_STATUS"].astype(bool)

    # Créer le format structured array pour scikit-survival
    y_surv = np.array(
        [
            (status, time)
            for status, time in zip(y_df["OS_STATUS"], y_df["OS_YEARS"])
        ],
        dtype=[("event", bool), ("time", float)],
    )

    return y_surv


def find_optimal_alpha(X_train, y_train, X_val, y_val, alphas=None):
    """Trouve la valeur optimale d'alpha (régularisation) pour le modèle Cox."""
    if alphas is None:
        alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

    best_alpha = None
    best_score = -np.inf
    scores = []

    print("Recherche de la meilleure régularisation (alpha)...")
    for alpha in alphas:
        model = CoxPHSurvivalAnalysis(alpha=alpha)
        try:
            model.fit(X_train, y_train)

            # Évaluer sur la validation
            pred = model.predict(X_val)
            score = concordance_index_censored(
                y_val["event"], y_val["time"], pred
            )[0]
            scores.append(score)

            print(f"Alpha = {alpha}: C-index = {score:.4f}")

            if score > best_score:
                best_score = score
                best_alpha = alpha
        except Exception as e:
            print(f"Erreur avec alpha={alpha}: {e}")
            scores.append(float("nan"))

    print(f"Meilleur alpha: {best_alpha} (C-index: {best_score:.4f})")

    # Tracer les résultats
    plt.figure(figsize=(10, 6))
    plt.semilogx(alphas, scores, "o-")
    plt.xlabel("Valeur d'alpha (régularisation L2)")
    plt.ylabel("C-index sur validation")
    plt.title("Performance du modèle Cox en fonction de la régularisation")
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.axvline(
        x=best_alpha,
        color="r",
        linestyle="--",
        label=f"Meilleur alpha = {best_alpha}",
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/cox_alpha_tuning.png")

    return best_alpha


def train_cox_model(X_train, y_train, alpha=0.1):
    """Entraîne un modèle de Cox avec la régularisation spécifiée."""
    print(f"Entraînement du modèle Cox avec alpha={alpha}...")

    # Créer et entraîner le modèle
    model = CoxPHSurvivalAnalysis(alpha=alpha)
    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X, y, dataset_name=""):
    """Évalue le modèle sur le jeu de données spécifié."""
    pred = model.predict(X)
    c_index = concordance_index_censored(y["event"], y["time"], pred)[0]
    print(f"C-index sur {dataset_name}: {c_index:.4f}")
    return c_index, pred


def visualize_feature_importance(model, feature_names):
    """Visualise l'importance des caractéristiques du modèle Cox."""
    coef = pd.Series(model.coef_, index=feature_names)
    coef_abs = coef.abs().sort_values(ascending=False)

    plt.figure(figsize=(12, 8))
    coef_abs.plot(kind="bar")
    plt.title(
        "Importance des caractéristiques (valeur absolue des coefficients)"
    )
    plt.ylabel("Importance")
    plt.xlabel("Caractéristiques")
    plt.tight_layout()
    plt.savefig("figures/cox_feature_importance.png")

    return coef


def create_submission(X_test, model, output_file="cox_submission.csv"):
    """Crée un fichier de soumission avec les prédictions du modèle."""
    # Prédire les scores de risque
    risk_scores = model.predict(X_test)

    # Créer le DataFrame de soumission
    submission = pd.DataFrame({"ID": X_test.index, "risk_score": risk_scores})

    # Sauvegarder
    submission_path = os.path.join("submissions", output_file)
    submission.to_csv(submission_path, index=False)
    print(f"Fichier de soumission généré: {submission_path}")

    return submission


def cross_validate_cox(X, y, n_folds=5, alpha=0.1):
    """Effectue une validation croisée du modèle Cox."""
    print(f"Validation croisée ({n_folds} folds) du modèle Cox...")

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    c_indices = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold = y[train_idx]

        # Construire y_val au format structuré
        y_val_fold_df = pd.DataFrame(
            {
                "OS_STATUS": [y_i[0] for y_i in y[val_idx]],
                "OS_YEARS": [y_i[1] for y_i in y[val_idx]],
            }
        )

        # Entraîner le modèle
        model = CoxPHSurvivalAnalysis(alpha=alpha)
        model.fit(X_train_fold, y_train_fold)

        # Évaluer le modèle
        pred = model.predict(X_val_fold)
        c_index = concordance_index_censored(
            y_val_fold_df["OS_STATUS"].astype(bool),
            y_val_fold_df["OS_YEARS"],
            pred,
        )[0]

        c_indices.append(c_index)
        print(f"Fold {fold+1}/{n_folds}: C-index = {c_index:.4f}")

    mean_c_index = np.mean(c_indices)
    std_c_index = np.std(c_indices)
    print(
        f"Performance moyenne sur {n_folds} folds: C-index = {mean_c_index:.4f} ± {std_c_index:.4f}"
    )

    return mean_c_index, std_c_index


def train_ensemble_model(X_train, y_train, X_val, y_val):
    """Entraîne un modèle d'ensemble combinant Cox et RandomSurvivalForest."""
    print("Entraînement d'un modèle d'ensemble...")

    # Trouver le meilleur alpha pour Cox
    best_alpha = find_optimal_alpha(X_train, y_train, X_val, y_val)

    # Entraîner le modèle Cox
    cox_model = CoxPHSurvivalAnalysis(alpha=best_alpha)
    cox_model.fit(X_train, y_train)

    # Entraîner un Random Survival Forest
    rsf_model = RandomSurvivalForest(
        n_estimators=100,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    )
    rsf_model.fit(X_train, y_train)

    # Évaluer les modèles individuels
    cox_score, cox_pred = evaluate_model(
        cox_model, X_val, y_val, "validation (Cox)"
    )
    rsf_score, rsf_pred = evaluate_model(
        rsf_model, X_val, y_val, "validation (RSF)"
    )

    # Créer et évaluer l'ensemble (moyenne des prédictions)
    ensemble_pred = (cox_pred + rsf_pred) / 2
    ensemble_c_index = concordance_index_censored(
        y_val["event"], y_val["time"], ensemble_pred
    )[0]
    print(f"C-index sur validation (Ensemble): {ensemble_c_index:.4f}")

    ensemble = {
        "cox_model": cox_model,
        "rsf_model": rsf_model,
        "alpha": best_alpha,
    }

    return ensemble, ensemble_c_index


def main():
    """Fonction principale pour l'entraînement et l'évaluation des modèles."""
    # Charger les données
    X_train, y_train_df, X_val, y_val_df, X_test = load_prepared_data()

    # Préparer les données de survie
    y_train = prepare_survival_data(y_train_df)
    y_val = prepare_survival_data(y_val_df)

    # Recherche du meilleur alpha
    best_alpha = find_optimal_alpha(X_train, y_train, X_val, y_val)

    # Validation croisée
    cv_score, cv_std = cross_validate_cox(
        pd.concat([X_train, X_val]),
        np.concatenate([y_train, y_val]),
        n_folds=5,
        alpha=best_alpha,
    )

    # Entraîner le modèle final avec toutes les données d'entraînement et validation
    print("Entraînement du modèle final...")
    X_train_full = pd.concat([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])

    final_model = train_cox_model(X_train_full, y_train_full, alpha=best_alpha)

    # Visualiser l'importance des caractéristiques
    feature_importance = visualize_feature_importance(
        final_model, X_train.columns
    )
    print("Importance des caractéristiques:")
    print(feature_importance.sort_values(ascending=False))

    # Créer la soumission
    submission = create_submission(X_test, final_model)

    # Sauvegarder le modèle
    joblib.dump(final_model, "models/cox_model_final.pkl")
    print("Modèle final sauvegardé: models/cox_model_final.pkl")

    # Bonus: Entraîner un modèle d'ensemble
    ensemble, ensemble_score = train_ensemble_model(
        X_train, y_train, X_val, y_val
    )

    # Sauvegarder l'ensemble
    joblib.dump(ensemble, "models/ensemble_model.pkl")
    print("Modèle d'ensemble sauvegardé: models/ensemble_model.pkl")

    # Créer une soumission d'ensemble
    cox_pred_test = ensemble["cox_model"].predict(X_test)
    rsf_pred_test = ensemble["rsf_model"].predict(X_test)
    ensemble_pred_test = (cox_pred_test + rsf_pred_test) / 2

    ensemble_submission = pd.DataFrame(
        {"ID": X_test.index, "risk_score": ensemble_pred_test}
    )
    ensemble_submission.to_csv(
        "submissions/ensemble_submission.csv", index=False
    )
    print(
        "Fichier de soumission d'ensemble généré: submissions/ensemble_submission.csv"
    )

    print("Traitement terminé avec succès!")


if __name__ == "__main__":
    main()
