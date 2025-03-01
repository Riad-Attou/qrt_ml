from data.data_preparation import traitement_donnees
from evaluation.evaluator import Evaluator
from models.lightgbm_model import LightGBMModel

# Paramètres
nb_classes = 3
max_depth = 3
max_depth_cyto = 2

# Préparation des données
merged_train, merged_test, target_df = traitement_donnees(
    nb_classes, max_depth, max_depth_cyto
)

# Définir les features à utiliser
features = [...]  # Liste des colonnes sélectionnées

# Séparer X et y pour l'entraînement
X_train = merged_train[features]
y_train = target_df.dropna(
    subset=["OS_YEARS", "OS_STATUS"]
)  # ou autre préparation nécessaire

# Instanciation et entraînement du modèle LightGBM
lgbm_model = LightGBMModel(
    params={"max_depth": 3, "learning_rate": 0.038, "verbose": -1}
)
lgbm_model.train(
    X_train, y_train["OS_YEARS"]
)  # Ajuster en fonction de ta cible

# Prédictions et sauvegarde
predictions = lgbm_model.predict(merged_test[features])
# Sauvegarder ou traiter les prédictions ...

# Évaluation (sur train ou si tu as un split)
evaluator = Evaluator(tau=7)
ci = evaluator.evaluate(y_train, predictions)
print(f"Concordance Index IPCW: {ci:.4f}")
