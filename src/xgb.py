import itertools

import numpy as np
import pandas as pd
import xgboost as xgb
from lifelines import CoxPHFitter
from sklearn.model_selection import train_test_split
from sksurv.metrics import concordance_index_ipcw
from sksurv.util import Surv

#############################################
# 1. Traitement des données cliniques
#############################################


def classify_cytogenetic_risk(cyto_str):
    """
    Classifie le risque cytogénétique en se basant sur des motifs connus.
    Renvoie une chaîne ("favorable", "intermédiaire", "defavorable").
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
    ):
        return "defavorable"
    # Sinon, c'est plutôt intermédiaire (par exemple un caryotype normal)
    return "intermédiaire"


def extract_features(cyto_str):
    """
    Extrait plusieurs caractéristiques à partir de la chaîne cytogénétique.
    On distingue notamment :
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
    nb_t = nb_t_total - nb_philadelphia
    if nb_t < 0:
        nb_t = 0
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


# Charger les données cliniques
clinical_df = pd.read_csv(
    "databases/X_train/clinical_train.csv", sep=",", quotechar='"'
)
clinical_df["ID"] = clinical_df["ID"].astype(str)
# Extraire les features existantes à partir de CYTOGENETICS
features_df = clinical_df["CYTOGENETICS"].apply(extract_features)
clinical_df = pd.concat([clinical_df, features_df], axis=1)
# Ajouter la classification du risque cytogénétique
clinical_df["cyto_risk"] = clinical_df["CYTOGENETICS"].apply(
    classify_cytogenetic_risk
)

cyto_risk_dummies = pd.get_dummies(
    clinical_df["cyto_risk"], prefix="cyto_risk"
)
clinical_df = pd.concat([clinical_df, cyto_risk_dummies], axis=1)


#############################################
# 2. Agrégation des données moléculaires
#############################################

# Charger les données moléculaires
mol_df = pd.read_csv(
    "databases/X_train/molecular_train.csv", sep=",", quotechar='"'
)
mol_df["ID"] = mol_df["ID"].astype(str)

# Agrégations existantes :
# a) Nombre total de mutations par patient
total_mutations = mol_df.groupby("ID").size().rename("total_mutations")
# b) Moyenne du VAF par patient
mean_vaf = mol_df.groupby("ID")["VAF"].mean().rename("mean_VAF")
# c) Nombre de mutations avec effet "stop_gained"
stop_gained = (
    mol_df[mol_df["EFFECT"] == "stop_gained"]
    .groupby("ID")
    .size()
    .rename("stop_gained_count")
)

# Nouvelles agrégations pour affiner :
# d) Nombre de mutations avec effet "frameshift_variant"
frameshift_variant = (
    mol_df[mol_df["EFFECT"].str.contains("frameshift", na=False)]
    .groupby("ID")
    .size()
    .rename("frameshift_variant_count")
)
# e) Nombre de mutations avec effet "non_synonymous_codon"
non_synonymous = (
    mol_df[mol_df["EFFECT"].str.contains("non_synonymous", na=False)]
    .groupby("ID")
    .size()
    .rename("non_synonymous_count")
)
# f) Moyenne de la profondeur de séquençage (DEPTH)
mean_depth = mol_df.groupby("ID")["DEPTH"].mean().rename("mean_depth")

# Combiner toutes les agrégations dans un DataFrame
mol_agg = pd.concat(
    [
        total_mutations,
        mean_vaf,
        stop_gained,
        frameshift_variant,
        non_synonymous,
        mean_depth,
    ],
    axis=1,
).reset_index()
mol_agg.fillna(0, inplace=True)  # Remplacer les NaN par 0

#############################################
# 3. Fusion des données cliniques, moléculaires et cibles
#############################################

# Charger les données cibles
target_df = pd.read_csv(
    "databases/X_train/target_train.csv", sep=",", quotechar='"'
)
target_df["ID"] = target_df["ID"].astype(str)

# Fusionner target et données cliniques (avec features extraites)
merged_df = pd.merge(
    target_df,
    clinical_df[
        [
            "ID",
            "nb_del",
            "nb_t",
            "nb_philadelphia",
            "nb_dup",
            "nb_inv",
            "complex_karyotype",
            "cyto_risk_favorable",
            "cyto_risk_intermédiaire",
            "cyto_risk_defavorable",
        ]
    ],
    on="ID",
    how="left",
)
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

# Fusionner ensuite les agrégations moléculaires
merged_df = pd.merge(merged_df, mol_agg, on="ID", how="left")
merged_df.fillna(
    0, inplace=True
)  # Pour les patients sans données moléculaires

# Vérification du résultat final
print("Aperçu du DataFrame fusionné (merged_df) :")
print(merged_df.head())
print("Nombre de lignes :", len(merged_df))
print("NaN dans OS_YEARS :", merged_df["OS_YEARS"].isnull().sum())

# Supprimer les lignes où OS_YEARS est manquant
merged_df = merged_df.dropna(subset=["OS_YEARS"])
print("Nombre de lignes après nettoyage :", len(merged_df))

data_for_model = merged_df.copy()  # merged_df contient déjà la colonne ID

#############################################
# 4. Préparation pour XGBoost
#############################################

# Définir la liste des features incluant celles basées sur les connaissances avancées
feature_cols = [
    "nb_del",
    "nb_t",
    "nb_philadelphia",
    "nb_dup",
    "nb_inv",
    "complex_karyotype",
    "total_mutations",
    "mean_VAF",
    "stop_gained_count",
    "cyto_risk_favorable",
    "cyto_risk_intermédiaire",
    "cyto_risk_defavorable",
    # Vous pouvez ajouter d'autres features moléculaires si nécessaire
]
X = merged_df[feature_cols]

IDs = data_for_model["ID"]
y_time = pd.to_numeric(merged_df["OS_YEARS"], errors="coerce")
y_event = merged_df["OS_STATUS"]

# Assurez-vous qu'il n'y a pas de NaN dans y_time
data = merged_df.dropna(subset=["OS_YEARS"])
X = data[feature_cols]
y_time = pd.to_numeric(data["OS_YEARS"], errors="coerce")
y_event = data["OS_STATUS"]


# Séparer en train et test en conservant aussi les ID
(
    X_train,
    X_test,
    y_time_train,
    y_time_test,
    y_event_train,
    y_event_test,
    IDs_train,
    IDs_test,
) = train_test_split(X, y_time, y_event, IDs, test_size=0.2, random_state=42)

# Pour XGBoost, on peut utiliser OS_STATUS comme sample weights (1 pour événement, 0 pour censuré)
w_train = y_event_train
w_test = y_event_test

# Créer la DMatrix pour XGBoost
dtrain = xgb.DMatrix(X_train, label=y_time_train, weight=w_train)
dtest = xgb.DMatrix(X_test, label=y_time_test, weight=w_test)

#############################################
# 5. Entraînement du modèle XGBoost pour la survie
#############################################

param_grid = {
    "eta": [0.003],
    "max_depth": [3],
    "min_child_weight": [1],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
}

combinations = list(
    itertools.product(
        param_grid["eta"],
        param_grid["max_depth"],
        param_grid["min_child_weight"],
        param_grid["subsample"],
        param_grid["colsample_bytree"],
    )
)

best_score = float("inf")
best_params = None
best_num_round = None

print("Recherche des meilleurs hyperparamètres...")
for (
    eta,
    max_depth,
    min_child_weight,
    subsample,
    colsample_bytree,
) in combinations:
    params = {
        "objective": "survival:cox",
        "eval_metric": "cox-nloglik",
        "eta": eta,
        "max_depth": max_depth,
        "min_child_weight": min_child_weight,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "seed": 42,
    }
    cv_results = xgb.cv(
        params,
        dtrain,  # dtrain est défini plus haut
        num_boost_round=2500,
        nfold=5,
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    current_score = cv_results["test-cox-nloglik-mean"].min()
    current_round = cv_results["test-cox-nloglik-mean"].argmin()
    print(
        f"Params: {params}, Score: {current_score:.5f} at round {current_round}"
    )
    if current_score < best_score:
        best_score = current_score
        best_params = params.copy()
        best_num_round = current_round

print("\nMeilleurs paramètres trouvés:")
print(best_params)
print("Meilleur score (test cox-nloglik):", best_score)
print("Nombre d'itérations optimal:", best_num_round)

#############################################
# Entraînement final avec les meilleurs hyperparamètres
#############################################
watchlist = [(dtrain, "train"), (dtest, "eval")]
final_model = xgb.train(
    best_params,
    dtrain,
    num_boost_round=best_num_round,
    evals=watchlist,
    early_stopping_rounds=50,
)

model = final_model

#############################################
# 5b. Calibration hybride pour estimer la durée de survie
#############################################

# Utiliser les scores de risque XGBoost comme covariable unique dans un modèle de Cox classique pour calibrer la fonction de survie.

# Récupérer les scores de risque pour l'ensemble d'entraînement
risk_scores_train = model.predict(dtrain)

df_train_cox = pd.DataFrame(
    {
        "risk_score": risk_scores_train,
        "OS_YEARS": y_time_train,
        "OS_STATUS": y_event_train,
    }
)

# Ajuster un modèle de Cox avec "risk_score" comme unique covariable
cph = CoxPHFitter()
cph.fit(df_train_cox, duration_col="OS_YEARS", event_col="OS_STATUS")
cph.print_summary()

# Pour l'ensemble test, créer un DataFrame avec le score de risque
preds = model.predict(
    dtest
)  # Ceci calcule les scores de risque pour l'ensemble test
risk_scores_test = preds  # Vous pouvez utiliser directement preds ou l'appeler risk_scores_test

# Créer le DataFrame pour le modèle de calibration (en utilisant les scores de risque du jeu de test)
df_test_cox = pd.DataFrame(
    {"risk_score": risk_scores_test}, index=X_test.index
)
df_test_cox = pd.DataFrame(
    {"risk_score": risk_scores_test}, index=X_test.index
)

# Prédire la fonction de survie calibrée pour chaque patient du jeu test
surv_funcs = cph.predict_survival_function(df_test_cox)


# Extraire la médiane de survie pour chaque patient
def get_median_survival(s):
    # Retourne le premier temps où la probabilité de survie tombe en dessous de 0.5
    below_half = s[s <= 0.5]
    if below_half.empty:
        return np.nan
    return below_half.index[0]


median_survival = surv_funcs.apply(get_median_survival, axis=0)

# Créer un DataFrame de sortie avec ID et la durée de survie prédite (médiane)
df_pred_surv = pd.DataFrame(
    {
        "ID": IDs_test.values,  # ici on utilise la colonne ID du jeu de test
        "predicted_OS_YEARS": median_survival,
    }
)
print("Aperçu des prédictions de survie (durée) :")
print(df_pred_surv.head())
df_pred_surv.to_csv("predicted_survival_times.csv", index=False)

#############################################
# 6. Calcul de l'IPCW-C-index
#############################################

# Conversion des labels de test en format structuré pour scikit-survival
y_test_struct = Surv.from_arrays(
    event=y_event_test.astype(bool), time=y_time_test
)
y_train_struct = Surv.from_arrays(
    event=y_event_train.astype(bool), time=y_time_train
)

# Comme dans votre convention un score élevé indique un risque élevé, on passe preds directement
c_index_ipcw = concordance_index_ipcw(
    y_train_struct, y_test_struct, preds, tau=7
)[0]
print("IPCW-C-index :", c_index_ipcw)
