import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw
from sksurv.util import Surv

#############################################
# 1. Traitement des données cliniques
#############################################


def classify_cytogenetic_risk(cyto_str):
    """
    Classifie le risque cytogénétique en se basant sur des motifs connus.
    Renvoie une chaîne ("favorable", "intermediaire", "defavorable").
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
    # Sinon, c'est plutôt intermediaire (par exemple, caryotype normal)
    return "intermediaire"


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
    "databases/X_train/cleaned_clinical_train.csv", sep=",", quotechar='"'
)
clinical_df["ID"] = clinical_df["ID"].astype(str)
# Extraire les features existantes à partir de CYTOGENETICS
features_df = clinical_df["CYTOGENETICS"].apply(extract_features)
clinical_df = pd.concat([clinical_df, features_df], axis=1)
# Ajouter la classification du risque cytogénétique
clinical_df["cyto_risk"] = clinical_df["CYTOGENETICS"].apply(
    classify_cytogenetic_risk
)
# Créer des variables dummies à partir de la classification
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
total_mutations = mol_df.groupby("ID").size().rename("total_mutations")
mean_vaf = mol_df.groupby("ID")["VAF"].mean().rename("mean_VAF")
stop_gained = (
    mol_df[mol_df["EFFECT"] == "stop_gained"]
    .groupby("ID")
    .size()
    .rename("stop_gained_count")
)

# Nouvelles agrégations pour affiner :
frameshift_variant = (
    mol_df[mol_df["EFFECT"].str.contains("frameshift", na=False)]
    .groupby("ID")
    .size()
    .rename("frameshift_variant_count")
)
non_synonymous = (
    mol_df[mol_df["EFFECT"].str.contains("non_synonymous", na=False)]
    .groupby("ID")
    .size()
    .rename("non_synonymous_count")
)
mean_depth = mol_df.groupby("ID")["DEPTH"].mean().rename("mean_depth")

# Agrégation pondérée par VAF pour stop_gained (exemple)
weighted_stop_gained = (
    mol_df[mol_df["EFFECT"] == "stop_gained"]
    .groupby("ID")["VAF"]
    .sum()
    .rename("weighted_stop_gained")
)
weighted_frameshift = (
    mol_df[mol_df["EFFECT"].str.contains("frameshift", na=False)]
    .groupby("ID")["VAF"]
    .sum()
    .rename("weighted_frameshift")
)
weighted_non_synonymous = (
    mol_df[mol_df["EFFECT"].str.contains("non_synonymous", na=False)]
    .groupby("ID")["VAF"]
    .sum()
    .rename("weighted_non_synonymous")
)

# Exemple d'indicateur pour des gènes clés (ici NPM1 et FLT3)
npm1_mut = (
    mol_df[mol_df["GENE"].str.upper() == "NPM1"]
    .groupby("ID")
    .size()
    .rename("npm1_mutated")
)
flt3_mut = (
    mol_df[mol_df["GENE"].str.upper() == "FLT3"]
    .groupby("ID")
    .size()
    .rename("flt3_mutated")
)

# Combiner toutes les agrégations dans un DataFrame
mol_agg = pd.concat(
    [
        total_mutations,
        mean_vaf,
        stop_gained,
        frameshift_variant,
        non_synonymous,
        mean_depth,
        weighted_stop_gained,
        weighted_frameshift,
        weighted_non_synonymous,
        npm1_mut,
        flt3_mut,
    ],
    axis=1,
).reset_index()


# # Imputer avec la médiane pour les colonnes continues, par exemple
# for col in ["mean_VAF", "mean_depth"]:
#     mol_agg[col] = mol_agg[col].fillna(mol_agg[col].median())
# # Pour les comptes, utiliser la médiane également
# for col in [
#     "total_mutations",
#     "stop_gained_count",
#     "frameshift_variant_count",
#     "non_synonymous_count",
# ]:
#     mol_agg[col] = mol_agg[col].fillna(mol_agg[col].median())
# # Pour les indicateurs pondérés, on peut utiliser la médiane ou 0 si la distribution est particulière
# for col in [
#     "weighted_stop_gained",
#     "weighted_frameshift",
#     "weighted_non_synonymous",
# ]:
#     mol_agg[col] = mol_agg[col].fillna(mol_agg[col].median())
# # Pour les indicateurs binaires npm1_mutated et flt3_mutated, remplacer NaN par 0
# mol_agg["npm1_mutated"] = mol_agg["npm1_mutated"].fillna(0)
# mol_agg["flt3_mutated"] = mol_agg["flt3_mutated"].fillna(0)

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
            "cyto_risk_intermediaire",
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


# Traduire les risques en int.
for col in [
    "cyto_risk_favorable",
    "cyto_risk_intermediaire",
    "cyto_risk_defavorable",
]:
    merged_df[col] = pd.to_numeric(merged_df[col], errors="coerce")


print("Aperçu du DataFrame fusionné (merged_df) :")
print(merged_df.head())
print("Nombre de lignes :", len(merged_df))
print("NaN dans OS_YEARS :", merged_df["OS_YEARS"].isnull().sum())

# Supprimer les lignes où OS_YEARS est manquant
merged_df = merged_df.dropna(subset=["OS_YEARS"])
print("Nombre de lignes après nettoyage :", len(merged_df))


# S'assurer que OS_YEARS est numérique et supprimer les lignes manquantes
merged_df["OS_YEARS"] = pd.to_numeric(merged_df["OS_YEARS"], errors="coerce")
merged_df = merged_df.dropna(subset=["OS_YEARS"])

# Définir la liste des features (cliniques et moléculaires) issues de votre pipeline de feature engineering
feature_cols = [
    "nb_del",
    "nb_t",
    "nb_philadelphia",
    "nb_dup",
    "nb_inv",
    "complex_karyotype",
    "cyto_risk_favorable",
    "cyto_risk_intermediaire",
    "cyto_risk_defavorable",
    "total_mutations",
    "mean_VAF",
    "stop_gained_count",
    "frameshift_variant_count",
    "non_synonymous_count",
    "mean_depth",
    "weighted_stop_gained",
    "weighted_frameshift",
    "weighted_non_synonymous",
    "npm1_mutated",
    "flt3_mutated",
]

# Si certaines colonnes (issues de get_dummies) sont de type object, les convertir en numérique
for col in [
    "cyto_risk_favorable",
    "cyto_risk_intermediaire",
    "cyto_risk_defavorable",
]:
    merged_df[col] = pd.to_numeric(merged_df[col], errors="coerce")

# Extraire les features et définir la cible
X = merged_df[feature_cols].copy()
# Les cibles : temps de survie (OS_YEARS) et indicateur d'événement (OS_STATUS)
y_time = merged_df["OS_YEARS"].values
y_event = merged_df["OS_STATUS"].values.astype(bool)

# Créer le tableau structuré requis par scikit-survival
y_structured = Surv.from_arrays(event=y_event, time=y_time)

IDs = merged_df["ID"]

#############################################
# 2. Séparation en ensembles d'entraînement et de test
#############################################

# Split utilisant le tableau structuré
X_train, X_test, y_train, y_test, IDs_train, IDs_test = train_test_split(
    X, y_structured, IDs, test_size=0.2, random_state=42
)

#############################################
# 3. Tuning et entraînement du modèle RSF
#############################################

# Instancier un Random Survival Forest
rsf = RandomSurvivalForest(random_state=42, n_jobs=-1)

# Définir une grille d'hyperparamètres à explorer
param_grid = {
    "n_estimators": [200],
    "max_depth": [3],
    "min_samples_split": [2],
    "min_samples_leaf": [1],
    "max_features": [0.8],
}


# Définir une fonction de score qui utilise le c-index censuré
def rsf_score(estimator, X, y):
    # Prédire des scores de risque; on prend l'opposé pour que des scores plus élevés indiquent un risque plus grand
    risk_scores = estimator.predict(X)
    return concordance_index_censored(y["event"], y["time"], risk_scores)[0]


# GridSearchCV avec 5-fold cross-validation
grid_search = GridSearchCV(
    rsf,
    param_grid,
    cv=5,
    scoring=rsf_score,
    n_jobs=-1,
    verbose=2,  # Augmentez la verbosité (2 affiche plus d'informations)
)
grid_search.fit(X_train, y_train)

print("Meilleurs hyperparamètres :", grid_search.best_params_)
print("Meilleur score (c-index censuré) :", grid_search.best_score_)

# Entraîner le meilleur modèle sur l'ensemble d'entraînement
best_rsf = grid_search.best_estimator_
best_rsf.fit(X_train, y_train)

#############################################
# 4. Évaluation du modèle RSF
#############################################

# Prédire des scores de risque pour l'ensemble test
# Ici, un score plus élevé (après inversion) signifie un risque plus grand
risk_scores_test = best_rsf.predict(X_test)
# Calculer le c-index classique
c_index_classic = concordance_index_censored(
    y_test["event"], y_test["time"], risk_scores_test
)[0]
print("Concordance index classique (RSF) :", c_index_classic)

# Calcul de l'IPCW-C-index (avec tau fixé à 7, à ajuster selon votre contexte)
c_index_ipcw = concordance_index_ipcw(
    y_train, y_test, risk_scores_test, tau=7
)[0]
print("IPCW-C-index (RSF) :", c_index_ipcw)

#############################################
# 5. (Optionnel) Prédire des durées de survie
#############################################
# Ici, il est plus difficile d'obtenir directement une "durée de survie" à partir d'un RSF.
# Une approche consiste à estimer pour chaque patient une fonction de survie et en extraire par exemple la médiane.
# RSF de scikit-survival offre la méthode predict_survival_function.
surv_funcs = best_rsf.predict_survival_function(X_test)


def get_median_survival(s):
    # Utilise la propriété "domain" de l'objet StepFunction pour obtenir le minimum et maximum
    t_min, t_max = s.domain
    # Créer une grille de temps sur le domaine de s
    times = np.linspace(t_min, t_max, 1000)
    surv_values = s(times)
    below_half = times[surv_values <= 0.5]
    if below_half.size > 0:
        return below_half[0]
    else:
        return np.nan


median_survival = [get_median_survival(s) for s in surv_funcs]
df_pred_surv = pd.DataFrame(
    {
        "ID": IDs_test.values,  # Utilise les véritables IDs du jeu de test
        "predicted_OS_YEARS": median_survival,
    }
)
print("Aperçu des prédictions de survie (durée) :")
print(df_pred_surv.head())
