import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.model_selection import train_test_split
from sksurv.metrics import concordance_index_ipcw
from sksurv.util import Surv

#############################################
# 1. Définition des fonctions de traitement
#############################################


def classify_cytogenetic_risk(cyto_str):
    """
    Classifie le risque cytogénétique en se basant sur des motifs connus.
    Renvoie "favorable", "intermediaire" ou "defavorable".
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
    return "intermediaire"


def extract_features(cyto_str):
    """
    Extrait plusieurs caractéristiques à partir de la chaîne cytogénétique.
    Renvoie notamment :
      - nb_del : nombre de délétion
      - nb_t : nombre de translocations hors t(9;22)
      - nb_philadelphia : 1 si présence de t(9;22), 0 sinon
      - nb_dup : nombre de duplications
      - nb_inv : nombre d'inversions
      - complex_karyotype : 1 si la chaîne contient "complex", 0 sinon
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


#############################################
# 2. Chargement et préparation des données cliniques
#############################################

# Charger le fichier clinique nettoyé (sans OS_YEARS ni OS_STATUS)
clinical_df = pd.read_csv(
    "databases/X_train/cleaned_clinical_train.csv", sep=",", quotechar='"'
)
clinical_df["ID"] = clinical_df["ID"].astype(str)

# Extraire les features de CYTOGENETICS
features_df = clinical_df["CYTOGENETICS"].apply(extract_features)
clinical_df = pd.concat([clinical_df, features_df], axis=1)

# Ajouter la classification du risque cytogénétique et créer des dummies
clinical_df["cyto_risk"] = clinical_df["CYTOGENETICS"].apply(
    classify_cytogenetic_risk
)
cyto_risk_dummies = pd.get_dummies(
    clinical_df["cyto_risk"], prefix="cyto_risk"
)
clinical_df = pd.concat([clinical_df, cyto_risk_dummies], axis=1)

#############################################
# 3. Chargement et filtrage des données cibles (target)
#############################################

# Charger le fichier target_train qui contient OS_YEARS et OS_STATUS
target_df = pd.read_csv(
    "databases/X_train/target_train.csv", sep=",", quotechar='"'
)
target_df["ID"] = target_df["ID"].astype(str)
target_df["OS_YEARS"] = pd.to_numeric(target_df["OS_YEARS"], errors="coerce")
target_df["OS_STATUS"] = pd.to_numeric(target_df["OS_STATUS"], errors="coerce")
target_df = target_df.dropna(subset=["OS_YEARS", "OS_STATUS"])

#############################################
# 4. Fusion des données cliniques et cibles
#############################################

merged_df = pd.merge(target_df, clinical_df, on="ID", how="left")
merged_df["OS_YEARS"] = pd.to_numeric(merged_df["OS_YEARS"], errors="coerce")
merged_df["OS_STATUS"] = pd.to_numeric(merged_df["OS_STATUS"], errors="coerce")
merged_df = merged_df.dropna(subset=["OS_YEARS", "OS_STATUS"])

#############################################
# 5. Définition des features et de la cible pour le modèle
#############################################

feature_cols = [
    "nb_del",
    "nb_t",
    "nb_philadelphia",
    "nb_dup",
    "nb_inv",
    "complex_karyotype",
    # "cyto_risk_favorable", # Variance faible.
    "cyto_risk_intermediaire",
    "cyto_risk_defavorable",
]

# Convertir en numérique au cas où
for col in feature_cols:
    merged_df[col] = pd.to_numeric(merged_df[col], errors="coerce")

# Créer un DataFrame pour le modèle (sans la colonne "ID")
df_model = merged_df[["OS_YEARS", "OS_STATUS"] + feature_cols].copy()
df_model.fillna(0, inplace=True)

#############################################
# 6. Séparation en ensembles d'entraînement et de test
#############################################

train_df, test_df = train_test_split(df_model, test_size=0.2, random_state=42)

#############################################
# 7. Entraînement du modèle de Cox avec lifelines
#############################################

cph = CoxPHFitter()
cph.fit(train_df, duration_col="OS_YEARS", event_col="OS_STATUS")
cph.print_summary()
print("C-index (entraînement) :", cph.concordance_index_)

#############################################
# 8. Prédiction de la fonction de survie et extraction de la médiane
#############################################

surv_funcs = cph.predict_survival_function(test_df)


def get_median_survival(s):
    below_half = s[s <= 0.5]
    if below_half.empty:
        return np.nan
    return below_half.index[0]


median_survival = surv_funcs.apply(get_median_survival, axis=0)

# Préparer le DataFrame de prédictions en récupérant les IDs du DataFrame fusionné initial
test_ids = merged_df.loc[test_df.index, "ID"]
pred_df = pd.DataFrame({"ID": test_ids, "predicted_OS_YEARS": median_survival})

print("Aperçu des prédictions de survie (médiane) :")
print(pred_df.head())
pred_df.to_csv("predicted_survival_times_cox_clinical.csv", index=False)

#############################################
# 9. Calcul de l'IPCW-C-index
#############################################

from sksurv.metrics import concordance_index_ipcw
from sksurv.util import Surv

# Créer les tableaux structurés pour l'ensemble d'entraînement et de test
y_train_struct = Surv.from_arrays(
    event=train_df["OS_STATUS"].astype(bool), time=train_df["OS_YEARS"]
)
y_test_struct = Surv.from_arrays(
    event=test_df["OS_STATUS"].astype(bool), time=test_df["OS_YEARS"]
)

# Prédire les scores de risque (partiels) pour le jeu de test
# Ici, nous utilisons predict_partial_hazard qui renvoie un score de risque
risk_scores_test = cph.predict_partial_hazard(test_df)

# Calculer l'IPCW-C-index, ici en fixant tau à 7 (ajustez tau selon vos données)
ipcw_c_index = concordance_index_ipcw(
    y_train_struct, y_test_struct, risk_scores_test, tau=7
)[0]
print("IPCW-C-index :", ipcw_c_index)
