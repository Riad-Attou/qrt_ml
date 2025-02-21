import numpy as np
import pandas as pd


# Préparation du tableau des données utilisé pour remplir les valeurs manquantes
def preparation_df(df):
    """
    Renvoie le tabeleau des données sans les colonnes ID, CENTER, CYTOGENETICS
    """
    df_copy = df.copy()
    data = np.array(df_copy)
    # Extraire les données numériques
    data = data[:, 2:-1]
    data = data.astype(float)
    return data


def preparation_df_cyto(df):
    # Remplir les données manquantes dans les colonnes numériques
    data = clean_df(df)
    df_copy = df.copy()
    df_array = np.array(df_copy)
    # Ajouter les colonnes supprimées ( ID, CENTER, CYTOGENETICS)
    data = np.concatenate(
        (
            df_array[:, 0].reshape(-1, 1),
            df_array[:, 1].reshape(-1, 1),
            data,
            df_array[:, -1].reshape(-1, 1),
        ),
        axis=1,
    )

    # Si le df est celui des données d'entrainement ( on supprime les nan )
    data2 = data[:, 2:-1]
    data2 = data2.astype(float)
    data = data[~np.isnan(data2).any(axis=1)]

    # # Sinon Remplacer les nan restants par la moyenne ou "Normal"
    # for indice_colonne in range(2, 8):
    #     mean_value = np.nanmean(data[:, indice_colonne].astype(float))
    #     data[:, indice_colonne] = np.where(
    #         np.isnan(data[:, indice_colonne].astype(float)),
    #         mean_value,
    #         data[:, indice_colonne],
    #     )
    # Extraire les colonnes ID, CENTER et CYNOGENETICS
    colones_id_center = data[:, 0:2]
    # Supprimer ces colonnes
    data = data[:, 2:]
    return data, colones_id_center


# Supprimer les nan, le patient à traiter, les colonnes non utilisées
def modif_data(i, data):
    """
    Renvoie le tableau des données sans liges avec nan, sans la ligne
    d'indice i et sans les colonnes avec nan dans la ligne i
    """
    # Supprimer les lignes avec des nan
    data_sans_nan = data[~np.any(np.isnan(data), axis=1)]
    # Supprimer les colonnes avec des nan dans la ligne i
    colonnes_a_supprimer = np.isnan(data[i])
    data_modif = data_sans_nan[:, ~colonnes_a_supprimer]
    return data_modif


def modif_data_cyto(i, data):
    # Supprimer la ligne du patient i
    data_sans_patient = np.delete(data, i, axis=0)
    # Supprimer la derniere colonne CYTOGENETICS qu'on cherche à remplir
    modif_data = data_sans_patient[:, :-1]
    return modif_data


# Recherche du patient qui représente les données les plus proche de celles du patient à traiter
def index_distance_min(i, data_modif, data_bis):
    """
    Renvoie l'indice du patient avec les données les plus proches
    de celles du patient d'indice i
    """
    # Calculer toutes les distances entre le patient i et es autres patients
    distances = np.sum(
        (data_modif - data_bis[i]) ** 2,
        axis=1,
    )
    # Indice du patient le plus proche du patient i
    indice = np.argmin(distances)
    return indice


# Retrouver l'indice du patient dans le df avant modification
def retrouver_indice_original(ligne, data_bis):
    """
    Renvoie l'indice du patient correspondant à la ligne (ligne) dans
    le tableau des données sans nan dans le dataframe
    """
    for i, ligne_orig in enumerate(data_bis):
        if np.array_equal(ligne, ligne_orig):
            return i
    return -1


# Completer le tableau des données avec les données manquantes
def completer_data(indices_nan, i, data_modif, data):
    """
    Remplit les valeurs manquantes pour le patient d'indice i
    """
    colonnes_a_supprimer = np.isnan(data[i])
    data_bis = data[:, ~colonnes_a_supprimer]
    indice = index_distance_min(i, data_modif, data_bis)
    ancien_indice = retrouver_indice_original(data_modif[indice], data_bis)
    for j in indices_nan:
        data[i, j] = data[ancien_indice, j]


def completer_data_cyto(i, modif_data, data):
    data_bis = data[:, :-1]
    indice = index_distance_min(i, modif_data, data_bis)
    ancien_indice = retrouver_indice_original(modif_data[indice], data_bis)
    data[i, -1] = data[ancien_indice, -1]


# Traitemet des données numériques
def clean_df(df):
    """
    Renvoie le tableau des données remplit
    """
    data = preparation_df(df)
    for i in range(len(data)):
        if not np.all(np.isnan(data[i])):
            data_modif = modif_data(i, data)
            indices_nan = np.where(np.isnan(data[i]))[0]
            completer_data(indices_nan, i, data_modif, data)
    cleaned_data = data
    return cleaned_data


# Traitement de la donnée cytogenetics
def clean_df_cyto(df):
    data = preparation_df_cyto(df)[0]
    for i in range(len(data)):
        if not data[i, -1] or str(data[i, -1]).lower() == "nan":
            modif_data = modif_data_cyto(i, data)
            completer_data_cyto(i, modif_data, data)
    cleaned_data = data
    return cleaned_data


# Création du noveau dataframe sans nan
def traitement_df():
    df = pd.read_csv("databases/X_train/clinical_train.csv")
    # df = pd.read_csv("databases/X_test/clinical_test.csv")
    cleaned_data = clean_df_cyto(df)
    colones_id_center = preparation_df_cyto(df)[1]
    new_df = pd.DataFrame(
        cleaned_data,
        columns=[
            "BM_BLAST",
            "WBC",
            "ANC",
            "MONOCYTES",
            "HB",
            "PLT",
            "CYTOGENETICS",
        ],
    )
    new_df["CYTOGENETICS"] = new_df["CYTOGENETICS"].fillna("Normal")
    new_df.insert(0, "ID", colones_id_center[:, 0])
    new_df.insert(1, "CENTER", colones_id_center[:, 1])
    return new_df


# Classification des cytogenetics
def classe_cyto(df):
    """
    Renvoie une liste avec les classes auquelles appartiennent chaque patient
    en fct de son cytogenetic
    """
    df_array = np.array(df)
    cytogenetics = np.unique(df_array[:, 8])
    liste_cyto = df_array[:, 8]
    classes_cyto = np.zeros(len(liste_cyto))
    for i in range(len(liste_cyto)):
        for j in range(len(cytogenetics)):
            if liste_cyto[i] == cytogenetics[j]:
                classes_cyto[i] = j + 1
                break
    return classes_cyto


def main():
    new_df = traitement_df()
    classes_cyto = classe_cyto(new_df)
    new_df.insert(len(new_df.columns), "CLASSE_CYT", classes_cyto)
    new_df.to_csv("new_df_train.csv", index=False)
    # new_df.to_csv("new_df_test.csv", index=False)


main()
