from typing import Tuple

import pandas as pd

from data.data_loader import DataLoader
from data.data_splitter import DataSplitter
from features.clinical_features import create_features_clinical_df
from features.cytogene_classifier import (
    classify_cytogene_tuples_with_score,
    extract_cytogenes,
)
from features.merge_features import fusion_df, trad_ris_int
from features.molecular_features import (
    classes_mutations,
    create_features_mol_df,
)
from features.mutation_classifier import (
    classify_mutation_tuples_with_score,
    extract_mutations,
)


def traitement_donnees(
    nb_classes: int, max_depth: int, max_depth_cyto: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # 1. Chargement des données via DataLoader
    loader = DataLoader(
        clinical_path_train="databases/X_train/new_df_train.csv",
        clinical_path_test="databases/X_test/new_df_test.csv",
        molecular_path_train="databases/X_train/molecular_train.csv",
        molecular_path_test="databases/X_test/molecular_test.csv",
        target_path_train="databases/X_train/target_train.csv",
    )
    df_train, df_test = loader.load_clinical_data()
    mol_train, mol_test = loader.load_molecular_data()
    target_df = loader.load_target_data()

    # Filtrer target_df pour ne garder que les patients présents dans df_train
    target_df = target_df[target_df["ID"].isin(df_train["ID"])]

    # 2. Découpage des patients à l'aide de DataSplitter
    splitter = DataSplitter(nb_classes, ignore_os_status_zero=True)
    # Ici, on utilise target_df pour obtenir un partitionnement (liste de listes d'ID)
    patient_classes = splitter.split_by_distribution(
        target_df, distribution="lognorm"
    )
    # Construire un mapping patient -> classe
    patient_to_class = {}
    for class_idx, id_list in enumerate(patient_classes):
        for pid in id_list:
            patient_to_class[pid] = class_idx

    # 3. Extraction et classification des mutations
    # Extraire les mutations depuis le DataFrame moléculaire d'entraînement
    mutations_by_patient = extract_mutations(mol_train)
    # Calculer les interactions globales et obtenir un score par patient
    mutation_scores = classify_mutation_tuples_with_score(
        mutations_by_patient,
        patient_to_class,
        global_mutation_interactions={},
        max_depth=max_depth,
        k=5,
    )

    # 4. Extraction et classification des cytogènes (similaire à la partie mutations)
    cytogenes_by_patient = extract_cytogenes(df_train)
    cytogene_scores = classify_cytogene_tuples_with_score(
        df_test,
        global_cytogene_interactions={},  # Par exemple, un dictionnaire vide si tu ne l'as pas encore construit
        global_classes=patient_classes,  # Ou toute liste représentant les classes obtenues du split
        max_depth=max_depth_cyto,
        k=5,
    )

    # 5. Génération de la liste des ensembles de mutations par classe
    mutations = classes_mutations(nb_classes, target_df, df_train, mol_train)

    # 6. Création des features cliniques et moléculaires
    df_train_enrichi = create_features_clinical_df(df_train)
    df_test_enrichi = create_features_clinical_df(df_test)

    # Pour les données moléculaires, on ne passe plus l'ancienne Database, on peut passer None
    mol_train_enrichi = create_features_mol_df(mol_train, None, mutations)
    mol_test_enrichi = create_features_mol_df(mol_test, None, mutations)

    # 7. Fusion des données cliniques et moléculaires
    merged_train = fusion_df(
        df_train_enrichi, mol_train_enrichi, None, mutations, target_df
    )
    merged_test = fusion_df(
        df_test_enrichi, mol_test_enrichi, None, mutations, target_df
    )

    # 8. Conversion de certaines colonnes en numérique
    trad_ris_int(merged_train)
    trad_ris_int(merged_test)

    return merged_train, merged_test, target_df
