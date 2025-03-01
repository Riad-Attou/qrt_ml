from typing import Tuple

import pandas as pd

from database import Database
from features.clinical_features import create_features_clinical_df
from features.merge_features import fusion_df, trad_ris_int
from features.molecular_features import (
    classes_mutations,
    create_features_mol_df,
)


def traitement_donnees(
    nb_classes: int, max_depth: int, max_depth_cyto: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    db = Database()
    df_train, df_test, mol_df, mol_eval, target_df = db.load_data()

    db.extract_mutations(mol_df)
    db.split_patients_by_os_years(target_df, nb_classes, True)
    db.classify_mutations(max_depth)
    db.classify_mutation_tuples_with_score(mol_df, max_depth)

    db.extract_cytogene(df_train)
    db.classify_cytogene(max_depth_cyto)
    db.classify_cytogene_tuples_with_score(df_train, max_depth_cyto)

    mutations = classes_mutations(nb_classes, target_df, df_train, mol_df)

    df_train_enrichi = create_features_clinical_df(df_train)
    mol_df_enrichi = create_features_mol_df(mol_df, db, mutations)
    merged_train = fusion_df(
        df_train_enrichi, mol_df_enrichi, db, mutations, target_df
    )

    db.classify_mutation_tuples_with_score(mol_eval, max_depth)
    db.classify_cytogene_tuples_with_score(df_test, max_depth_cyto)

    df_eval_enrichi = create_features_clinical_df(df_test)
    mol_eval_enrichi = create_features_mol_df(mol_eval, db, mutations)
    merged_test = fusion_df(
        df_eval_enrichi, mol_eval_enrichi, db, mutations, target_df
    )

    trad_ris_int(merged_train)
    trad_ris_int(merged_test)

    return merged_train, merged_test, target_df
