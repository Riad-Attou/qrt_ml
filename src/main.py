from database import Database

if __name__ == "__main__":
    nb_classes = 25
    max_depth = 5

    db = Database()
    df_train, df_eval, mol_df, mol_eval, target_df = db.load_data()
    db.extract_mutations(mol_df)
    db.split_patients_by_os_years(target_df, nb_classes, True)
    db.classify_mutations(max_depth)

    # print(db.mutations_classes)

    print(db.classify_mutation_tuples(mol_eval, nb_classes))
    print(len(db.classify_mutation_tuples_with_score(mol_eval, nb_classes)))
