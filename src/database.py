import itertools

import pandas as pd

from mutation import Mutation
from patient import Patient


class Database:
    def __init__(
        self,
    ):
        self.patients = []
        self.mutations = []
        # Définir autres éléments des .csv.
        self.classes = []  # Classes des patients

    def split_patients_by_os_years(
        self,
        df: pd.DataFrame,
        nb_classes: int,
        ignore_os_status_zero: bool = True,
    ) -> list[list[str]]:
        """
        Trie les patients selon leur OS_YEARS et divise le résultat en nb_classes classes de taille (presque) égale.

        Parameters:
        - df : DataFrame contenant au moins les colonnes 'ID', 'OS_YEARS' et 'OS_STATUS'
        - nb_classes : nombre de classes souhaitées
        - ignore_os_status_zero : si True, ignore les lignes où OS_STATUS vaut 0

        Returns:
        - Une liste de listes, chaque sous-liste contenant les IDs des patients de la classe correspondante.
        """
        # Filtrer les lignes avec OS_STATUS == 0 si demandé
        if ignore_os_status_zero:
            df = df[df["OS_STATUS"] != 0]

        # Trier le DataFrame selon OS_YEARS (ordre croissant)
        df_sorted = df.sort_values(by="OS_YEARS")

        # Extraire la liste des IDs triée
        patient_ids = df_sorted["ID"].tolist()

        # Diviser la liste en nb_classes classes de taille égale (ou presque)
        n = len(patient_ids)
        classes = []

        # Calculer la taille de base de chaque classe et le reste
        base_size = n // nb_classes
        remainder = n % nb_classes
        start = 0

        for i in range(nb_classes):
            # Répartir le reste pour que les premières classes contiennent un élément de plus
            size = base_size + (1 if i < remainder else 0)
            classes.append(patient_ids[start : start + size])
            start += size

        return classes

    def classify_mutations(self, depth: int):
        assert depth > 0
        classes = self.classes

        # AJOUTER LE COMPTAGE DU NOMBRE D'OCCURENCE D'UNE COMBINAISON DANS UNE CLASSE
        # (OU METTRE DES POIDS POUR GERER LE CAS OU CA APPARAIT DANS PLUSIEURS CLASSES)

        # Génération des tuples (permutations) de taille depth
        combinaisons_tuples = itertools.combinations(self.mutations, depth)
        combinaisons_listes = [list(combo) for combo in combinaisons_tuples]

        for combi in combinaisons_listes:
            for num_classe, classe in enumerate(classes):
                for patient in classe:
                    is_in_same_patient = True
                    for mutation in combi:
                        if mutation.carrier != patient.id:
                            is_in_same_patient = False

                    # Si on trouve la combinaison, on met à jour les interactions des mutations
                    if is_in_same_patient:
                        for mutation in combi:
                            combi_sans_une_mutation_ids = [
                                mut.id for mut in combi if mut != mutation
                            ]
                            mutation.add_interaction(
                                combi_sans_une_mutation_ids, num_classe
                            )

        return

    def load_data(self) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ]:
        # Charger les données cliniques
        df_train = pd.read_csv(
            "databases/X_train/new_df_train.csv", sep=",", quotechar='"'
        )
        df_train["ID"] = df_train["ID"].astype(str)
        df_eval = pd.read_csv("databases/X_test/new_df_test.csv")
        df_eval["ID"] = df_eval["ID"].astype(str)

        # Charger les données moléculaires
        mol_df = pd.read_csv(
            "databases/X_train/molecular_train.csv", sep=",", quotechar='"'
        )
        mol_df["ID"] = mol_df["ID"].astype(str)
        mol_eval = pd.read_csv(
            "databases/X_test/molecular_test.csv", sep=",", quotechar='"'
        )
        mol_eval["ID"] = mol_eval["ID"].astype(str)

        # Charger les données des prédictions
        target_df = pd.read_csv("databases/X_train/target_train.csv")
        target_df = target_df[target_df["ID"].isin(df_train["ID"])]
        target_df["OS_YEARS"] = pd.to_numeric(
            target_df["OS_YEARS"], errors="coerce"
        )
        target_df["OS_STATUS"] = target_df["OS_STATUS"].astype(bool)

        self.patients = [
            Patient(
                list(target_df["ID"])[i],
                False,
                list(target_df["OS_STATUS"])[i] == 1,
                list(target_df["OS_YEARS"])[i],
            )
            for i in range(len(target_df["ID"]))
        ]

        return df_train, df_eval, mol_df, mol_eval, target_df

    def extract_mutations(self, df: pd.DataFrame) -> list:
        """
        Extrait des instances de Mutation pour une liste de gènes clés.
        Pour chaque ligne du DataFrame correspondant à un gène clé,
        une instance de Mutation est créée en extrayant notamment le VAF, l'effet, etc.
        """
        genes = [
            "NPM1",
            "FLT3",
            "GATA2",
            "EZH2",
            "CREBBP",
            "TP53",
            "ASXL1",
            "ZRSR2",
            "SF3B1",
            "TET2",
            "DNMT3A",
            "BCOR",
            "RUNX1",
            "STAG2",
            "ABL1",
            "NFE2",
            "DDX41",
            "CSNK1A1",
            "SH2B3",
        ]
        # Itérer sur chaque ligne du DataFrame
        for _, row in df.iterrows():
            # Comparaison en majuscules pour être insensible à la casse
            if row["GENE"].upper() in genes:
                mutation = Mutation(
                    carrier=row["ID"],
                    gene=row["GENE"],
                    vaf=row["VAF"],
                    effect=row["EFFECT"],
                    # Aajouter d'autres attributs ici, par exemple :
                    # depth=row["DEPTH"],
                    # protein_change=row["PROTEIN_CHANGE"],
                    # etc.
                )
                self.mutations.append(mutation)
        return self.mutations
