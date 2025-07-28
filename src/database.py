import itertools
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import lognorm, weibull_min

from cytogene import Cytogene
from mutation import Mutation
from patient import Patient


class Database:
    def __init__(self):
        self.patients = []  # Liste de Patient().
        self.patient_test_combinations = {}
        self.mutations = []

        self.classes = []  # Classes des patients
        self.mutations_classes = {}

        self.cytogenes = []
        self.cytogene_classes = {}
        self.cyto_test_combinations = {}

    def split_patients_by_os_years(
        self,
        df: pd.DataFrame,
        nb_classes: int,
        ignore_os_status_zero: bool = True,
    ) -> list[list[str]]:
        """
        Trie les patients selon une transformation non linéaire de leur OS_YEARS basée sur un modèle log-normale,
        puis divise le résultat en nb_classes classes de taille (presque) égale.

        Processus :
        1. Filtrer les patients selon OS_STATUS et retirer les valeurs non finies (NaN, inf, -inf) de OS_YEARS.
        2. Estimer les paramètres du modèle log-normale à partir des OS_YEARS en fixant loc=0.
        3. Pour chaque patient, calculer la probabilité cumulative via la CDF de lognorm.
        4. Trier les patients selon cette valeur transformée.
        5. Répartir les patients triés en nb_classes groupes de taille égale.

        Parameters:
        - df : DataFrame contenant au moins les colonnes 'ID', 'OS_YEARS' et 'OS_STATUS'
        - nb_classes : nombre de classes souhaitées
        - ignore_os_status_zero : si True, ignore les lignes où OS_STATUS vaut 0

        Returns:
        - Une liste de listes, chaque sous-liste contenant les IDs des patients de la classe correspondante.

        Références :
        - Documentation SciPy sur lognorm : https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html :contentReference[oaicite:0]{index=0}
        - Exemple d'analyse de survie avec des modèles log-normaux : https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3908600/ :contentReference[oaicite:1]{index=1}
        """
        # Filtrer les lignes avec OS_STATUS == 0 si demandé
        if ignore_os_status_zero:
            df = df[df["OS_STATUS"] != 0]

        # Filtrer les valeurs non finies dans OS_YEARS (NaN, inf, -inf)
        df = df[np.isfinite(df["OS_YEARS"])]

        # Extraire les valeurs de OS_YEARS
        os_years = df["OS_YEARS"].values

        # Estimer les paramètres du modèle log-normale en fixant loc=0
        shape, loc, scale = lognorm.fit(os_years, floc=0)

        # Calculer la CDF de lognorm pour chaque patient
        df = df.copy()  # pour ne pas modifier le DataFrame original
        df["lognorm_cdf"] = lognorm.cdf(os_years, shape, loc=loc, scale=scale)

        # Trier le DataFrame par la valeur transformée (CDF)
        df_sorted = df.sort_values(by="lognorm_cdf")

        # Extraire la liste des IDs triée
        patient_ids = df_sorted["ID"].tolist()

        # Diviser la liste en nb_classes groupes de taille (presque) égale
        n = len(patient_ids)
        classes = []
        base_size = n // nb_classes
        remainder = n % nb_classes
        start = 0
        for i in range(nb_classes):
            size = base_size + (1 if i < remainder else 0)
            classes.append(patient_ids[start : start + size])
            start += size

        self.classes = classes
        return classes

    def split_patients_by_os_years3(
        self,
        df: pd.DataFrame,
        nb_classes: int,
        ignore_os_status_zero: bool = True,
    ) -> list[list[str]]:
        """
        Trie les patients selon une transformation non linéaire de leur OS_YEARS basée sur un modèle de Weibull,
        puis divise le résultat en nb_classes classes de taille (presque) égale.

        On procède de la manière suivante :
        1. On filtre les patients (si demandé) pour exclure ceux avec OS_STATUS == 0.
        2. On estime les paramètres du modèle de Weibull à partir des OS_YEARS.
        3. Pour chaque patient, on calcule la probabilité cumulative via la CDF de Weibull.
        4. On trie les patients selon cette valeur transformée.
        5. On répartit les patients triés en nb_classes groupes égaux.

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

        # Filtrer les valeurs non finies dans OS_YEARS (NaN, inf, -inf)
        df = df[np.isfinite(df["OS_YEARS"])]

        # Extraire les valeurs de OS_YEARS
        os_years = df["OS_YEARS"].values

        # Estimer les paramètres du modèle de Weibull en fixant loc=0 pour se concentrer sur les temps positifs
        shape, loc, scale = weibull_min.fit(os_years, floc=0)

        # Calculer la CDF de Weibull pour chaque patient
        df = df.copy()  # pour éviter de modifier le DataFrame original
        df["weibull_cdf"] = weibull_min.cdf(
            os_years, shape, loc=loc, scale=scale
        )

        # Trier le DataFrame selon la valeur transformée (CDF)
        df_sorted = df.sort_values(by="weibull_cdf")

        # Extraire la liste des IDs triée
        patient_ids = df_sorted["ID"].tolist()

        # Diviser la liste en nb_classes classes de taille (presque) égale
        n = len(patient_ids)
        classes = []
        base_size = n // nb_classes
        remainder = n % nb_classes
        start = 0
        for i in range(nb_classes):
            size = base_size + (1 if i < remainder else 0)
            classes.append(patient_ids[start : start + size])
            start += size

        self.classes = classes
        return classes

    def split_patients_by_os_years2(
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

        self.classes = classes
        return classes

    def classify_mutations(self, max_depth: int):
        """
        Met à jour l'attribut interaction de chaque Mutation de self.mutations.
        """
        assert max_depth > 0

        # Créer un mapping patient_id -> classe (numéro de classe)
        patient_to_class = {}
        for num_classe, classe in enumerate(self.classes):
            for patient_id in classe:
                patient_to_class[patient_id] = num_classe

        # Regrouper les mutations par patient
        mutations_by_patient = {}
        for mutation in self.mutations:
            # On suppose que mutation.carrier contient l'ID du patient
            mutations_by_patient.setdefault(mutation.carrier, []).append(
                mutation
            )

        for depth in range(max_depth, 0, -1):
            # Pour chaque patient, générer les combinaisons locales
            for patient_id, patient_mutations in mutations_by_patient.items():
                # Ne traiter que les patients possédant au moins 'depth' mutations
                if len(patient_mutations) < depth:
                    continue

                # Récupérer la classe du patient
                num_classe = patient_to_class.get(patient_id)
                if num_classe is None:
                    continue  # Le patient n'est pas assigné à une classe

                # Générer les combinaisons de mutations pour ce patient
                for combo in itertools.combinations(patient_mutations, depth):
                    # Pour chaque mutation de la combinaison, ajouter une interaction
                    combo_id = tuple(sorted([m.id for m in combo]))

                    if combo_id in list(self.mutations_classes.keys()):
                        self.mutations_classes[combo_id].append(num_classe)
                    else:
                        self.mutations_classes[combo_id] = [num_classe]

                    for mutation in combo:
                        # Construire la combinaison privée de cette mutation (liste des IDs des autres mutations)
                        combo_without = sorted(
                            [m.id for m in combo if m != mutation]
                        )
                        mutation.add_interaction(combo_without, num_classe)

    def classify_cytogene(self, max_depth: int):
        """
        Met à jour l'attribut interaction de chaque Cytogene de self.cytogenes.
        """
        assert max_depth > 0

        # Créer un mapping patient_id -> classe (numéro de classe)
        patient_to_class = {}
        for num_classe, classe in enumerate(self.classes):
            for patient_id in classe:
                patient_to_class[patient_id] = num_classe

        # Regrouper les mutations par patient
        cytogene_by_patient = {}
        for cytogene in self.cytogenes:
            # On suppose que cytogene.carrier contient l'ID du patient
            cytogene_by_patient.setdefault(cytogene.carrier, []).append(
                cytogene
            )

        for depth in range(max_depth, 0, -1):
            # Pour chaque patient, générer les combinaisons locales
            for patient_id, patient_cytogene in cytogene_by_patient.items():
                # Ne traiter que les patients possédant au moins 'depth' mutations
                if len(patient_cytogene) < depth:
                    continue

                # Récupérer la classe du patient
                num_classe = patient_to_class.get(patient_id)
                if num_classe is None:
                    continue  # Le patient n'est pas assigné à une classe

                # Générer les combinaisons de mutations pour ce patient
                for combo in itertools.combinations(patient_cytogene, depth):
                    # Pour chaque mutation de la combinaison, ajouter une interaction
                    combo_id = tuple(sorted([c.id for c in combo]))

                    if combo_id in list(self.cytogene_classes.keys()):
                        self.cytogene_classes[combo_id].append(num_classe)
                    else:
                        self.cytogene_classes[combo_id] = [num_classe]

                    for cytogene in combo:
                        # Construire la combinaison privée de cette mutation (liste des IDs des autres mutations)
                        combo_without = sorted(
                            [c.id for c in combo if c != cytogene]
                        )
                        cytogene.add_interaction(combo_without, num_classe)

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

    def classify_mutation_tuples_with_score(
        self, df: pd.DataFrame, max_depth: int, k: int = 5
    ) -> dict:
        """
        Pour un DataFrame de mutations, cette fonction :
        - Extrait les mutations via self.extract_mutations(df, True)
        - Regroupe les mutations par patient (en se basant sur l'attribut 'carrier')
        - Pour chaque patient, génère toutes les combinaisons (de taille 'depth')
            de ses mutations (si le patient possède au moins 'depth' mutations)
        - Pour chaque combinaison, regarde dans le dictionnaire d'interactions (self.mutations_classes)
            la liste de classes associée (la clé est construite à partir des IDs triés)
        - Pour chaque combinaison présente, on choisit la classe la plus fréquente dans sa liste.
        - Enfin, pour chaque patient, on regroupe les classes obtenues sur l'ensemble des combinaisons
            et on calcule un score pondéré : somme_{i} (classe_i * beta_i),
            où beta_i = i^5 / (sum_{j=1}^{nb_classes} j^k).

        Parameters:
        - df : DataFrame contenant les mutations.
        - max_depth : profondeur maximale pour générer les combinaisons.
        - k : exposant pour le calcul du poids (beta), par défaut 5.

        Returns:
        Un dictionnaire dont les clés sont les IDs des patients et les valeurs sont le score pondéré
        calculé à partir des classes de leurs combinaisons de mutations, ou None si aucune combinaison n'est trouvée.
        """
        # Extraire les mutations du DataFrame
        mutations = self.extract_mutations(df, True)

        # Regrouper les mutations par patient
        mutations_by_patient = {}
        for mutation in mutations:
            mutations_by_patient.setdefault(mutation.carrier, []).append(
                mutation
            )

        patient_classification = {}
        # Parcourir les profondeurs de max_depth à 1
        for depth in range(max_depth, 0, -1):
            # Pour chaque patient, générer les combinaisons et déterminer la classe
            for patient_id, patient_mutations in mutations_by_patient.items():
                # Si le patient est déjà classifié via une combinaison de plus grande profondeur, passer
                if patient_id in patient_classification:
                    continue

                if len(patient_mutations) < depth:
                    continue

                # Générer les combinaisons de mutations pour ce patient
                combos = list(itertools.combinations(patient_mutations, depth))
                classes_for_patient = []

                for combo in combos:
                    # Construire une clé hashable : un tuple des IDs triés
                    key = tuple(sorted([mut.id for mut in combo]))
                    if key in self.mutations_classes:
                        # Récupérer la liste des classes associées à cette combinaison
                        classes_list = self.mutations_classes[key]
                        # Choisir la classe la plus fréquente pour cette combinaison
                        most_common_class, _ = Counter(
                            classes_list
                        ).most_common(1)[0]
                        classes_for_patient.append(most_common_class)

                if classes_for_patient:
                    # Calculer le score pondéré pour ce patient.
                    nb_classes = len(self.classes)
                    denom = sum(j**k for j in range(1, nb_classes + 1))
                    weighted_score = sum(
                        (cl + 1) * (((cl + 1) ** 5) / denom)
                        for cl in classes_for_patient
                    )
                    patient_classification[patient_id] = -weighted_score
                else:
                    patient_classification[patient_id] = None

        self.patient_test_combinations = patient_classification
        return patient_classification

    def classify_cytogene_tuples_with_score(
        self, df: pd.DataFrame, max_depth: int, k: int = 5
    ) -> dict:
        """
        Parameters:
        - df : DataFrame contenant les mutations.
        - max_depth : profondeur maximale pour générer les combinaisons.
        - k : exposant pour le calcul du poids (beta), par défaut 5.

        Returns:
        Un dictionnaire dont les clés sont les IDs des patients et les valeurs sont le score pondéré
        calculé à partir des classes de leurs combinaisons de cytogènes, ou None si aucune combinaison n'est trouvée.
        """
        # Extraire les mutations du DataFrame
        cytogenes = self.extract_cytogene(df, True)

        # Regrouper les mutations par patient
        cytogenes_by_patient = {}
        for cytogene in cytogenes:
            cytogenes_by_patient.setdefault(cytogene.carrier, []).append(
                cytogene
            )

        patient_classification = {}
        # Parcourir les profondeurs de max_depth à 1
        for depth in range(max_depth, 0, -1):
            # Pour chaque patient, générer les combinaisons et déterminer la classe
            for patient_id, patient_cytogenes in cytogenes_by_patient.items():
                # Si le patient est déjà classifié via une combinaison de plus grande profondeur, passer
                if patient_id in patient_classification:
                    continue

                if len(patient_cytogenes) < depth:
                    continue

                # Générer les combinaisons de mutations pour ce patient
                combos = list(itertools.combinations(patient_cytogenes, depth))
                classes_for_patient = []

                for combo in combos:
                    # Construire une clé hashable : un tuple des IDs triés
                    key = tuple(sorted([cyt.id for cyt in combo]))
                    if key in self.cytogene_classes:
                        # Récupérer la liste des classes associées à cette combinaison
                        classes_list = self.cytogene_classes[key]
                        # Choisir la classe la plus fréquente pour cette combinaison
                        most_common_class, _ = Counter(
                            classes_list
                        ).most_common(1)[0]
                        classes_for_patient.append(most_common_class)

                if classes_for_patient:
                    # Calculer le score pondéré pour ce patient.
                    nb_classes = len(self.classes)
                    denom = sum(j**k for j in range(1, nb_classes + 1))
                    weighted_score = sum(
                        (cl + 1) * (((cl + 1) ** 5) / denom)
                        for cl in classes_for_patient
                    )
                    patient_classification[patient_id] = weighted_score
                else:
                    patient_classification[patient_id] = None

        self.cyto_test_combinations = patient_classification
        return patient_classification

    def extract_mutations(
        self, df: pd.DataFrame, is_test: bool = False
    ) -> list:
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
        if is_test:
            mutations = []
        # Itérer sur chaque ligne du DataFrame
        for _, row in df.iterrows():
            if row["GENE"] in genes:
                mutation = Mutation(
                    carrier=row["ID"],
                    gene=row["GENE"],
                    vaf=row["VAF"],
                    effect=row["EFFECT"],
                    # Ajouter d'autres attributs ici, par exemple :
                    # depth=row["DEPTH"],
                    # protein_change=row["PROTEIN_CHANGE"],
                    # etc.
                )
                if is_test:
                    mutations.append(mutation)
                else:
                    self.mutations.append(mutation)

        if is_test:
            return mutations
        else:
            return self.mutations

    def extract_cytogene(
        self, df: pd.DataFrame, is_test: bool = False
    ) -> list:
        """
        Extrait des instances de Cytogene pour une liste de cytogène clés.
        """
        important_cytogenes = [
            "del(5)",
            "del(7)",
            "trisomy 8",
            "complex",
            "t(3;3)",
            "inv(3)",
            "t(11;19)",
            "del(5q)",
            "-7",
            "7",
            "-17",
        ]
        if is_test:
            cytogenes = []
        # Itérer sur chaque ligne du DataFrame
        for _, row in df.iterrows():
            for important_cytogene in important_cytogenes:
                if important_cytogene in row["CYTOGENETICS"]:
                    cytogene = Cytogene(
                        carrier=row["ID"], cytogene=important_cytogene
                    )
                    if is_test:
                        cytogenes.append(cytogene)
                    else:
                        self.cytogenes.append(cytogene)

        if is_test:
            return cytogenes
        else:
            return self.cytogenes
