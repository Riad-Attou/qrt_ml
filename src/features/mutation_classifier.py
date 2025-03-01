import itertools
from collections import Counter
from typing import Dict, List

import pandas as pd

from mutation import Mutation


def classify_mutation_tuples_with_score(
    mutations: List[Mutation],
    patient_to_class: Dict[str, int],
    global_mutation_interactions: Dict[tuple, List[int]],
    max_depth: int,
    k: int = 5,
) -> Dict[str, float]:
    """
    Pour chaque patient, génère les combinaisons de mutations et calcule un score pondéré
    à partir des classes issues des interactions.

    Parameters:
      - mutations : liste d'instances de Mutation (retournée par extract_mutations)
      - patient_to_class : dictionnaire liant chaque patient à sa classe (ex. résultat d'un split)
      - global_mutation_interactions : dictionnaire global où la clé est un tuple de mutation (ex. les IDs triés) et la valeur une liste de classes observées
      - max_depth : profondeur maximale pour générer les combinaisons
      - k : exposant pour le calcul des poids (par défaut 5)

    Returns:
      - Un dictionnaire avec pour chaque patient (clé) le score pondéré calculé.
    """
    # Convertir la liste de mutations en dictionnaire : patient_id -> [mutation, ...]
    mutations_by_patient = {}
    for mutation in mutations:
        mutations_by_patient.setdefault(mutation.carrier, []).append(mutation)

    patient_classification = {}
    # Parcourir les profondeurs de max_depth à 1
    for depth in range(max_depth, 0, -1):
        for patient_id, patient_mutations in mutations_by_patient.items():
            # Si déjà classifié via une combinaison de profondeur supérieure, on passe
            if patient_id in patient_classification:
                continue
            if len(patient_mutations) < depth:
                continue

            combos = list(itertools.combinations(patient_mutations, depth))
            classes_for_patient = []
            for combo in combos:
                key = tuple(sorted([m.id for m in combo]))
                if key in global_mutation_interactions:
                    classes_list = global_mutation_interactions[key]
                    most_common_class, _ = Counter(classes_list).most_common(
                        1
                    )[0]
                    classes_for_patient.append(most_common_class)
            if classes_for_patient:
                nb_classes = len(set(patient_to_class.values()))
                denom = sum(j**k for j in range(1, nb_classes + 1))
                weighted_score = sum(
                    (cl + 1) * (((cl + 1) ** 5) / denom)
                    for cl in classes_for_patient
                )
                patient_classification[patient_id] = -weighted_score
            else:
                patient_classification[patient_id] = None

    return patient_classification


def extract_mutations(df: pd.DataFrame) -> list:
    """
    Extrait des instances de Mutation pour une liste de gènes clés.
    Pour chaque ligne du DataFrame correspondant à un gène clé,
    une instance de Mutation est créée en extrayant notamment le VAF, l'effet, etc.
    """
    from mutation import Mutation

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
    mutations = []  # Liste locale pour stocker les mutations

    # Itérer sur chaque ligne du DataFrame
    for _, row in df.iterrows():
        if row["GENE"] in genes:
            mutation = Mutation(
                carrier=row["ID"],
                gene=row["GENE"],
                vaf=row["VAF"],
                effect=row["EFFECT"],
                # Ajouter d'autres attributs ici si nécessaire, par ex :
                # depth=row["DEPTH"],
                # protein_change=row["PROTEIN_CHANGE"],
            )
            mutations.append(mutation)

    return mutations
