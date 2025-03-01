import itertools
from collections import Counter

import pandas as pd

from cytogene import Cytogene


def classify_cytogene_tuples_with_score(
    df: pd.DataFrame,
    global_cytogene_interactions: dict,  # Nouveau nom pour l'argument attendu
    global_classes: list,
    max_depth: int,
    k: int = 5,
) -> dict:
    """
    Calcule pour chaque patient un score pondéré à partir des combinaisons de cytogènes.

    Parameters:
      - df : DataFrame contenant les données (avec la colonne 'CYTOGENETICS')
      - global_cytogene_interactions : dictionnaire où la clé est un tuple de cytogènes (IDs) et la valeur une liste de classes
      - global_classes : liste de classes globales (par exemple, obtenue via le découpage des patients)
      - max_depth : profondeur maximale pour générer les combinaisons
      - k : exposant pour le calcul du poids (par défaut 5)

    Returns:
      Un dictionnaire dont les clés sont les IDs des patients et les valeurs le score pondéré,
      ou None si aucune combinaison n'est trouvée.
    """
    # Extraire les cytogenes en mode test (sans stocker dans un attribut d'instance)
    cytogenes = extract_cytogenes(df)

    # Regrouper les cytogenes par patient
    cytogenes_by_patient = {}
    for cytogene in cytogenes:
        cytogenes_by_patient.setdefault(cytogene.carrier, []).append(cytogene)

    patient_classification = {}
    for depth in range(max_depth, 0, -1):
        for patient_id, patient_cytogenes in cytogenes_by_patient.items():
            if patient_id in patient_classification:
                continue
            if len(patient_cytogenes) < depth:
                continue

            combos = list(itertools.combinations(patient_cytogenes, depth))
            classes_for_patient = []
            for combo in combos:
                key = tuple(sorted([cyt.id for cyt in combo]))
                if key in global_cytogene_interactions:
                    classes_list = global_cytogene_interactions[key]
                    most_common_class, _ = Counter(classes_list).most_common(
                        1
                    )[0]
                    classes_for_patient.append(most_common_class)
            if classes_for_patient:
                nb_classes = len(global_classes)
                denom = sum(j**k for j in range(1, nb_classes + 1))
                weighted_score = sum(
                    (cl + 1) * (((cl + 1) ** 5) / denom)
                    for cl in classes_for_patient
                )
                patient_classification[patient_id] = weighted_score
            else:
                patient_classification[patient_id] = None

    return patient_classification


def extract_cytogenes(df: pd.DataFrame) -> list:
    """
    Extrait des instances de Cytogene pour une liste de cytogènes clés.
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
    # Créer une liste locale pour stocker les cytogenes
    cytogenes = []
    # Itérer sur chaque ligne du DataFrame
    for _, row in df.iterrows():
        for ic in important_cytogenes:
            if ic in row["CYTOGENETICS"]:
                cytogene = Cytogene(carrier=row["ID"], cytogene=ic)
                cytogenes.append(cytogene)
    return cytogenes
