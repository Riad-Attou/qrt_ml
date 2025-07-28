import re
from typing import Any

import pandas as pd


def monosomie(cyto_str: Any) -> Any:
    """
    Renvoie le nombre de monosomies que présente un patient.
    Si l'entrée n'est pas une chaîne, retourne "inconnu".
    """
    if not isinstance(cyto_str, str):
        return "inconnu"
    s = cyto_str.lower()
    monosomies = re.findall(r"-(\d+)", s)
    return len(monosomies)


def classify_cytogenetic_risk(cyto_str: Any) -> str:
    """
    Classifie le risque cytogénétique en se basant sur des motifs connus.
    Renvoie une chaîne ("favorable", "intermediaire", "defavorable") ou "inconnu".
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
        or ("t(3;3)" in s)
        or ("inv(3)" in s)
        or ("t(11;19)" in s)
        or ("del(5q)" in s)
        or ("-7" in s)
        or ("7" in s)
        or ("-17" in s)
        or (monosomie(s) > 2)
    ):
        return "defavorable"

    # Sinon, c'est plutôt intermediaire (ex : caryotype normal)
    return "intermediaire"


def extract_features(cyto_str: Any) -> pd.Series:
    """
    Extrait plusieurs caractéristiques à partir de la chaîne cytogénétique.
    Renvoie un Series avec :
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
    nb_t = max(nb_t_total - nb_philadelphia, 0)
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


def create_features_clinical_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrichit le DataFrame clinique en ajoutant :
      - Les features extraites depuis la colonne 'CYTOGENETICS'
      - La classification du risque cytogénétique
      - Le nombre de monosomies
      - Des variables dummies à partir de la classification
    """
    features_df = df["CYTOGENETICS"].apply(extract_features)
    df = pd.concat([df, features_df], axis=1)
    df["cyto_risk"] = df["CYTOGENETICS"].apply(classify_cytogenetic_risk)
    df["nbr_mono"] = df["CYTOGENETICS"].apply(monosomie)
    cyto_risk_dummies = pd.get_dummies(df["cyto_risk"], prefix="cyto_risk")
    df = pd.concat([df, cyto_risk_dummies], axis=1)
    return df
