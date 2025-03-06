from src.data_preprocessing import clean_data, load_data
from src.feature_engineering import (
    aggregate_molecular_features,
    merge_features,
)
from src.model_training import train_cox_model


def main():
    # Charger les données
    (
        clinical_train,
        molecular_train,
        target_train,
        clinical_test,
        molecular_test,
    ) = load_data()

    # Nettoyer les données
    clinical_train, clinical_test = clean_data(clinical_train, clinical_test)

    # Créer les features à partir des mutations
    molecular_features = aggregate_molecular_features(
        molecular_train, molecular_test
    )

    # Fusionner toutes les données
    train_merged, test_merged = merge_features(
        clinical_train,
        molecular_features,
        target_train,
        clinical_test,
        molecular_test,
    )

    # Séparer X et y
    X = train_merged.drop(columns=["ID"])
    y = target_train.set_index("ID").loc[X.index]

    # Entraîner un modèle
    cox_model = train_cox_model(X, y)

    print(" Modèle entraîné avec succès !")


if __name__ == "__main__":
    main()
