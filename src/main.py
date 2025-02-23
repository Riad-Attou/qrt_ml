import warnings

import matplotlib.pyplot as plt

warnings.simplefilter(action="ignore", category=FutureWarning)

from clinical_processing import modele_survival, traitement_donnees


def plot_scores():
    nb_classes_range = list(range(15, 60, 1))
    max_depth_fixed = 15

    train_scores = []

    for nb_classes in nb_classes_range:
        print(
            f"Calcul pour nb_classes = {nb_classes}, max_depth = {max_depth_fixed}"
        )
        merged_train, _, target_df = traitement_donnees(
            nb_classes, max_depth_fixed
        )
        train_ci_ipcw = modele_survival(
            1, merged_train, merged_train, target_df
        )
        train_scores.append(train_ci_ipcw)

    # Plot des scores en fonction de nb_classes
    plt.figure(figsize=(10, 6))
    plt.plot(nb_classes_range, train_scores, label="Train CI IPCW", marker="o")
    plt.xlabel("Nombre de classes (nb_classes)")
    plt.ylabel("Score CI IPCW")
    plt.title("Scores CI IPCW vs nb_classes pour max_depth = 15")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # --- Première grille d'exploration sur nb_classes et max_depth ---
    nb_classes = 56  # 30 pour opti score_train, 56 pour opti score_test
    max_depth = 15
    results = []

    print(f"Exécution pour nb_classes = {nb_classes}, max_depth = {max_depth}")
    merged_train, merged_test, target_df = traitement_donnees(
        nb_classes, max_depth
    )

    modele_survival(1, merged_train, merged_test, target_df)

    # plot_scores()
