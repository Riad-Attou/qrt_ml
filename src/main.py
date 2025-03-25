import warnings

import matplotlib.pyplot as plt

warnings.simplefilter(action="ignore", category=FutureWarning)

from clinical_processing import modele_survival, traitement_donnees


def plot_scores():
    nb_classes_range = list(range(15, 60, 1))
    max_depth_fixed = 15
    max_depth_cyto = 10

    train_scores = []

    for nb_classes in nb_classes_range:
        print(
            f"\nCalcul pour nb_classes = {nb_classes}, max_depth = {max_depth_fixed}, max_depth_cyto = {max_depth_cyto}"
        )
        merged_train, _, target_df = traitement_donnees(
            nb_classes, max_depth_fixed, max_depth_cyto
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


def plot_scores_depth():
    nb_classes = 56
    max_depth_fixed_range = list(range(1, 20, 1))
    max_depth_cyto = 10

    train_scores = []

    for max_depth_fixed in max_depth_fixed_range:
        print(
            f"\nCalcul pour nb_classes = {nb_classes}, max_depth = {max_depth_fixed}, max_depth_cyto = {max_depth_cyto}"
        )
        merged_train, _, target_df = traitement_donnees(
            nb_classes, max_depth_fixed, max_depth_cyto
        )
        train_ci_ipcw = modele_survival(
            1, merged_train, merged_train, target_df
        )
        train_scores.append(train_ci_ipcw)

    # Plot des scores en fonction de nb_classes
    plt.figure(figsize=(10, 6))
    plt.plot(
        max_depth_fixed_range, train_scores, label="Train CI IPCW", marker="o"
    )
    plt.xlabel("Nombre de combinaisons de mutations (nb_max_depth)")
    plt.ylabel("Score CI IPCW")
    plt.title("Scores CI IPCW vs nb_max_depth pour nb_classes = 56")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    nb_classes = 56  # 39 pour opti score_train, 56 pour opti score_test
    max_depth = 15
    max_depth_cyto = 10
    results = []

    print(
        f"Exécution pour nb_classes = {nb_classes}, max_depth = {max_depth}, max_depth_cyto = {max_depth_cyto}"
    )
    merged_train, merged_test, target_df = traitement_donnees(
        nb_classes, max_depth, max_depth_cyto
    )

    modele_survival(1, merged_train, merged_test, target_df)

    plot_scores_depth()
