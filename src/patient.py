class Patient:
    def __init__(
        self, id: str, is_train: bool, is_dead: bool, survival_time: float
    ):
        self.__id = id
        self.__mutations = []
        self__is_train = is_train  # Est une donnée d'entrainement (True) ou de test (False).
        # Définir d'autres données depuis les .csv.
        self.__is_dead = is_dead  # Gestion de la censure (Vivant: 0 = False, mort: 1 = True).
        self.__survival_time = (
            survival_time  # -1 si censure ou donnée de test.
        )

    def get_id(self):
        return self.__id

    def get_mutations(self):
        return self.__mutations
