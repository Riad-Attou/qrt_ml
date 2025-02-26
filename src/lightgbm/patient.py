class Patient:
    def __init__(
        self,
        id: str,
        is_train: bool,
        is_dead: bool,
        survival_time: float,
        mutations: list = [],
    ):
        self.id = id
        self.mutations = mutations
        self.mutations_combinations = []
        self.is_train = (
            is_train  # Est une donnée d'entrainement (True) ou de test (False).
        )
        # Définir d'autres données depuis les .csv.
        self.is_dead = (
            is_dead  # Gestion de la censure (Vivant: 0 = False, mort: 1 = True).
        )
        self.survival_time = survival_time  # -1 si censure ou donnée de test.

    def __repr__(self):
        return f"Patient(patient_id={self.id}, mutations={self.mutations}, is_train={self.is_train}, is_dead={self.is_dead}, survival_time={self.survival_time})"
