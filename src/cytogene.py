class Cytogene:
    def __init__(self, carrier: str, cytogene: str):
        self.cytogene = cytogene
        self.carrier = carrier
        self.interactions = (
            {}
        )  # Dictionnaire dont les clés sont des tuples d'autres mutations et la valeur la (ou les) classes du tuple complet (clé + cette mutation).

        self.id = str(self.cytogene)

    def add_interaction(self, key: list, value: int):
        key_tuple = tuple(key)  # Rendre immuable pour l'utiliser comme clé
        if (
            key_tuple in list(self.interactions.keys())
            and self.interactions[key_tuple] != value
        ):
            self.interactions[key_tuple].append(value)
        else:
            self.interactions[key_tuple] = [value]

        return

    def __repr__(self):
        return f"Mutation(patient_id={self.carrier}, cytogene={self.cytogene})"
