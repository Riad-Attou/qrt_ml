class Mutation:
    def __init__(self, carrier: str, gene: str, vaf: float, effect: str):
        self.gene = gene
        self.carrier = carrier
        self.interactions = (
            {}
        )  # Dictionnaire dont les clés sont des tuples d'autres mutations et la valeur la (ou les) classes du tuple complet (clé + cette mutation).
        self.vaf = vaf
        self.effect = effect

    def add_carriers(self, carrier):
        self.carriers.append(carrier)
        return

    def add_interaction(self, key: tuple, value: int):
        if key in self.interactions and self.interactions[key] != value:
            self.interactions[key].append(value)
        else:
            self.interactions[key] = [value]

        self.interactions[key] = value
        return

    def __repr__(self):
        return f"Mutation(patient_id={self.carrier}, gene={self.gene}, vaf={self.vaf}, effect={self.effect})"
