class Mutation:
    def __init__(self, name: str, vaf: float):
        self.__name = name  # ID
        self.__carriers = []  # Individus porteurs de la mutation
        self.__interactions = (
            {}
        )  # Dictionnaire dont les clés sont des tuples d'autres mutations et la valeur la (ou les) classes du tuple complet (clé + cette mutation).
        self.__vaf = vaf

    def get_name(self):
        return self.__name

    def get_carriers(self):
        return self.__carriers

    def get_interactions(self):
        return self.__interactions

    def add_carriers(self, carrier):
        self.__carriers.append(carrier)
        return

    def add_interaction(self, key: tuple, value: int):
        if key in self.__interactions and self.__interactions[key] != value:
            self.__interactions[key].append(value)
        else:
            self.__interactions[key] = [value]

        self.__interactions[key] = value
        return
