class Mutation:
    def __init__(self, name):
        self.__name = name
        self.__carriers = []  # Individus porteurs de la mutation
        self.__interactions = (
            {}
        )  # Dictionnaire dont les clés sont des tuples d'autres mutations et la valeur la classe du tuple complet (clé + cette mutation).

    def get_name(self):
        return self.__name

    def get_carriers(self):
        return self.__carriers

    def add_carriers(self, carrier):
        self.__carriers.append(carrier)
        return
