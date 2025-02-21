class Patient:
    def __init__(self, id):
        self.__id = id
        self.__mutations = []
        # Define other data from the .csv.

    def get_id(self):
        return self.__id

    def get_mutations(self):
        return self.__mutations
