from database import Database
from patient import Patient

if __name__ == "__main__":
    patient = Patient("1000", True, True, 0.5)
    db = Database()
    df_train, df_eval, mol_df, mol_eval, target_df = db.load_data()
    print(len(db.patients))
