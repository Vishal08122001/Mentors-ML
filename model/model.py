import pickle
import os
from dotenv import load_dotenv, dotenv_values
from pathlib import Path
import pandas as pd
import pyodbc
import pickle

def main(employeeCode):
    load_dotenv()
    print(employeeCode, "I am EmployeeCode")
    __version__ = "0.1.0"

    BASE_DIR = Path(__file__).resolve(strict=True).parent


    def connectDB():
        conn_str = os.getenv("DB_STRING")
        try:
            conn = pyodbc.connect(conn_str)
            print("Connected")
        except pyodbc.Error as e:
            print("Error connecting to the database:", e)
        return conn


    def load_model_objects():
        with open(f"{BASE_DIR}/trained-pipeline-{__version__}.pkl", 'rb') as model_file:
            knn_model = pickle.load(model_file)
        with open(f"{BASE_DIR}/mlb.pkl", 'rb') as mlb_file:
            mlb_object = pickle.load(mlb_file)

        with open(f"{BASE_DIR}/label.pkl", 'rb') as label_file:
            label_object = pickle.load(label_file)
        return knn_model, mlb_object, label_object


    knn_model, mlb_object, label_object = load_model_objects()

    conn = connectDB()

    def preProcessData():
        query = 'EXEC [ED].[GetAllEmployeeMLData]'
        df = pd.read_sql_query(query, conn);
        mentors_df = df[df['IsMentor'] == 1].drop_duplicates(subset='EmpID')
        mentors_df['Skills'] = mentors_df['Skills'].str.split(', ')
        expanded_skill_data = mlb_object.fit_transform(mentors_df['Skills'])
        expanded_skill_df = pd.DataFrame(expanded_skill_data, columns=mlb_object.classes_, index=mentors_df.index)
        mentors_df = mentors_df.drop('Skills', axis=1).join(expanded_skill_df)
        return mentors_df



    def recommend_mentors(employee_code, knn_model, mlb, label, n_recommendations=5,):
        mentors_df = preProcessData()
        try:
            mentee_query = f"EXEC [ED].[GetEmployeeMLData] @Employeecode = {employee_code}"
            mentee_data = pd.read_sql_query(mentee_query, conn).iloc[:1]
        except pd.errors.DatabaseError as e:
            print("Database Error:", e)
            return None

        try:
            # Preprocessing mentee data
            mentee_data['Skills'] = mentee_data['Skills'].str.split(', ')
            mentee_skill_data = mlb.transform(mentee_data['Skills'])
            mentee_skill_df = pd.DataFrame(mentee_skill_data, columns=mlb.classes_, index=mentee_data.index)
            # mentee_data['Designation'] = label.transform(mentee_data['Designation'])
            mentee_data = mentee_data.drop('Skills', axis=1).join(mentee_skill_df)
        except KeyError:
            print("Column 'Skills' or 'Designation' not found in the mentee data.")
            # Handle the error accordingly
            return None
        
        try:
            # Finding nearest neighbors
            distances, indices = knn_model.kneighbors(mentee_data.drop(['EmployeeNo', 'Name', 'IsMentor'], axis=1), n_neighbors=n_recommendations)
        except ValueError as e:
            print("Value Error:", e)
            # Handle the error accordingly
            return None

        try:
            # Fetching recommended mentors
            recommended_mentors = mentors_df.iloc[indices[0]]
            return recommended_mentors
        except IndexError:
            print("Index Error: Insufficient data to recommend mentors.")
            # Handle the error accordingly
            return None


    answer = recommend_mentors(employeeCode, knn_model=knn_model, mlb=mlb_object, label=label_object)
    print(answer)
    return answer