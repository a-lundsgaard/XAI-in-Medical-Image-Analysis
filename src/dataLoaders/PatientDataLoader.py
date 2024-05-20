import pandas as pd
import numpy as np
from typing import Dict

""" 

Descriptions and scales, extracted from the comments file:

1. **BMI** (Body Mass Index):
   - Scale: Numeric value representing BMI, typically calculated from weight in kilograms divided by height in meters squared. This value helps in assessing underweight, normal, overweight, and obesity conditions based on established BMI categories.

2. **AGE** (Age):
   - Scale: Numeric value indicating the age of the participant in years at the time of the visit.

3. **WOMTSL** and **V01WOMTSR** (Western Ontario and McMaster Universities Osteoarthritis Index, Total Score Left and Right Knee):
   - Scale: Ranges from 0 (no pain, stiffness, or difficulty) to 96 (extreme pain, stiffness, and difficulty) .

4. **KQOL2** (Knee injury and Osteoarthritis Outcome Score, Quality of Life Subscale):
   - Scale: Scores are typically on a normalized scale from 0 to 100, where 0 represents extreme knee problems and 100 represents no knee problems.

5. **PASE** (Physical Activity Scale for the Elderly):
   - Scale: Scores are calculated by combining the frequency and duration of various activities. The total score can range theoretically from 0 to over 400, with higher scores indicating a greater level of physical activity .

6. **DILKN10**, **V01DILKN11**, and **V01DILKN2** (Difficulty in Left Knee specific activities):
   - Scale: These variables measure difficulty associated with specific movements or activities, scored from 0 (no difficulty) to a higher number indicating greater difficulty. The activities covered could include walking, climbing stairs, etc., though exact details require further specification.

7. **DIRKN10**, **V01DIRKN11**, and **V01DIRKN2** (Difficulty in Right Knee specific activities):
   - Similar to the left knee difficulty measures, these scores reflect the level of difficulty experienced in activities involving the right knee.

These scales help in quantitatively assessing various health and lifestyle metrics important in medical studies related to knee osteoarthritis and overall health assessments.

"""

""" 
self.metricDict = {
    "BMI": "Body mass index (calc)",
    "AGE": "Age (calc, used for study eligibility)",
    "WOMTSL": "WOMAC pain score for left knee",
    "WOMTSR": "WOMAC pain score for right knee",
    "KQOL2": "Knee injury and Osteoarthritis Outcome Score (KOOS) - Quality of Life",
    "PASE": "Physical Activity Scale for the Elderly",
    "DILKN10": "Left knee, intensity of pain during last week, digit 0",
    "DILKN11": "Left knee, intensity of pain during last week, digit 1",
    "DILKN2": "Left knee, intensity of pain during last week, digit 2",
    "DIRKN10": "Right knee, intensity of pain during last week, digit 0",
    "DIRKN11": "Right knee, intensity of pain during last week, digit 1",
    "DIRKN2": "Right knee, intensity of pain during last week, digit 2"
}
 """


class PatientDataProcessor:
    def __init__(
        self,
        base_path: str = "/Users/askelundsgaard/Documents/datalogi/6-semester/Bachelor/XAI-in-Medical-Image-Analysis/datasets/meta_data/OAIdata21/",
        labels = ["BLFPD", "ALTPD", "IBMFPD"]
    ):
        self.base_path = base_path
        self.data: dict[str, pd.DataFrame] = {}
        # This will store the enrollment data including sex
        self.enroll_df: pd.DataFrame = {}
        self.subjects = np.loadtxt(base_path + "SubjectChar00.txt",
                                   delimiter='|', skiprows=1, usecols=(0,), dtype=int)
        self.visit: str = None
        self.visits: Dict[int, str] = {}
        self.labels = labels

        self.metricDict = {
            "BMI": "BMI",
            "AGE": "AGE",
            "WOMTSL": "WOMTSL",
            "WOMTSR": "WOMTSR",
            "KQOL2": "KQOL2",
            "PASE": "PASE",
            "DILKN10": "DILKN10",
            "DILKN11": "DILKN11",
            "DILKN2": "DILKN2",
            "DIRKN10": "DIRKN10",
            "DIRKN11": "DIRKN11",
            "DIRKN2": "DIRKN2"
        }


    def get_data(self, visit_no: int = None):

        if visit_no and visit_no in self.data:
            return self.data[visit_no]
        """Combine and display all collected data."""

        # print(f"Total number of subjects: {self.data}")
        combined_data: pd.DataFrame = pd.concat(self.data.values(), axis=1)
        # replace V0* from the column names
        # combined_data.columns = [col[0:3] + "_" + col[3:]
        #                          for col in combined_data.columns]
        # print(f"Total length of the dataframe: {len(combined_data)}")
        return combined_data
    
    def get_all_clinical_data(self):
        all_data = pd.DataFrame()
        for visit_no in range(0, 12):
            all_data = pd.concat([all_data, self.get_clinical_data(visit_no)], axis=1)
        return all_data    
    
    def get_clinical_data(self, visit_no: int):
        visit = "V0" + str(visit_no) if visit_no < 10 else "V" + str(visit_no)
        self.visit = visit
        self.visits[visit_no] = visit
        df_visit = self.load_data(f"AllClinical{visit[1:]}.txt")
        if df_visit.empty:
            return

        df_visit = df_visit[df_visit.index.isin(self.subjects)]
        variables_of_interest = self.define_variables_of_interest(
            visit, df_visit.columns)
        df_visit = df_visit[variables_of_interest]

        self.clean_data(df_visit)
        self.get_enrollment_data()
        # print(f"Enrollment data: {self.enroll_df}")
        self.fill_missing_with_mean(df_visit, variables_of_interest)
        return df_visit
    

    def create_meta_data_for_visit(self, visit_no: int):
        visit = "V0" + str(visit_no) if visit_no < 10 else "V" + str(visit_no)
        # self.visit = visit
        # self.visits[visit_no] = visit
        # df_visit = self.load_data(f"AllClinical{visit[1:]}.txt")
        # if df_visit.empty:
        #     return

        # df_visit = df_visit[df_visit.index.isin(self.subjects)]
        # variables_of_interest = self.define_variables_of_interest(
        #     visit, df_visit.columns)
        # df_visit = df_visit[variables_of_interest]

        # self.clean_data(df_visit)
        # self.get_enrollment_data()
        # # print(f"Enrollment data: {self.enroll_df}")
        # self.fill_missing_with_mean(df_visit, variables_of_interest)

        # df_visit = df_visit.merge(
        #     self.enroll_df, left_index=True, right_index=True, how='left')
        
        # kmri_data = self.load_kMRI_data(visit_no)
        # if kmri_data is not None:
        #     df_visit = df_visit.merge(
        #         kmri_data, left_index=True, right_index=True)
        
        # self.data[visit_no] = df_visit
        self.data[visit_no] = self.load_kMRI_data(visit_no, labels = self.labels)
        # print length of the dataframe
        print(f"Length of the dataframe: {len(self.data[visit_no])}")
        # self.get_kellberg_lawrence_grade(visit_no)
        # self.load_kMRI_data(visit_no)

        return self.data


    def get_visit(self, visit_no: int = None):
        if visit_no in self.data:
            visit = "V0" + str(visit_no) if visit_no < 10 else "V" + str(visit_no)
            return visit
        else : 
            raise ValueError(f"Visit {visit_no} not created. Please create the visit")
    
    def get_enrollment_data(self):
        # Load the enrollment data
        enroll = self.base_path + "Enrollees.txt"
        df_enroll = pd.read_csv(enroll, sep="|", index_col="ID")
        # Create a new DataFrame that contains only the 'Sex' column with converted values
        self.enroll_df = df_enroll[['P02SEX']].replace(
            {'1: Male': 0, '2: Female': 1}).infer_objects(copy=False)
        self.enroll_df.rename(columns={'P02SEX': 'Sex'}, inplace=True)


    # def load_kMRI_data(self, visit_no: int):
    #     visit = "V0" + str(visit_no) if visit_no < 10 else "V" + str(visit_no)
    #     print(f"Loading kMRI data for visit {visit}")
    #     df = self.load_data(f"kMRI_QCart_Eckstein{visit[1:]}.txt")
    #     if df.empty:
    #         return
    #     df = df[df.index.isin(self.subjects)]

    #     # print(f"Loaded kMRI data for visit {df}")

    #     variables = ["BLFPD", "ALTPD", "IBMFPD"]
    #     for var in variables:
    #         right_var = f"{visit}{var}R"
    #         left_var = f"{visit}{var}L"
    #         df[right_var] = df.apply(lambda row: row[f"{visit}{var}"] if row['SIDE'] == '1: Right' else np.nan, axis=1)
    #         df[left_var] = df.apply(lambda row: row[f"{visit}{var}"] if row['SIDE'] == '2: Left' else np.nan, axis=1)

    #     variables_with_suffix = [f"{visit}{var}R" for var in variables] + [f"{visit}{var}L" for var in variables]
    #     df = df[variables_with_suffix]

    #     return df
    
    # def load_kMRI_data(self, visit_no: int):
    #     visit = "V0" + str(visit_no) if visit_no < 10 else "V" + str(visit_no)
    #     print(f"Loading kMRI data for visit {visit}")
    #     # df = self.load_data(f"kMRI_QCart_Eckstein{visit[1:]}.txt", index_col="ID")
    #     name = f"kMRI_QCart_Eckstein{visit[1:]}.txt"
    #     path = f"{self.base_path}/{name}"

    #     df = pd.read_csv(path, sep="|", header=0)
    #     if df.empty:
    #         print(f"No data found for visit {visit}")
    #         return
    #     # df = df[df.index.isin(self.subjects)]
    #     # if df.empty:
    #     #     print(f"No matching subjects found for visit {visit}")
    #     #     return
        
    #     return df
    
    def load_all_kMRI_data(self, labels: list[str] = ["BLFPD", "ALTPD", "IBMFPD"]):
        all_data = pd.DataFrame()
        for i in range(0, 12):
            all_data =pd.concat([all_data, self.load_kMRI_data(i, labels)], axis=1)

        return all_data

    def load_kMRI_data(self, visit_no: int, labels: list[str] = ["BLFPD", "ALTPD", "IBMFPD"]):
        visit = "V0" + str(visit_no) if visit_no < 10 else "V" + str(visit_no)
        print(f"Loading kMRI data for visit {visit}")
        name = f"kMRI_QCart_Eckstein{visit[1:]}.txt"
        path = f"{self.base_path}/{name}"

        try:
            df = pd.read_csv(path, sep="|", header=0)
            if df.empty:
                print(f"No data found for visit {visit}")
                return
        except Exception as e:
            print("No file found at: ", path, "skipping.")
            return pd.DataFrame() 
        
        # Variables of interest
        variables = labels
        variables_with_prefix = [f"{visit}{var}" for var in variables]

        # Filter the dataframe to only include the variables of interest
        variables_with_side = ['ID', 'SIDE'] + variables_with_prefix
        df = df[variables_with_side]

        # if emtpy return the dataframe
        if df.empty:
            print(f"Labels: {self.labels} not+ found for visit {visit}")
            return df

        df['SIDE'] = df['SIDE'].astype(str)
        print(df['SIDE'].dtype)
        df_left = df[df['SIDE'].str.contains('2')]
        df_left = df_left.drop(columns=['SIDE'])
        df_left = df_left.groupby('ID').mean().add_suffix('L')

        df_right = df[df['SIDE'].str.contains('1')]
        df_right = df_right.drop(columns=['SIDE'])
        df_right = df_right.groupby('ID').mean().add_suffix('R')
        df_combined = pd.merge(df_left, df_right, left_index=True, right_index=True)

        # Set ID as the index
        df_combined.index.name = 'ID'
        print(f"Length of the dataframe: {len(df_combined)}")
        return df_combined



    def get_kellberg_lawrence_grade(self, visit_no: int):
        visit = "0" + str(visit_no) if visit_no < 10 else str(visit_no)
        df = self.load_data("kxr_sq_bu" + visit[1:] + ".txt", "ID")
        if df.empty:
            raise ValueError(f"Data not found for visit {visit_no}")
        df = df[df.index.isin(self.subjects)]
        column_name = 'V' + visit + 'XRKL'
        df = df[[column_name]]
        if df.empty:
            column_name = 'v' + visit + 'XRKL'
            df = df[[column_name]]
        if df.empty:
            raise ValueError(f"Data not found for visit {visit_no}")
        # df.rename(columns={'XRKL': 'KLGrade'}, inplace=True)
        self.data[visit_no] = self.data[visit_no].merge(
            df, left_index=True, right_index=True, how='left')
        return self.data


    def load_data(self, file_name: str, index_col: str = "ID") -> pd.DataFrame:
        """Load data from a given file within the base path, handling pipe-delimited format."""
        path = f"{self.base_path}/{file_name}"
        try:
            return pd.read_csv(path, sep="|", header=0, index_col=index_col)
        except Exception as e:
            print(f"Error loading data from {path}: {e}")
            return pd.DataFrame()
        
    def load_all_visits(self):
        for visit_no in range(0, 12):  # Assuming there are 10 visits
            self.create_meta_data_for_visit(visit_no)

    def load_specific_visits(self, visit_nos: list[int]):
        for visit_no in visit_nos:
            self.create_meta_data_for_visit(visit_no)

    def fill_missing_with_mean(self, dataframe: pd.DataFrame, variables_of_interest: list[str]):
        """Fill missing variables with the mean of available data."""
        for column in variables_of_interest:
            if dataframe[column].isnull().sum() > 0:
                mean = dataframe[column].mean().__round__(1)
                dataframe[column].fillna(mean, inplace=True)

    def define_variables_of_interest(self, visit: str, available_columns: pd.Index) -> list:
        """Dynamically define variables of interest based on the visit code and check if they exist in the DataFrame."""
        # base_vars = ["BMI", "AGE"]
        base_vars = ["AGE"]

        # common_vars = ["WOMTSL", "WOMTSR", "KQOL2", "PASE"]
        common_vars = []
        # pain_vars = ["DILKN10", "DILKN11", "DILKN2",
        #              "DIRKN10", "DIRKN11", "DIRKN2", "WOMKPL", "WOMKPR", "XRKL"]
        pain_vars = ["WOMKPL", "WOMKPR", "XRKL"]
        # follow_up_vars = ["P01BMI"]

        all_vars_to_prefix = base_vars + common_vars + pain_vars
        all_vars = [visit + var for var in all_vars_to_prefix]

        # Only include variables that exist in the DataFrame
        existing_vars = [var for var in all_vars if var in available_columns]
        missing_cols = set(all_vars) - set(existing_vars)
        if missing_cols:
            print(f"Missing columns for visit {visit}: {missing_cols}")

        return existing_vars

    def clean_data(self, dataframe: pd.DataFrame):
        """Convert categorical data and handle missing values."""
        for column in dataframe.columns:
            if dataframe[column].dtype == 'object':
                dataframe[column] = self.convert_categorical_to_numeric(
                    dataframe[column])

    def convert_categorical_to_numeric(self, series: pd.Series) -> pd.Series:
        """Convert categorical series to numeric by extracting numeric codes if formatted as 'number: description'."""
        def safe_convert(x):
            try:
                return int(x.split(':')[0]) if isinstance(x, str) and ':' in x else x
            except ValueError:
                return np.nan
        return series.apply(safe_convert)



