import pandas as pd
import numpy as np

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

class PatientDataProcessor:
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.data = {}
        self.enroll_df = {}  # This will store the enrollment data including sex

        self.metricDict  = {
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

    # def get_enrollment_data(self):
    #     # Load the enrollment data
    #     enroll = self.base_path + "Enrollees.txt"
    #     df_enroll = pd.read_csv(enroll, sep="|", index_col="ID")
    #     df_enroll['Sex'] = df_enroll['P02SEX'].replace({'1: Male': 0, '2: Female': 1})
    #     self.enroll_df = df_enroll

    def get_enrollment_data(self):
        # Load the enrollment data
        enroll = self.base_path + "Enrollees.txt"
        df_enroll = pd.read_csv(enroll, sep="|", index_col="ID")
        # Create a new DataFrame that contains only the 'Sex' column with converted values
        self.enroll_df = df_enroll[['P02SEX']].replace({'1: Male': 0, '2: Female': 1})
        self.enroll_df.rename(columns={'P02SEX': 'Sex'}, inplace=True)

    def load_data(self, file_name: str, index_col: str = "ID") -> pd.DataFrame:
        """Load data from a given file within the base path, handling pipe-delimited format."""
        path = f"{self.base_path}/{file_name}"
        try:
            return pd.read_csv(path, sep="|", header=0, index_col=index_col)
        except Exception as e:
            print(f"Error loading data from {path}: {e}")
            return pd.DataFrame()

    def extract_variables(self, visit_no: int, subjects: np.ndarray):
        visit = "V0" + str(visit_no) if visit_no < 10 else "V" + str(visit_no)
        df_visit = self.load_data(f"AllClinical{visit[1:]}.txt")
        if df_visit.empty:
            return
        
        df_visit = df_visit[df_visit.index.isin(subjects)]
        variables_of_interest = self.define_variables_of_interest(visit, df_visit.columns)
        df_visit = df_visit[variables_of_interest]

        self.clean_data(df_visit)
        self.get_enrollment_data()
        # print(f"Enrollment data: {self.enroll_df}")

        self.fill_missing_with_mean(df_visit, variables_of_interest)
        df_visit = df_visit.merge(self.enroll_df, left_index=True, right_index=True, how='left')
        self.data[visit_no] = df_visit

    def fill_missing_with_mean(self, dataframe: pd.DataFrame, variables_of_interest: list[str]):
        """Fill missing variables with the mean of available data."""
        for column in variables_of_interest:
            if dataframe[column].isnull().sum() > 0:
                mean = dataframe[column].mean().__round__(1)
                dataframe[column].fillna(mean, inplace=True)

    def define_variables_of_interest(self, visit: str, available_columns: pd.Index) -> list:
        """Dynamically define variables of interest based on the visit code and check if they exist in the DataFrame."""
        base_vars = ["BMI", "AGE"]
        common_vars = ["WOMTSL", "WOMTSR", "KQOL2", "PASE"]
        pain_vars = ["DILKN10", "DILKN11", "DILKN2", "DIRKN10", "DIRKN11", "DIRKN2"]
        follow_up_vars = ["P01BMI"]
        all_vars_to_prefix = base_vars + common_vars + pain_vars
        all_vars_to_prefix = [visit + var for var in all_vars_to_prefix]
        all_vars = follow_up_vars + all_vars_to_prefix

        # print(f"Variables for visit {visit}")
        # if visit != "V00":
        #     prefixed_vars = follow_up_vars + prefixed_vars
        # else: 
        #     prefixed_vars = ["V00BMI"] + prefixed_vars

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
                dataframe[column] = self.convert_categorical_to_numeric(dataframe[column])

    def convert_categorical_to_numeric(self, series: pd.Series) -> pd.Series:
        """Convert categorical series to numeric by extracting numeric codes if formatted as 'number: description'."""
        def safe_convert(x):
            try:
                return int(x.split(':')[0]) if isinstance(x, str) and ':' in x else x
            except ValueError:
                return np.nan
        return series.apply(safe_convert)

    def display_data(self):
        """Combine and display all collected data."""
        combined_data = pd.concat(self.data.values(), axis=1)
        # replace V0* from the column names
        combined_data.columns = [ col[0:3] + "_" + col[3:] for col in combined_data.columns]
        print(f"Total length of the dataframe: {len(combined_data)}")
        return combined_data

# Example usage:
base_path = "/Users/askelundsgaard/Documents/datalogi/6-semester/Bachelor/XAI-in-Medical-Image-Analysis/datasets/meta_data/OAIdata21/"
processor = PatientDataProcessor(base_path)
subjects = np.loadtxt(base_path + "SubjectChar00.txt", delimiter='|', skiprows=1, usecols=(0,), dtype=int)
processor.extract_variables(visit_no=1, subjects=subjects)
df_display = processor.display_data()

# print first 10 rows of the dataframe
print(df_display.head(10))