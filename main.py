import PySimpleGUI as sg
import numpy as np
import pandas as pd
from tkinter.filedialog import askopenfilename
from joblib import load
# from sklearn.linear_model import LogisticRegression
import csv
import warnings


def warn(*args, **kwargs):
    # Overriding warnings
    pass


warnings.warn = warn


"""Patient data and health parameters are to be input
the numerical values as sliding
Radio toggles for arbitrary choices with description.
age: age in years
sex: sex (1 = male; 0 = female)
cp: chest pain type
-- Value 1: typical angina
-- Value 2: atypical angina
-- Value 3: non-anginal pain
-- Value 4: asymptomatic
trestbps: resting blood pressure (in mm Hg on admission to the hospital)
chol: serum cholestoral in mg/dl
fbs: (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
restecg: resting electrocardiographic results
-- Value 0: normal
-- Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
thalach: maximum heart rate achieved
exang: exercise induced angina (1 = yes; 0 = no)
oldpeak = ST depression induced by exercise relative to rest
slope: the slope of the peak exercise ST segment
-- Value 1: upsloping
-- Value 2: flat
-- Value 3: downsloping
ca: number of major vessels (0-3) colored by flourosopy
thal: 3 = normal; 6 = fixed defect; 7 = reversable defect
(not to be input)num: diagnosis of heart disease (angiographic disease status)
-- Value 0: < 50% diameter narrowing
-- Value 1: > 50% diameter narrowing
"""

menu_def = [['Help', ['User Manual']],
            ['About', 'About...']]
ATTRIBUTES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca',
              'thal']

# Setting the layout of the GUI and buttons that call functions
layout = [[sg.Text("Please, enter the patient's data.")],
          [sg.Text('Age (from the dropdown menu)'),
           sg.OptionMenu(list(range(25, 100)), size=(4, 8), default_value='Select', key='age')],
          [sg.Text('Sex'), sg.OptionMenu(["Male", 'Female'], default_value='Select', key='sex')],
          [sg.Text('Chest pain type'),
           sg.OptionMenu(['typical angina', 'atypical angina', 'non-anginal pain', 'asymptomatic'], size=(16, 16),
                         default_value='Select', key='cp')],
          [sg.Text('resting systolic blood pressure'), sg.Input(size=(6, 2), key="trestbps")],
          [sg.Text('Serum cholesterol in mg/dl'), sg.Input(size=(6, 2), key="chol")],
          [sg.Text('Fasting blood sugar'),
           sg.OptionMenu(list(["Higher than 120 mg/dl", "Lower than 120 mg/dl"]), size=(4, 16), default_value='Select',
                         key='fbs')],
          [sg.Text('Resting electrocardiographic results'),
           sg.OptionMenu(list(["normal", "ST-T wave abnormality"]),
                         size=(20, 8), default_value='Select', key='restecg')],
          [sg.Text('Maximum heart rate achieved'), sg.Input(size=(6, 2), key='thalach')],
          [sg.Text('Exercise induced angina'),
           sg.OptionMenu(list(["Yes", "No"]), size=(4, 8), default_value='Select', key='exang')],
          [sg.Text('ST depression induced by exercise relative to rest'), sg.Input(size=(6, 2), key='oldpeak')],
          [sg.Text('The slope of the peak exercise ST segment'),
           sg.OptionMenu(list(["Up", "Flat", "Down"]), size=(4, 8), default_value='Select', key='slope')],
          [sg.Text('Number of major vessels'),
           sg.OptionMenu(list([0, 1, 2, 3]), size=(4, 8), default_value='Select', key='ca')],
          [sg.Text('Thal'),
           sg.OptionMenu(['normal', 'fixed defect', 'reversible defect'],
                         size=(20, 8), default_value='Select', key='thal')],
          [sg.Button('Submit'), sg.Button('Cancel'), sg.Button('Submit File')], [sg.Menu(menu_def, tearoff=True)]]
window = sg.Window("HeartDisease", layout, size=(720, 500), ttk_theme='#2A2929')

'''
Defining functions'''


def test_data_types(data):
    """
    Test if the dictionary values have acceptable data types.
    """
    # Define the acceptable data types for each key
    expected_types = {
        'age': str,
        'sex': bool,
        'cp': str,
        'trestbps': int,
        'chol': int,
        'fbs': str,
        'restecg': str,
        'thalach': int,
        'exang': str,
        'oldpeak': int,
        'slope': str,
        'ca': int,
        'thal': str
    }

    # Check if each value in the dictionary matches the expected data type
    for key, expected_type in expected_types.items():
        value = data.get(key)
        if not isinstance(value, expected_type):
            raise ValueError(f"Unexpected data type for '{key}': expected {expected_type}, but got {type(value)}")

    # If all values have the expected data types, return True
    return True


def array_convert(tempdict):
    return pd.DataFrame.from_dict(tempdict, orient='index')


def display_data(temp_dict):
    text = ""
    for i in temp_dict:
        text = text + str(i) + ": " + str(temp_dict[i]) + " " + '\n'
    boolean_confirm = sg.popup_ok_cancel("You have entered:", text,
                                         "Please note that each missing inputs reduce classification reliability",
                                         title="Confirm?", font=("Arial Bold", 12))
    return boolean_confirm


def open_file():
    filepath = askopenfilename(
        filetypes=[('Text Files', '*.csv'), ('All Files', '*.*')])
    if not filepath:
        return
    temp_list = []
    with open(filepath, 'r', encoding='utf-8-sig') as file:
        csv_reader = csv.reader(file)

        for row in csv_reader:

            if len(row) != 13:

                return
            for element in row:
                try:
                    temp_list.append(float(element))
                except ValueError:
                    return

    return temp_list


# Transforming the data from GUI into set useable by the predictor dictionary->pd dataframe
# chol,restecg,exang NEED to be converted positive float
def transform_data(dict_to_convert):
    values_ser = pd.Series(dict_to_convert)
    values_df1 = values_ser.to_frame()
    values_df = values_df1.transpose()

    values_df['age'] = pd.to_numeric(values_df['age'], errors='coerce')
    values_df['sex'] = values_df['sex'].replace(['Female', 'Male', 'Select'], [0, 1, np.nan])
    values_df['cp'] = values_df['cp'].replace(
        ['non-anginal pain', 'asymptomatic', 'typical angina', 'atypical angina', 'Select'], [1, 2, 3, 4, np.nan])
    values_df['trestbps'] = pd.to_numeric(values_df['trestbps'], errors='coerce')
    values_df['chol'] = pd.to_numeric(values_df['chol'], errors='coerce')
    values_df['fbs'] = values_df['fbs'].replace(['Lower than 120 mg/dl', 'Higher than 120 mg/dl', 'Select'],
                                                [0, 1, np.nan])
    values_df['restecg'] = values_df['restecg'].replace(['normal', 'ST-T wave abnormality', 'Select'], [0, 1, np.nan])
    values_df['thalach'] = pd.to_numeric(values_df['thalach'], errors='coerce')
    values_df['exang'] = values_df['exang'].replace(['No', 'Yes', 'Select'], [0, 1, np.nan])
    values_df['oldpeak'] = pd.to_numeric(values_df['oldpeak'], errors='coerce')
    values_df['slope'] = values_df['slope'].replace(['Up', 'Flat', 'Down', 'Select'], [1, 2, 3, np.nan])
    values_df['ca'] = pd.to_numeric(values_df['ca'], errors='coerce')
    values_df['thal'] = values_df['thal'].replace(['normal', 'fixed defect', 'reversible defect', 'Select'],
                                                  [3, 6, 7, np.nan])
    # print(values_df)
    return values_df


def value_checker(dict_to_check):
    if (dict_to_check['chol'] != "" and dict_to_check['chol'].isnumeric() is False) or (
            dict_to_check['trestbps'] != "" and dict_to_check['trestbps'].isnumeric() is False) or (
            dict_to_check['oldpeak'] != "" and dict_to_check['oldpeak'].isnumeric() is False) or (
            dict_to_check['thalach'] != "" and dict_to_check['thalach'].isnumeric() is False):
        return 1
    return 0


def df_preparer(dataframe):
    df_crop = dataframe.drop('ca', axis=1)
    df_crop['age'] = df_crop['age'].fillna(54)
    df_crop['sex'] = df_crop['sex'].fillna(1)
    df_crop['cp'] = df_crop['cp'].fillna(4)
    df_crop['trestbps'] = df_crop['trestbps'].fillna(130)
    df_crop['chol'] = df_crop['chol'].fillna(214)
    df_crop['fbs'] = df_crop['fbs'].fillna(0)
    df_crop['restecg'] = df_crop['restecg'].fillna(0)
    df_crop['thalach'] = df_crop['thalach'].fillna(130)
    df_crop['exang'] = df_crop['exang'].fillna(0)
    df_crop['oldpeak'] = df_crop['oldpeak'].fillna(0)
    df_crop['slope'] = df_crop['slope'].fillna(2)
    df_crop['thal'] = df_crop['thal'].fillna(7)
    return df_crop


def dm_predictor(dataframe):

    clf = load('Heart.joblib')
    y_pred = clf.predict(dataframe)
    if y_pred == 0:
        return "Classification: 'Healthy'"
    else:
        return "Classification:'HeartDisease'"


'''Initializing the GUI'''
while True:
    event, values = window.read()
    # End program if user closes window or
    # presses the OK button
    if event in (sg.WIN_CLOSED, 'Exit'):
        break
    elif event == "Cancel":
        break

    elif event == 'Submit':
        values.pop(0)  # Menu bar creates dict element. deleted that.
        ch = display_data(values)
        checker = value_checker(values)
        if checker == 0 and ch == "OK":
            submitted_df = transform_data(values)
            submitted_df_prep = df_preparer(submitted_df)
            result = dm_predictor(submitted_df_prep)
            sg.popup_scrolled(result, title="Result")

        elif checker == 1:
            sg.popup_scrolled("Please insert correct(positive number) to numerical values", title="error")

    elif event == "Submit File":

        try:
            df = pd.DataFrame(open_file(), ATTRIBUTES).transpose()
            submitted_df_prep = df_preparer(df)
            result = dm_predictor(submitted_df_prep)
            sg.popup_scrolled(result, title="Result")
        except Exception as error:
            sg.popup_scrolled(f'Please make sure that CSV follows requirement. \nError:'
                              f' {error} \nPlease, refer to User Manual from the menu.')
    if event == 'Support':
        sg.popup("Support", 'If you need any help please contact via batzoriganar@hallgato.ppke.hu')
    elif event == "User Manual":
        with open('User Manual.txt', 'r') as f:
            file_contents = f.read()

        sg.popup_scrolled(file_contents)
    elif event == 'About...':
        sg.popup('HeartDisease', 'Version 1.0', "The Heart Disease program is designed to help doctors gain more "
                                                "information about a patient's test result based on machine learning.")
window.close()
