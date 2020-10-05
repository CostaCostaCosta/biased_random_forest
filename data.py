import numpy as np
import pandas as pd


# Load data from path
def load_data(data_path):
    df = pd.read_csv(data_path, low_memory=False)
    return df


# Split data into training/test, naive split justified in notebook
def split_data(data):
    split_number = round(len(data) * .8)
    training, test = data[:split_number].copy(), data[split_number:].copy()
    return training, test


def pima_training_data_transformation(data):
    # Data transformations for pima, full exploration in notebook

    # Glucose Transformation
    gl_median = data['Glucose'].median()
    gl_mask = data['Glucose'] == 0
    data.loc[gl_mask, 'Glucose'] = gl_median

    # Blood Pressure Transformation
    bp_median = data['BloodPressure'].median()
    bp_mask = data['BloodPressure'] == 0
    data.loc[bp_mask, 'BloodPressure'] = bp_median

    # BMI Transformation
    bmi_median = data['BMI'].median()
    bmi_mask = data['BMI'] == 0
    data.loc[bmi_mask, 'SkinThickness'] = bmi_median

    # Skin Thickness Tranformation
    st_array = data['SkinThickness'].to_numpy()
    st_nonzero = st_array[np.nonzero(st_array)]

    # remove zeroes before calculating median -
    st_median = np.median(st_nonzero)
    mask = data['SkinThickness'] == 0
    data.loc[mask, 'SkinThickness'] = st_median

    # Drop Insulin Column
    data = data.drop(columns=['Insulin'])

    medians = {
        "gl_median": gl_median,
        "bp_median": bp_median,
        "bmi_median": bmi_median,
        "st_median": st_median
    }

    return data, medians


def pima_test_data_transformation(data, medians):
    # Data transformation used for test set, using training data medians

    # Glucose Transformation
    gl_mask = data['Glucose'] == 0
    data.loc[gl_mask, 'Glucose'] = medians['gl_median']

    # Skin Thickness Tranformation
    st_mask = data['SkinThickness'] == 0
    data.loc[st_mask, 'SkinThickness'] = medians['st_median']

    # Blood Pressure Transformation
    bp_mask = data['BloodPressure'] == 0
    data.loc[bp_mask, 'BloodPressure'] = medians['bp_median']

    # BMI Transformation
    bmi_mask = data['BMI'] == 0
    data.loc[bmi_mask, 'SkinThickness'] = medians['bmi_median']

    # Drop Insulin Column
    data = data.drop(columns=['Insulin'])

    return data