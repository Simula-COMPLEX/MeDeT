"""
Created on September 03, 2023

@author: Hassan Sartaj
@version: 1.0
"""

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder


def preprocess_data_sk(file, file_separator):
    df = pd.read_csv(file, sep=file_separator, skipinitialspace=True, on_bad_lines="warn")
    df = df.drop(columns=["response_time"])  # do not need response time for training

    feature_cols = len(df.columns)
    x = df.iloc[:, 0:feature_cols - 1]
    y = df.iloc[:, feature_cols - 1]

    data_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=88, encoded_missing_value=99)
    new_x = data_encoder.fit_transform(x)

    label_encoder = LabelEncoder()
    label_encoder.fit(y)
    sc_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    new_y = label_encoder.transform(y)

    new_df = pd.DataFrame(new_x, columns=df.columns[0:feature_cols - 1])
    new_df = new_df.assign(response_status_code=new_y)

    return new_df, data_encoder, sc_mapping


def preprocess_rawdata(files, file_separator, devices_names):
    processed_files = []
    status_codes_map = {}
    for file in files:
        # write to file
        file_name = file.replace("raw", "pro")
        df, data_encoder, sc_map = preprocess_data_sk(file, file_separator)
        df.to_csv(file_name, sep=file_separator, encoding="utf-8", index=False)
        for device_name in devices_names:
            if device_name in file_name:
                status_codes_map[device_name] = sc_map
                break
        processed_files.append(file_name)

    print("Complete data preprocessing")
    return processed_files, data_encoder, status_codes_map
