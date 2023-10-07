"""
Created on September 03, 2023

@author: Hassan Sartaj
@version: 1.0
"""

from dsgen.DataGenerator import compile_device_data

file_separator = ";"

if __name__ == '__main__':
    print("Phase 1: Data Generation")
    json_conf = {"device": {"configs": {}}}
    d_url=""
    parameters = []
    file_path = ""
    compile_device_data(file_path, file_separator, parameters, d_url, json_conf)
