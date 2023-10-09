"""
Created on September 03, 2023

@author: Hassan Sartaj
@version: 1.0
"""

from dsgen.DataGenerator import compile_device_data
from dsgen.DataPrep import preprocess_rawdata
from dtbuild.OperateDT import DTServer
from mldt.DTModel import train_dtmodel
from dtbuild.BuildDTs import clone_model_storage

devices_names = ["D1", "D2", "D3"]
file_separator = ";"
exp_root = "mldt-exp/"
ml_root_dir = exp_root+"mldt-models/"
st_root_dir = exp_root+"mldt-store/"
in_root_dir = exp_root+"mldt-inputs/"
dataset_files = ["D1.csv", "D2.csv", "D3.csv"]

# hyperparams
maml_lr = 0.05
meta_lr = 0.001
adaption_steps = 1
shots = 1 #1, 2, 5
num_tasks = 10
total_iterations = 5000 # 1000 - for fine-tuning
tps = 256 # 64 - for fine-tuning

ex_name = str(shots)+"shot"
# get from file
dtsnum = 1

if __name__ == '__main__':
    print("Phase 1: Data Generation")
    json_conf = {"device": {"configs": {}}}
    d_url=""
    parameters = []
    file_path = ""
    dts_map = {}
    status_codes_map = {}

    compile_device_data(file_path, file_separator, parameters, d_url, json_conf)

    print("Phase 2: Data Preparation")
    processed_files = preprocess_rawdata(files=dataset_files, file_separator=file_separator)

    print("Phase 3: Meta-learning - Training")
    train_dtmodel(devices_names, parameters, processed_files, ml_root_dir, ex_name, shots, num_tasks, maml_lr, meta_lr, total_iterations, tps, adaption_steps)

    print("Phase 4: Build DTs")
    exp_shots = [ex_name]
    clone_model_storage(devices_names, ml_root_dir, in_root_dir, st_root_dir, dtsnum, exp_shots, json_conf)

    print("Phase 5&6: DTs Request Handling and Device Communication")
    DTServer.init_server(dts_map, status_codes_map, devices_names, parameters)
    DTServer.start_server(host="0.0.0.0", port=5000)

