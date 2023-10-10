"""
Created on September 03, 2023

@author: Hassan Sartaj
@version: 1.0
"""

from dsgen.DataGenerator import generate_exdevice_data
from dsgen.DataPrep import preprocess_rawdata
from dtbuild.OperateDT import DTServer
from mldt.DTModel import train_dtmodel, finetune_dtmodel
from dtbuild.BuildDTs import clone_model_storage

devices_names = ["D1", "D2", "D3"]
file_separator = ";"
exp_root = "mldt-exp/"
ml_root_dir = exp_root + "dt-models/"
st_root_dir = exp_root + "dt-store/"
in_root_dir = exp_root + "dt-inputs/"
ds_root_dir = exp_root + "datasets/"

dataset_files = [ds_root_dir + "D1_rawdata.csv", ds_root_dir + "D2_rawdata.csv", ds_root_dir + "D3_rawdata.csv"]

# hyperparams
maml_lr = 0.05
meta_lr = 0.001
adaption_steps = 1
shots = 1  # 1, 2, 5
num_tasks = 2
total_iterations = 2000  # 5000 - for training, & 1000 - for fine-tuning
tps = 32  # 64 - for training & 64 - for fine-tuning

ex_name = str(shots) + "shot"
# get from file
dtsnum = 1
dts_map = {}
status_codes_map = {}

if __name__ == '__main__':
    print("=> Phase 1: Data Generation - Example Device")
    json_conf = {"device": {"configs": {}}}
    d_url = "http://0.0.0.0:6001/exdevice/" + devices_names[0]
    parameters = ["p1", "p2", "p3", "p4", "p5", "p6"]
    file_path = dataset_files[0]
    generate_exdevice_data(dataset_files[0], parameters, file_separator, d_url, total_time=2, delay=3)

    print("=> Phase 2: Data Preparation")
    processed_files, data_encoder, status_codes_map = preprocess_rawdata(files=[dataset_files[0]],
                                                                         file_separator=file_separator,
                                                                         devices_names=[devices_names[0]])

    print("=> Phase 3: Meta-learning - Training")
    train_dtmodel([devices_names[0]], parameters, processed_files, status_codes_map, ml_root_dir, ex_name, shots,
                  num_tasks, maml_lr, meta_lr, total_iterations, tps, adaption_steps)

    # print("=> Phase 3: Meta-learning - Fine-Tuning")
    # finetune_dtmodel([devices_names[0]], parameters, processed_files, status_codes_map, ml_root_dir, ex_name,
    #                   shots, num_tasks, maml_lr, meta_lr, total_iterations, tps, adaption_steps)

    print("=> Phase 4: Build DTs")
    exp_shots = [ex_name]
    dts_map = clone_model_storage([devices_names[0]], ml_root_dir, in_root_dir, st_root_dir, dtsnum, exp_shots, json_conf)

    print("=> Phase 5&6: Operating DTs")  # Request Handling and Device Communication
    DTServer.init_server(dts_map, status_codes_map, [devices_names[0]], parameters, data_encoder)
    DTServer.start_server(host="0.0.0.0", port=5000)
