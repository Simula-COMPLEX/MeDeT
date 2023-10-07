"""
Created on September 05, 2023

@author: Hassan Sartaj
@version: 1.0
"""

import shutil
import os
import json

dts_map = {}


def clone_model(devices_names, ml_root_dir, in_root_dir, st_root_dir, dtsnum, exp_shots, json_conf):
    ml_names = []
    json_object = json.dumps(json_conf, indent=4)
    dts_shot_map = {}
    for dn in devices_names:
        # create device model folders
        model_path = ml_root_dir + dn
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        model_path = ml_root_dir + dn

        sn_path = in_root_dir + dn + "-sn-list-" + str(dtsnum) + ".txt"
        for es in exp_shots:
            # create device shot model folders
            es_path = model_path + "/" + es
            if not os.path.exists(es_path):
                os.mkdir(es_path)

            # create device data folders
            data_path = st_root_dir + dn
            if not os.path.exists(data_path):
                os.mkdir(data_path)
            data_path += "/" + es
            if not os.path.exists(data_path):
                os.mkdir(data_path)
            print("data path: ", data_path)

            sn_list = []
            # iterate all shots folders
            with open(sn_path) as snfile:
                for line in snfile:
                    d_sn = line.rstrip()
                    sn_list.append(d_sn)

            i = 0
            for _ in range(len(sn_list)):
                for ml_file in os.listdir(ml_root_dir + es):
                    if dn in ml_file and i < len(sn_list):
                        src = ml_root_dir + es + "/" + ml_file

                        m_name = ml_file.split(".")[0]
                        ml_names.append(m_name)
                        # creating model copy
                        dtm_path = es_path + "/" + m_name + "_" + sn_list[i] + ".pth"
                        shutil.copy2(src, dtm_path)

                        # creating json data file
                        df_path = data_path + "/" + m_name + "_data_" + sn_list[i] + ".json"
                        with open(df_path, "w") as outfile:
                            outfile.write(json_object)

                        dts_shot_map[sn_list[i] + "-" + m_name] = [dtm_path, df_path]
                        i += 1

            dts_map[dn + "-" + es] = dts_shot_map
