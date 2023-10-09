"""
Created on September 08, 2023

@author: Hassan Sartaj
@version: 1.0
"""

from flask import Flask, request
from flask_restful import Resource, Api
import json
import torch

# ------------------------
# Create DT Server & APIs
# ------------------------
# creating the flask app
app = Flask(__name__.split(".")[0])
# creating an API object
api = Api(app)
dts_map = {}
status_codes_map = {}
devices_names = []
params = []


def get_dtmodel_response(dtm_path, edata):
    with torch.no_grad():
        t_model = torch.load(dtm_path)
        t_model.eval()
        input = torch.tensor(edata, dtype=torch.float64)
        input = input.view(1, len(edata))
        output = t_model(input)
        prediction = int(torch.max(output.data, 1)[1].numpy())
        return prediction


def decode_json_object(json_obj, parameters):
    data_values = []
    return data_values


# ---------------------
# DTs Request Handler
# ---------------------

class RequestHandler(Resource):
    def get(self, num, name, shot, model):
        print("Received Get Request for  Device # " + str(num), " - name: ", name, " - shot: ", shot, " - model: ",
              model)
        f_key = name + "-" + shot
        s_key = str(num) + "-" + model
        if dts_map is None or f_key not in dts_map.keys():
            response = {"Error": "No data available for Device # " + str(num)}
            return response, 503
        else:
            # return data from DT storage
            dtn_map = dts_map[f_key]
            _, df_path = dtn_map[s_key]

            with open(df_path, "r") as openfile:
                json_obj = json.load(openfile)
                print("Data for  Device # " + str(num) + "\n")
                print(json_obj)
            return json_obj, 200

    def post(self, num, name, shot, model):
        print("Received Get Request for  Device # "+str(num), " - name: ", name, " - shot: ", shot, " - model: ", model)
        f_key = name + "-" + shot
        s_key = str(num) + "-" + model
        if dts_map is None or f_key not in dts_map.keys():
            response = {"Error": "Device # " + str(num) + " Not Found!"}
            return response, 503
        else:
            data = request.get_json()
            dtn_map = dts_map[f_key]
            dtm_path, df_path = dtn_map[s_key]
            for dev_n in devices_names:
                if dev_n in df_path:
                    device_name = dev_n
                    break

            edata = decode_json_object(data, params)
            prediction = get_dtmodel_response(dtm_path, edata)
            sc_map = status_codes_map[device_name]
            inv_status_codes_map = {v: k for k, v in sc_map.items()}
            status_code = inv_status_codes_map[prediction]
            if status_code == 200:
                with open(df_path, "w") as outfile:
                    json_obj = json.dumps(data, indent=4)
                    outfile.write(json_obj)
                response = {"Message": "Updated Settings of Device # " + str(num)}
            else:
                response = data
            return response, int(status_code)


# ------------------------
# DT Communication Server
# ------------------------
class DTServer:
    @staticmethod
    def init_server(dt_map, sc_map, dev_names, devparams):
        api.add_resource(RequestHandler, "/devdt/<string:dtname>/<string:shot>/<int:num>/<string:dtmodel>")
        global dts_map, status_codes_map, devices_names, params
        dts_map = dt_map
        status_codes_map = sc_map
        devices_names = dev_names
        params = devparams
        # TODO: add link to physical device

    @staticmethod
    def start_server(host, port):
        print("Starting DT Server at host:: {}".format(host) + " - port:: {}".format(str(port)))
        from waitress import serve
        serve(app, host=host, port=port)
        app.run(debug=True)



