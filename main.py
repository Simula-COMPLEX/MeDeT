"""
Created on September 03, 2023

@author: Hassan Sartaj
@version: 1.0
"""

from metads.DataGenerator import send_request

if __name__ == '__main__':
    print("Phase 1: Data Generation")
    json_conf = {"device": {"configs": {}}}
    [res, time_device, httpcode_device] = send_request(method="get", url="", json_option=json_conf,
                                                       print_opt=0)
    print("Received data: \n", res)