"""
Created on September 03, 2023

@author: Hassan Sartaj
@version: 1.0
"""
import random

import requests
import time
import json


# ------------------------------
# Make Connection with HIoT App
# ------------------------------

# Send HTTP requests
def send_request(method, url, json_option=None, print_opt=0):
    print("Request URL: ", url)
    if method == "get":
        response = s.get(url, params=json_option)
    elif method == "post":
        try:
            response = s.post(url=url, json=json_option)
        except Exception as e:
            print("Got %s error %s, retrying" % (type(e).__name__, e))
            time.sleep(2)
            response = s.post(url=url, json=json_option)
    elif method == "delete":
        response = s.delete(url, params=json_option)

    print("Response time:", response.elapsed)
    print("Status Code:", response.status_code)

    try:
        response_data = response.json()
        if print_opt == 1:
            print(response_data)
            # display(response.headers)
    except ValueError:
        response_data = response
        print("JSONDecodeError! Returning object as it is.")

    return response_data, response.elapsed.total_seconds(), response.status_code


# Login
with requests.Session() as s:
    print("Logging in...")
    # -> provide login credentials to send request to device
    # res_obj, res_time, res_code = send_request(method="post", url="",
    #                                             json_option={"auth": {"login": "", "password": ""}},
    #                                             print_opt=0)
    # if res_code == 200:
    #     print("Successfully logged in!")
    # else:
    #     print("Invalid login credentials!")


# -------------------------
# Data compilation module
# -------------------------
def compile_device_data(file_path, file_separator, parameters, d_url, json_obj, print_res=0):
    # res_obj, res_time, res_code = send_request(method="post", url=d_url, json_option=json_obj,
    #                                                        print_opt=print_res)
    # print("Received data: \n", res_obj)

    # -> use res_time, res_code if real device is connected
    # random response code and time
    res_code = random.choice([200, 503])
    res_time = random.uniform(2, 4)

    data = ""
    no_data = False
    # -> use res_obj below if real device is connected
    if "device" in json_obj:
        if "configs" in json_obj["device"]:
            for param in parameters:
                if param in json_obj["device"]["configs"]:
                    d1 = json_obj["device"]["configs"][param]
                    d1 = json.dumps(d1)
                    data += d1.strip("\"") + file_separator
                else:
                    data += file_separator

            data += str(res_time) + file_separator + str(res_code) + "\n"

            with open(file_path, "a") as file:
                file.write(data)
        else:
            no_data = True
            print("No device configs")
    else:
        no_data = True
        print("No device")

    if no_data:
        with open(file_path, "a") as file:
            file.write(
                file_separator + file_separator + file_separator + file_separator + file_separator + file_separator + file_separator + file_separator + file_separator + file_separator +
                file_separator + file_separator + file_separator + file_separator + file_separator + file_separator + file_separator + file_separator + str(
                    res_time) + file_separator + str(res_code) + "\n")


# --------------------------------------
# Data generation for an example device
# --------------------------------------
def generate_exdevice_data(device_ds_file, parameters, file_separator, d_url, total_time, delay):
    header = file_separator.join(
        str(p) for p in parameters) + file_separator + "response_time" + file_separator + "response_code "

    with open(device_ds_file, "w") as file:
        file.write(header + "\n")

    print("Generating data for example device...\nRunning for {} minutes".format(total_time))
    t_end = time.time() + total_time * 60
    i = 0
    while time.time() < t_end:
        if i % 2 == 0:
            # random outbound values
            p1 = random.choice(
                [random.randint(-15, -1), random.randint(101, 300), None, "null", " ", "XX", "0", "45", "-900"])
            p2 = random.choice(["XX", " ", "-", "0", "x1y3z", "0", "145", "-110"])
            p3 = random.choice([bool(random.getrandbits(1)), None, "None", " ", 7, "0", "56", "-40"])
            p4 = random.choice(
                [random.randint(-15, 9), random.randint(250, 500), None, "null", " ", "XX", "0", "34", "-50"])
            p5 = random.choice([bool(random.getrandbits(1)), None, "null", " ", 3, "0", "34", "-30"])
            p6 = random.choice([bool(random.getrandbits(1)), None, "null", " ", 1, "0", "45", "-30"])
        else:
            # random normal range values
            p1 = random.randint(0, 100)  # 0-100
            p2 = random.choice(["x1", "x2", "x3", "x4"])
            p3 = bool(random.getrandbits(1))  # Boolean
            p4 = random.randint(10, 250)  # 10-250
            p5 = bool(random.getrandbits(1))  # Boolean
            p6 = bool(random.getrandbits(1))  # Boolean

        i += 1

        json_obj = {"device": {
            "configs": {
                "p1": p1,
                "p2": p2,
                "p3": p3,
                "p4": p4,
                "p5": p5,
                "p6": p6,
            }}}

        print("Compiling rawdata for example device...")
        compile_device_data(device_ds_file, file_separator, parameters, d_url, json_obj, 0)
        time.sleep(delay)  # delay
    print("Generated raw data for example device")
