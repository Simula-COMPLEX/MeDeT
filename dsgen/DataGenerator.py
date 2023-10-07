"""
Created on September 03, 2023

@author: Hassan Sartaj
@version: 1.0
"""
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
    [res, time_device, res_code] = send_request(method="post", url="",
                                                json_option={"auth": {"login": "", "password": ""}},
                                                print_opt=0)
    if res_code == 200:
        print("Successfully logged in!")
    else:
        print("Invalid login credentials!")


# Data compilation module

def compile_device_data(file_path, file_separator, parameters, d_url, json_obj, print_res=0):

    [res, time_device, httpcode_device] = send_request(method="post", url=d_url, json_option=json_obj,
                                                       print_opt=print_res)

    # print("Received data: \n", res)
    data = ""
    no_data = False
    if "device" in json_obj:
        if "configs" in json_obj["device"]:
            for param in parameters:
                if param in json_obj["device"]["configs"]:
                    d1 = json_obj["device"]["configs"][param]
                    d1 = json.dumps(d1)
                    data += d1.strip("\"") + file_separator
                else:
                    data += file_separator

            data += str(time_device) + file_separator + str(httpcode_device) + "\n"

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
                    time_device) + file_separator + str(httpcode_device) + "\n")

    return res