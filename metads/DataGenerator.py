"""
Created on September 03, 2023

@author: Hassan Sartaj
@version: 1.0
"""
import requests
import time


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
