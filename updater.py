import os
import sys
import requests
import shutil
import json
from packaging import version
import base64
import logging
from tqdm import tqdm

ADDRESS = "127.0.0.1"
PORT = "8080"
API_VERSION = "v1"

VERSION_URL = f"http://{ADDRESS}:{PORT}/{API_VERSION}/version"

DOWNLOAD_URL = f"http://{ADDRESS}:{PORT}/{API_VERSION}/download"

VERSION = "0.1.0"

def request_exception_handler(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except requests.exceptions.ConnectionError:
            print("Error: Unable to connect to the server. Please check your network or the server URL.")
        except requests.exceptions.Timeout:
            print("Error: The request timed out. The server may be slow, or the network may be congested.")
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")
        except requests.exceptions.RequestException as req_err:
            print(f"An error occurred: {req_err}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
    return wrapper        

def get_current_version() -> str:       
    """ get current version of the application

    Returns:
        str: current version
    """
    try: 
        with open("version.json", "r") as f:
            version = json.load(f)["version"]
    except:
        version = "0.0.0"
    return version


def check_for_updates() -> bool:
    """ check if there is an update available

    Returns:
        bool: True if there is an update, False otherwise
    """
    current_version = get_current_version()
    response = requests.get(VERSION_URL)
    if response.status_code == 200:
        latest_version = response.json()["message"]
        if version.parse(latest_version) > version.parse(current_version):
            print(f"A new version of the application is available: {latest_version}")

            ## update json
            with open("version.json", "w") as f:
                json.dump({"version": latest_version}, f, indent=4)
            return True
    return False

@request_exception_handler
def update(): 
    """ download the latest version of the application
    """
    print("Downloading update...")
    with requests.get(DOWNLOAD_URL) as response:
        r = response
        # r.raise_for_status()
        size = int(r.headers["Content-Length"])
        chunk_size = 8192

        if response.status_code == 200:
            base_path = os.path.join(sys.argv[0])
            tmp_file_name = os.path.join(base_path, "./tmp_file")
            target = os.path.join(base_path, "./diarize")
            n_chunks = (size + chunk_size - 1) // chunk_size
            with open(tmp_file_name, "wb") as f:
                for chunk in tqdm(
                    r.iter_content(chunk_size=chunk_size),
                    total=n_chunks,                # 1
                    unit="KB",                     # 2
                    unit_scale=chunk_size / 1024,  # 3
                ):
                    f.write(chunk)

            os.chmod(tmp_file_name, 0o755)
            print(f"move tmp file from {tmp_file_name} to {target}")
            shutil.move(tmp_file_name, target)
            print("Update downloaded successfully.")

        else:
            print("Failed to download update.")


if __name__ == "__main__":

    if check_for_updates():
        update()
    else:
        print("No updates available.")