import os
import pprint
import re
import requests
import time

# VARIABLES
ADD_MODEL_NAME = "my-add-model-name"
API_KEY = os.getenv("API_KEY", "my-api-key")
API_URL = "https://rclcapi.lumina247.io"
BASE_MODEL_ID = "<YOUR_EXISTING_MODEL_ID>"  # check your dashboard to see model ids
BOXDOWN = 2 # uses pixel average when folding an image into a smaller area. Set to 0 or any positive integer. The larger your image, the larger values RCLC can tolerate.
CHANNEL_PICK = "combine" # choose a RGB channel, or combination, to train on. Allowed values are: "red", "green", "blue", "avg" and "combine"
EVAL_TYPE = "naive-bayes" # allowed evaluation types are: "naive-bayes", "chi-squared", "chi-squared-dist", and "fractal"
IMAGINARY = False # collapses low frequency patterns into an imaginary pattern and use this as part of the given evaluation method selected
EXTRA_MODEL_IDS = [1, 2]  # change this list to your extra model ids
RCL_TICKS = 3 # how granular RCLC should interpret a value from channelpick
RESULTS_ZIP_PATH = "results.zip" # Where the results zipfile will be downloaded. Default is current working directory.
TEST_SIZE = 0.1  # splits data into a test set. Set to 0 to train only with no test.
WAIT_TIME = 5  # polling interval in seconds to check add model training status


def main():
    # pass api key in header
    headers = {"api_key": API_KEY}
    print("Using API key", API_KEY)

    # add models
    # specifies a set of models to be added to a base model
    print("Starting add models training session...")
    jsonRequest = {
        "base_model_id": BASE_MODEL_ID,
        "extra_model_ids": EXTRA_MODEL_IDS,
        "strategy": "add",
        "name": ADD_MODEL_NAME,
        "test_size": TEST_SIZE,
        "eval_type": EVAL_TYPE,
        "channel_pick": CHANNEL_PICK,
        "rcl_ticks": RCL_TICKS,
        "boxdown": BOXDOWN,
        "imaginary": IMAGINARY
    }
    r = requests.post(f"{API_URL}/models", headers=headers, json=jsonRequest, verify=False)
    print(r.json())
    job_id = r.json()["job_id"]
    print(
        "Please wait until add model training session with job id",
        job_id,
        "has finished...",
    )

    # wait until model training session is complete
    print("Continuously polling to get model training status...")
    pp = pprint.PrettyPrinter(indent=4)
    job = None
    while True:
        r = requests.get(f"{API_URL}/jobs/{job_id}", headers=headers, verify=False)
        job = r.json()
        if job["status"] == "finished":
            print(
                "Add model training is complete! Check your dashboard to get the new model id."
            )
            break
        if job["status"] == "cancelled":
            print("Add model training session cancelled.")
            return
        print(
            "Model training status on",
            re.sub(r"[^A-Z\d\:\-\s]", "", time.strftime("%Y-%m-%dT%H:%M:%S %p - %Z")),
        )
        pp.pprint(job)
        time.sleep(WAIT_TIME)

    
    # get results if model training and evaluation session
    if "test_size" in jsonRequest: 
        print("Retrieving results from add model training and evaluation session...")
        r = requests.get(f"{API_URL}/results/{job_id}", headers=headers, verify=False)
        with open(RESULTS_ZIP_PATH, 'wb') as fd:
            for chunk in r.iter_content(chunk_size=128):
                fd.write(chunk)
            print(f"All done! Your results are located in {RESULTS_ZIP_PATH}")


if __name__ == "__main__":
    main()
