import os
import pprint
import re
import requests
import time

# VARIABLES
API_KEY = os.getenv("API_KEY", "my-api-key")
API_URL = "https://rclcapi.lumina247.io"
BOXDOWN = 0 # uses pixel average when folding an image into a smaller area. Set to 0 or any positive integer. The larger your image, the larger values RCLC can tolerate.
CHANNEL_PICK = "combine" # choose a RGB channel, or combination, to train on. Allowed values are: "red", "green", "blue", "avg" and "combine"
DATASET_NAME = "my-dataset-name"
EVAL_TYPE = "naive-bayes" # allowed evaluation types are: "naive-bayes", "chi-squared", "chi-squared-dist", and "fractal"
IMAGINARY = False # collapses low frequency patterns into an imaginary pattern and use this as part of the given evaluation method selected
MODEL_NAME = "my-model-name"
RCL_TICKS = 10 # how granular RCLC should interpret a value from channelpick
RESULTS_ZIP_PATH = "results.zip" # Where the results zipfile will be downloaded. Default is current working directory.
TEST_SIZE = 0.2  # splits data into a test set. Set to 0 to train only with no test. You can pass TEST_SIZE or TEST_DATASET_ID, but not both. Comment out test_size if you are passing in test_dataset_id.
TEST_DATASET_ID =  "<YOUR_EXISTING_DATASET_ID>" # uses an existing dataset for testing. You can pass TEST_SIZE or TEST_DATASET_ID, but not both. Comment out test_dataset_id if you are passing in test_size.
TRAINING_DATASET_PATH = "<PATH_TO_YOUR_DATASET_ZIP>"  # make sure dataset is a zip file
WAIT_TIME = 5  # polling interval in seconds to check model training status


def main():
    # pass api key in header
    headers = {"api_key": API_KEY}
    print("Using API key", API_KEY)

    # upload training dataset
    with open(TRAINING_DATASET_PATH, "rb") as fdata:
        files = {"data": fdata}
        params = {"name": DATASET_NAME}
        print("Please wait while your dataset is being uploaded to the API...")
        r = requests.post(f"{API_URL}/datasets", headers=headers, files=files, params=params, verify=False)
        data_train = r.json()
        print("Upload is complete. Dataset id is", data_train["dataset_id"])

    # train new model
    print("Starting new model training session...")
    jsonRequest = {
        "dataset_id": data_train["dataset_id"],
        "strategy": "new",
        "name": MODEL_NAME,
        "test_size": TEST_SIZE, # You can pass TEST_SIZE or TEST_DATASET_ID, but not both. Comment out test_size if you are passing in test_dataset_id.
        #"test_dataset_id": TEST_DATASET_ID, # You can pass TEST_SIZE or TEST_DATASET_ID, but not both. Comment out test_dataset_id you are passing in test_size.
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
        "Please wait until model training session with job id",
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
                "Model training is complete! Check your dashboard to get the new model id."
            )
            break
        print(
            "Model training status on",
            re.sub(r"[^A-Z\d\:\-\s]", "", time.strftime("%Y-%m-%dT%H:%M:%S %p - %Z")),
        )
        if job["status"] == "cancelled":
            print("Add model training session cancelled.")
            return
        pp.pprint(job)
        time.sleep(WAIT_TIME)

    # get results if model training and evaluation session
    if "test_size" in jsonRequest or "test_dataset_id" in jsonRequest:
        print("Retrieving results from model training and evaluation...")
        r = requests.get(f"{API_URL}/results/{job_id}", headers=headers, verify=False)
        with open(RESULTS_ZIP_PATH, 'wb') as fd:
            for chunk in r.iter_content(chunk_size=128):
                fd.write(chunk)
            print(f"All done! Your results are located in {RESULTS_ZIP_PATH}")


if __name__ == "__main__":
    main()
