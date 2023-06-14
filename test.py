import os
import pprint
import re
import requests
import time

# VARIABLES
API_KEY = os.getenv("API_KEY", "my-api-key")
API_URL = "https://rclcapi.lumina247.io"
INFERENCE_STYLE = ['inftoimage'] # accepted values are: inftoimage, inftotext, and textinftotext
MODEL_ID = 39 # '<YOUR_EXISTING_MODEL_ID>' # check your dashboard to get model ids
RESULTS_ZIP_PATH = "results.zip" # Where the results zipfile will be downloaded. Default is current working directory.
TEST_DATASET_NAME = "my-test-dataset-name"
TEST_DATASET_PATH = "<PATH_TO_YOUR_TEST_DATASET_ZIP>" # make sure dataset is a zip file
WAIT_TIME = 5 # polling interval in seconds to check inference session status

def main():
    # pass api key in header
    headers = {'api_key': API_KEY}
    print("Using API key", API_KEY)


    # upload test dataset
    with open(TEST_DATASET_PATH, 'rb') as fdata: 
        files = {'data': fdata}
        params = {'name': TEST_DATASET_NAME} 
        print("Please wait while your dataset is being uploaded to the API...")
        r = requests.post(f"{API_URL}/datasets", headers=headers, files=files, params=params, verify=False)
        data_train = r.json()
        dataset_id = data_train['dataset_id']
        print("Upload is complete. Dataset id is", dataset_id)


    # test model on test dataset
    print("Starting inference session on dataset id", dataset_id)
    r = requests.post(f"{API_URL}/inferences", headers=headers, verify=False, json={
        'dataset_id': dataset_id,
        'model_id': MODEL_ID,
        'inference_style': INFERENCE_STYLE,
    })
    job = r.json()
    job_id = job["job_id"]
    print("Please wait until inference session with job id", job_id, "has finished...")


    # wait until inference session is complete
    print("Continuously polling to get inference session status...")
    pp = pprint.PrettyPrinter(indent=4)
    while True:
        r = requests.get(f"{API_URL}/jobs/{job_id}", headers=headers, verify=False)
        if (job["status"] == 'finished'):
            print("Inference session is complete")
            break
        if job["status"] == "cancelled":
            print("Add model training session cancelled.")
            return
        print("Model training status on", re.sub(r'[^A-Z\d\:\-\s]', '', time.strftime('%Y-%m-%dT%H:%M:%S %p - %Z')))
        pp.pprint(r.json())
        time.sleep(WAIT_TIME)


    # get results
    print("Retrieving results from inference session...")
    r = requests.get(f"{API_URL}/results/{job_id}", headers=headers, verify=False)
    with open(RESULTS_ZIP_PATH, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=128):
            fd.write(chunk)
        print("All done! Check the zip file to view the inference session results")


if __name__ == '__main__':
    main()
