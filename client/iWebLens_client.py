# Generate the parallel requests based on the ThreadPool Executor
from concurrent.futures import ThreadPoolExecutor as PoolExecutor
import sys
import time
import glob
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import threading
import uuid
import base64
import json
import os
import random


# Show all out put unless experimenting
TESTING = False


def call_object_detection_service(image):
    """Prepare and make request to web service.

    :param image: Source to perform object detection on.
    """
    global TESTING

    try:
        url = str(sys.argv[2])
        data = {}
        # generate uuid for image
        ID = uuid.uuid5(uuid.NAMESPACE_OID, image)
        # Encode image into base64 string
        with open(image, 'rb') as image_file:
            data['image'] = base64.b64encode(image_file.read()).decode('utf-8')

        # Prepare the request uuid and headers
        data['id'] = str(ID)
        headers = {'Content-Type': 'application/json'}

        # Create a retry backoff since the server is small and sometimes
        # rejects too many requests from the same ip
        session = requests.Session()
        # retry = Retry(connect=3, backoff_factor=random.uniform(0, 1))
        # adapter = HTTPAdapter(max_retries=retry)
        # session.mount('http://', adapter)
        # session.mount('https://', adapter)

        # Perform http request with data and headers
        response = session.post(
            url, json=json.dumps(data), headers=headers)

        if response.ok:
            if not TESTING:
                # Display the thread and related image uuid
                output = "Thread : {},  input image: {}".format(
                    threading.current_thread().getName(), image)
                print(output)
                # Format and display the json object returned to human readable text
                formatted_results = json.loads(response.text)
                print(json.dumps(formatted_results, indent=4))
        else:
            # Describe error, try resolve with retry or continue
            print("Error, response status:{}\n TRYING AGAIN...".format(response))

    except Exception as e:
        # Raise unresolvable error
        print("Exception in web_service call: {}".format(e))


def get_images_to_be_processed(input_folder) -> list:
    """Get individual image paths to process.

    :param input_folder: Path to dir of images to run detection on
    :rtype: list, All image paths
    """

    # gets list of all images path from the input folder
    images = []
    for image_file in glob.iglob(input_folder + "*.jpg"):
        images.append(image_file)
    return images


def run_experiments():
    """
    Run a specific set of worker-pod combinations several times to test 
    web service performance and save out the results as a CSV.
    """
    import pandas as pd
    from tqdm import tqdm

    global TESTING
    # Disable output for each call but allow error output
    TESTING = True

    # Fetch arguments for input and set run flags manually
    input_folder = os.path.join(sys.argv[1], "")
    images = get_images_to_be_processed(input_folder)
    num_images = images.__len__()
    # at most max_workers threads to execute calls asynchronously
    num_workers = [1, 6, 11, 16, 21, 26, 31]
    pods = [1, 2, 3]  # How many pod replicas are in the activate cluster
    n_tests = 3  # How many times to run each configuration for more accuracy
    print("Experiment mode on, running tests for pods {} and workers {}, each a total of {} times.".format(
        pods, num_workers, n_tests))

    # Table using pandas for recording results
    cols = ['pod_count', 'client_threads', 'avg_response']
    df = pd.DataFrame(columns=cols)

    for pod in pods:
        # Initalise and increment progress bar
        for num_worker in tqdm(num_workers):
            aggregate_average_response = 0  # Store each avg time for n_tests
            # Repeat test for worker-pod combo n_test times for more representative results
            for i in range(n_tests):
                start_time = time.time()
                with PoolExecutor(max_workers=num_worker) as executor:
                    for _ in executor.map(call_object_detection_service,  images):
                        pass
                elapsed_time = time.time() - start_time
                aggregate_average_response += elapsed_time / \
                    num_images  # Update total average time
                # Add some sleep to prevent small server overload

            # Create a data row and append it to the results table
            row = pd.DataFrame(
                [[pod, num_worker, aggregate_average_response/n_tests]], columns=cols)
            df = df.append(row)

        # Wait indefinitely for user to change deployment
        _ = input(
            "Please increment total pod replicas to {}. Sleeping, press enter to wake.".format(pod+1))

    # Save results table as CSV
    df.to_csv('experiments.csv', index=False)


def main() -> int:
    """main path of execution, direct to experiments or regular functions.

    :rtype: int, Return 0 if completed without errors 
    """
    # Provide arguments-> input folder, url, number of workers
    if len(sys.argv) != 4:
        raise ValueError("Arguments list is wrong. Please use the following format: {} {} {} {}".
                         format("python iWebLens_client.py", "<input_folder>", "<URL>", "<number_of_workers>"))

    if int(sys.argv[3]) == 0:
        # If client thread count arg is 0
        run_experiments()
        return 0

    # Get arguments for inputs and run flags
    input_folder = os.path.join(sys.argv[1], "")
    images = get_images_to_be_processed(input_folder)
    num_images = images.__len__()
    num_workers = int(sys.argv[3])

    start_time = time.time()
    # Create a worker  thread  to  invoke the requests in parallel
    with PoolExecutor(max_workers=num_workers) as executor:
        for _ in executor.map(call_object_detection_service,  images):
            pass
    elapsed_time = time.time() - start_time
    print("Total time spent: {} average response time: {}".format(
        elapsed_time, elapsed_time/num_images))
    return 0


if __name__ == "__main__":
    main()
