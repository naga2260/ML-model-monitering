import argparse
import time

import pandas as pd
import requests


# -----------------------------------------
# Configuration
# -----------------------------------------

API_URL = "http://127.0.0.1:8000/api/v1/predict"

# Your uploaded SPF dataset
DATASET_PATH = "data/SPF.csv"

# Target column — DO NOT send this to the API
TARGET_COLUMN = "Exam_Score"


# -----------------------------------------
# Send requests
# -----------------------------------------

def send_requests(
    dataset_path: str,
    api_url: str,
    number_of_requests: int,
    delay: float = 0.1
):

    # Load dataset
    df = pd.read_csv(dataset_path)

    print(f"Dataset loaded: {df.shape}")

    # Check requested number
    if number_of_requests > len(df):

        print(
            f"Requested {number_of_requests} requests, "
            f"but dataset only contains {len(df)} rows."
        )

        print(
            "Sampling with replacement..."
        )

        samples = df.sample(
            n=number_of_requests,
            replace=True
        )

    else:

        # Randomly select students
        samples = df.sample(
            n=number_of_requests,
            random_state=42
        )

    successful = 0
    failed = 0

    print(
        f"\nSending {number_of_requests} requests..."
    )

    for index, (_, row) in enumerate(
        samples.iterrows(),
        start=1
    ):

        # Remove target column
        input_data = row.drop(
            labels=[TARGET_COLUMN]
        ).to_dict()

        try:

            response = requests.post(
                api_url,
                json=input_data,
                timeout=10
            )

            if response.status_code == 200:

                successful += 1

                result = response.json()

                print(
                    f"[{index}/{number_of_requests}] "
                    f"SUCCESS | "
                    f"Prediction: "
                    f"{result['predicted_exam_score']}"
                )

            else:

                failed += 1

                print(
                    f"[{index}/{number_of_requests}] "
                    f"FAILED | "
                    f"Status: {response.status_code}"
                )

        except requests.RequestException as exc:

            failed += 1

            print(
                f"[{index}/{number_of_requests}] "
                f"ERROR | {exc}"
            )

        # Small delay between requests
        time.sleep(delay)

    print("\n-----------------------------")
    print("Request simulation completed")
    print("-----------------------------")
    print(f"Total requests : {number_of_requests}")
    print(f"Successful     : {successful}")
    print(f"Failed         : {failed}")


# -----------------------------------------
# Command-line interface
# -----------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Send simulated production requests to FastAPI"
    )

    parser.add_argument(
        "--requests",
        type=int,
        default=100,
        help="Number of requests to send"
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="Delay between requests in seconds"
    )

    args = parser.parse_args()

    send_requests(
        dataset_path=DATASET_PATH,
        api_url=API_URL,
        number_of_requests=args.requests,
        delay=args.delay
    )