import itertools
import os
from pathlib import Path

import requests
from dotenv import load_dotenv
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

load_dotenv()

max_workers = 4
destination = Path(os.getenv("LEELA_DATA_PATH").removesuffix("/*.tar"))

base_url = "https://storage.lczero.org/files/training_data/test80/"
dates = ["20240818", "20240819"]
times = [f"{i:02d}17" for i in range(24)]
filenames = [
    f"training-run1-test80-{date}-{time}.tar"
    for date, time in itertools.product(dates, times)
]


def download_file(filename: str):
    response = requests.get(base_url + filename)

    with open(destination / filename, "wb") as f:
        f.write(response.content)


missing_filenames = [
    filename
    for filename in tqdm(filenames)
    if not os.path.exists(destination / filename)
]
thread_map(download_file, missing_filenames, max_workers=max_workers)
