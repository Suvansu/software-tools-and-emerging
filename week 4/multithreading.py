import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import math
import time

# Load a sample open dataset (NYC Taxi Trips)
url = ""
df = pd.read_csv("yellow_tripdata_2019-01.csv", nrows=100000)  # Load first 100k rows for demo

# Simulate an expensive row-wise operation
def expensive_operation(row):
    # Example: complex computation per row
    return math.sqrt(row['trip_distance'] ** 2 + row['total_amount'] ** 2)

# ------------------ Single-threaded ------------------
start = time.time()
results_single = df.apply(expensive_operation, axis=1)
end = time.time()
print("Single-threaded execution time:", end - start, "seconds")

# ------------------ Multithreaded using ThreadPoolExecutor ------------------
# Split DataFrame into chunks
n_chunks = 5
chunks = [df.iloc[i:i + len(df)//n_chunks] for i in range(0, len(df), len(df)//n_chunks)]

def process_chunk(chunk):
    return chunk.apply(expensive_operation, axis=1)

start = time.time()
with ThreadPoolExecutor(max_workers=n_chunks) as executor:
    results_chunks = list(executor.map(process_chunk, chunks))

# Combine results
results_parallel = pd.concat(results_chunks)
end = time.time()
print("Multithreaded execution time:", end - start, "seconds")
