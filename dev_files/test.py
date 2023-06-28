import threading
from time import sleep
import random

def worker(results, tid):
    # Do some work
    # Set the result when finished
    result = f"RESULT<{tid}>"
    sleep(random.randint(1, 5)/10)
    results[tid] = result

def main():
    # Create a queue to store the results
    results = {}

    # Create some worker threads
    threads = []
    for i in range(5):
        t = threading.Thread(target=worker, args=(results,i))
        threads.append(t)
        t.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    # Retrieve the results from the queue
    while len(results) > 0:
        key, result = results.popitem()
        print(key, result)

if __name__ == '__main__':
    main()
