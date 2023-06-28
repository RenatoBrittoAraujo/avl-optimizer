import subprocess
import threading

# args = """
# test
# 123
# testa
# """

# process = subprocess.run(["python3", "exp_exec.py"], input=bytes(args, encoding="utf-8"), capture_output=True)

# print(f"return code = {process.returncode}")
# print(f"stdout = <-->\n{process.stdout.decode()}\n<-->")

def run_avl_test(ind):
    avl_input_file = "in.avl"
    avl_output_file = "out.txt"

    commands = """
    LOAD {input_file}
    OPER
    X
    ST
    {output_file}
    """.format(input_file=avl_input_file, output_file=avl_output_file)

    print(f"[{ind}] running thread {ind}")

    process = subprocess.run(["./avl"], cwd="env", input=bytes(commands, encoding="utf-8"), capture_output=True)

    print(f"[{ind}] thread {ind} finished!")
    print(f"[{ind}] return code = {process.returncode}")

    outp = process.stdout.decode()
    if "Enter filename, or <return> for screen output" in outp:
        print(f"[{ind}] success!")
    else:
        print(f"[{ind}] failed!")

def worker(ind):
    # write input file
    # run avl
    run_avl_test(ind)
    # read output file
    # remove input and output file
    # return

n_threads = 1
threads = []

for t_i in range(n_threads):
    t = threading.Thread(target=worker, daemon=True, args=(t_i,))
    threads.append(t)

for thread in threads:
    thread.start()

for thread in threads:
    thread.join()
