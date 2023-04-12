import subprocess


def write_file(target, content):
    with open(target, "w") as f:
        f.write(content)


def read_file(target):
    txt = ""
    with open(target) as f:
        txt = f.read()
    return txt


def delete_file(target):
    subprocess.call("rm " + target)

# interact with program itself


class AVL():

    def __init__(self):
        pass

    def send_command(self, cmd) -> bool:
        # exec("sei la")
        pass

    def read_last(self) -> str:
        # exec("sei la")
        pass

    def analyse(self, file: str) -> str:
        write_file("temp_input.txt", file)

        self.send_command("LOAD temp_input.txt")
        self.send_command("OPER")
        self.send_command("X")
        self.send_command("ST")
        self.send_command("temp_output.txt")

        res = read_file("temp_output.txt")

        delete_file("temp_input.txt")
        delete_file("temp_output.txt")

        return res

# reader writer


class AVLFileParser():

    def __init__(self, arquivo: str):
        pass

    def parse_into_file(self) -> str:
        pass

    def get_value(self, key: str):
        pass

    def set_value(self, key: str, value: str):
        pass

# reader


class AVLResultParser():

    def __init__(self, arquivo: str = ""):
        pass

    def parse_into_file(self) -> str:
        pass

    def get_value(self, key: str) -> str:
        pass

    def get_values(self, keys: list[str]) -> list[str]:
        pass

    def set_template(self, org: 'AVLResultParser'):
        pass


class Evaluator():
    optmize_target = "cl"
    optmizing_parameters = ["corda", "envergadura"]
    last = []
    v = []

    def __init__(self, optmizing_parameters, optimize_target):
        self.optmizing_parameters = optmizing_parameters
        self.optmize_target = optimize_target

    def set_results(self, fp: AVLResultParser):
        nv = []
        for param in self.optmizing_parameters:
            nv.append(fp.get_value(param))
        self.v.append(nv)

    def is_max(self, err=1e-6) -> bool:
        pass

    def get_next_params(self) -> list[any]:
        # calculate highest value since last
        # set last as new target
        # calculate best diffs to investigate around new target
        # return each diff
        pass


def main():

    avl = AVL()
    evaluator = Evaluator(
        [
            "corda",
            "envergadura"
        ],
        "cl"
    )

    fp = AVLFileParser(read_file("geometria.txt"))

    # use multidimentional set thing to avoid going similar values?
    queue = set()
    queue.add(
        [
            fp.get_values(evaluator.optmizing_parameters)
        ]
    )

    iter_count = 0
    MAX_ITER_COUNT = 1e3

    tgt = None

    while True:
        iter_count += 1
        if iter_count > MAX_ITER_COUNT:
            break

        params = queue.pop()

        new_fp = AVLFileParser()
        new_fp.set_template(fp)

        for param in params:
            new_fp.set_value(param[0], param[1])
        file = new_fp.parse_into_file()

        res = avl.analyse(file)
        rp = AVLResultParser(res)

        evaluator.set_results(rp)

        if evaluator.is_max(1e-6):
            tgt = fp
            break
        else:
            nextp = evaluator.get_next_params()
            for param in nextp:
                queue.add(param)

    if tgt is not None:
        f = fp.parse_into_file()
        write_file("geometria_otima.txt", f)
        print("success, optimal configuration in \"geometria_otima.txt\"")
    else:
        print("error: failed to find optimal configuration")
