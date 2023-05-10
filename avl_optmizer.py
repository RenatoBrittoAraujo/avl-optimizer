import subprocess
from os.path import exists
import copy
import json

from avl_parse_util import parse_avl_file

# em teoria, isso não deveria ser mudado. se vc precisa
# mudar isso, é porque não tá usando os parametros do programa
AVL_ENV_PATH = "env/"
AVL_INPUT_FILE = "t1.avl"
AVL_OUTPUT_FILE = "out.txt"
AVL_INPUT_KEYWORDS = [
    "SECTION", "SURFACE", "AFILE", "CLAF", "COMPONENT", "TRANSLATE", "ANGLE", "SCALE", "YDUPLICATE", "CONTROL", "NOWAKE"
]


def to_float(v: str) -> tuple[bool, float]:
    try:
        return True, float(v)
    except Exception:
        return False, 0.0


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
    command_list: str
    avl_folder_path: str
    input_file: str
    output_file: str
    overwrite_any: bool

    def __init__(self, avl_folder_path: str, output_file: str, input_file: str, overwrite_any: bool = False):
        if not avl_folder_path.endswith("/"):
            avl_folder_path += "/"
        if not input_file.endswith(".avl"):
            raise Exception("error: file '" +
                            input_file + "' is not .avl")

        self.avl_folder_path = avl_folder_path
        self.input_file = input_file
        self.output_file = output_file
        self.command_list = ""
        self.overwrite_any = overwrite_any

    def exec(self):
        inp = self.avl_folder_path + self.input_file
        out = self.avl_folder_path + self.output_file

        if exists(out):
            if self.overwrite_any:
                delete_file(out)
            else:
                raise Exception("error: file '" + out + "' is not .avl")

        if not exists(inp):
            raise Exception("error: avl input file " + inp + " does not exist")

        if not self.command_list:
            raise Exception("error: no avl commands have been provided")

        return subprocess.run(
            "./avl", cwd="env", input=bytes(self.command_list, encoding="utf-8"))

    def add_command(self, cmd: str):
        self.command_list += cmd + '\n'

    def analyse_v1(self) -> str:
        commands = """
        LOAD {input_file}
        OPER
        X
        ST
        {output_file}
        """.format(input_file=self.input_file, output_file=self.output_file)

        for line in commands.split('\n'):
            self.add_command(line)

        return self.exec()


class FileParser():
    structure: dict

    def __init__(self, arquivo: str = None, structure: dict = None):
        if arquivo:
            self.structure = self.parse_from_file(arquivo)
        elif structure:
            self.structure = copy.deepcopy(structure)
        else:
            raise Exception("error: no file or structure provided")

    def set_value(self, key: str, value: str):
        last_l = None
        last_d = None
        d = self.structure
        for label in key.split("."):
            last_l = label
            last_d = d
            d = d[label]
        last_d[last_l] = value

    def get_value(self, key: str):
        d = self.structure
        for label in key.split("."):
            if type(d) == list:
                d = d[int(label)]
            else:
                d = d[label]
        return d

    def set_values(self, keys: list[str], values: list[str]):
        for key, value in keys, values:
            self.set_values(key, value)

    def get_values(self, keys: list[str]):
        return [self.get_value(key) for key in keys]

    def parse_into_file(self) -> str:
        raise NotImplementedError

    def parse_from_file(self, arquivo: str):
        raise NotImplementedError


class AVLFileParser(FileParser):
    def parse_into_file(self) -> str:
        raise NotImplementedError

    def parse_from_file(self, arquivo: str):
        return parse_avl_file(arquivo)


class AVLResultParser(FileParser):
    def parse_into_file(self) -> str:
        raise NotImplementedError

    def parse_from_file(self, arquivo: str) -> str:
        raise NotImplementedError


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


def test():
    # avl = AVL(AVL_ENV_PATH, AVL_OUTPUT_FILE,  AVL_INPUT_FILE)
    # avl.analyse_v1()
    with open("geometria.txt") as f:
        x = AVLFileParser(arquivo=f.read())
        print(json.dumps(x.structure, indent=4))
        base_lbl = "children.surfaces"
        for i in list(x.get_value(base_lbl)):
            lbl = base_lbl + "." + str(i)
            print(f"'{lbl}' = '{x.get_value(lbl)}'\n")


def main():
    avl = AVL(AVL_ENV_PATH, AVL_OUTPUT_FILE,  AVL_INPUT_FILE)

    evaluator = Evaluator(
        [
            "corda",
            "envergadura"
        ],
        "cl"
    )

    # otimizar ->
    # - cn
    # - 3 momentos

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

        new_fp = AVLFileParser(str)
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


if __name__ == "__main__":
    # main()
    test()
