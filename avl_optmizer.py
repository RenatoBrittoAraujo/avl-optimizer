import subprocess
from os.path import exists
import copy
import json
import math

from avl_parse_util import parse_avl_file, build_avl_file, parse_avl_out_file, build_out_avl_file

CONFIG_FILE = "config.json"


def to_float(v: str) -> tuple[bool, float]:
    try:
        return True, float(v)
    except Exception:
        return False, 0.0


def write_file(target, content):
    with open(target, "w") as f:
        f.write(content)


def copy_file(source, target):
    subprocess.run("cp " + source + " " + target, shell=True)


def read_file(target):
    txt = ""
    with open(target) as f:
        txt = f.read()
    return txt


def read_json(file):
    with open(file, "r") as f:
        return json.load(f)


def delete_file(target):
    try:
        subprocess.call("rm " + target)
    except Exception:
        pass


class FileParser():
    structure: dict

    def __init__(self, arquivo: str = None, structure: dict = None):
        if arquivo:
            self.structure = self.parse_from_file(arquivo)
        elif structure:
            self.structure = copy.deepcopy(structure)
        else:
            raise Exception("error: no file or structure provided")

    def set_template(self, fp: 'FileParser'):
        self.structure = copy.deepcopy(fp.structure)

    def set_value(self, key: str, value: str):
        last_l = None
        last_d = None
        d = self.structure
        for label in key.split("."):
            last_l = label
            last_d = d
            d = self.__get(d, label)
        self.__set(last_d, last_l, value)

    def get_value(self, key: str):
        d = self.structure
        for label in key.split("."):
            d = self.__get(d, label)
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

    def __get(self, obj: any, key: str) -> any:
        if type(obj) == list:
            return obj[int(key)]
        else:
            return obj[key]

    def __set(self, obj: any, key: str, val: any):
        if type(obj) == list:
            obj[int(key)] = val
        else:
            obj[key] = val


class AVLFileParser(FileParser):
    def parse_into_file(self) -> str:
        if self.structure is None:
            raise EOFError
        return build_avl_file(self.structure)

    def parse_from_file(self, arquivo: str):
        return parse_avl_file(arquivo)


class AVLResultParser(FileParser):
    def parse_into_file(self) -> str:
        return build_out_avl_file(self.structure)

    def parse_from_file(self, arquivo: str) -> str:
        return parse_avl_out_file(arquivo)


# interact with program itself

class AVL():
    command_list: str
    avl_folder_path: str
    input_file: str
    output_file: str
    overwrite_any: bool

    def __init__(self, avl_folder_path: str, input_file: str, output_file: str, overwrite_any: bool = False):
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
                raise Exception("error: file '" + out +
                                "' is not overwritable")

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

    def analyse(self, in_fp: AVLFileParser) -> AVLResultParser:
        inp = self.avl_folder_path + self.input_file
        out = self.avl_folder_path + self.output_file
        write_file(inp, in_fp.parse_into_file())
        try:
            self.analyse_v1()
        except:
            # o programa sempre fecha com erro de EOF, ainda não consertardo
            pass
        res_str = read_file(out)
        res_fp = AVLResultParser(res_str)
        return res_fp


class Input:
    key: str
    index: int
    interval: list[float]
    min_variation: float
    max_variation: float
    curr: float

    def __init__(self, key: str, curr: float, value: dict):
        self.key = key
        self.interval = value.get("interval")
        self.min_variation = value.get("min_variation")
        self.max_variation = value.get("max_variation")
        self.curr = curr

        global INPUT_INDEX
        if 'INPUT_INDEX' not in globals():
            INPUT_INDEX = 0
        self.index = INPUT_INDEX
        INPUT_INDEX += 1

    def get_interval(self) -> tuple[float, float]:
        return max(*self.interval), min(self.interval)

    def get_interval_amplitude(self) -> float:
        return max(*self.interval) - min(self.interval)


class Output:

    key: str

    def __init__(self, key: str):
        self.key = key


class Scorer:

    ev_adatper: 'EvaluatorAdapter'

    def set_ev_adapter(self, ev_adapter: 'EvaluatorAdapter'):
        self.ev_adatper = ev_adapter

    def get_score_from_outfile(self, fp: AVLResultParser) -> tuple[float, list[str]]:
        raise NotImplementedError


class SumScorer(Scorer):

    # [TODO] Implementar uma função de score real
    def get_score_from_outfile(self, x: list[float]) -> tuple[float, list[str]]:
        out_fp, in_fp, inputs, outputs = \
            self.ev_adatper.get_results_from_avl(x)

        vals_sum = 0
        for out in outputs:
            val = float(out_fp.get_value(out.key))
            vals_sum += val
        print("============== GOT NEW SCORE!!!!", vals_sum)
        return vals_sum, None


class V1Scorer(Scorer):

    # Lista de limitadores:
    # \Garantindo estabilidade
    # Clb Cnr / Clr Cnb  > 1
    # Cma < 0
    # Clb < 0
    # Cnb > 0
    # \Garantindo controlabiliade, pode ser feito separadamente
    # |Cm(elevador)| >= 0.03
    # |Cl(flaperon)| >= 0.005 !Depende de flaperon mais do que de cauda
    # |Cn(leme)| >= 0.0012

    # \Equação de pontuação
    # P = -0.1*(|Cma - 0.675|) + -0.1(|Cnb - 0.07|) + -0.1(|Clb - 0.07|) + +0.1CLtot + -0.1CDtot + -0.1Cmtot

    # retorna score e os erros que ocorreram
    def get_score_from_outfile(self, x: list[float]) -> tuple[float, list[str]]:
        # NOTA: Não inclue garantias de controle ainda!
        # NOTA: Não tem garantias e limites de tamanho!
        # NOTA: Não inclue o caso '!Para caso em alpha = 0'

        erros = []

        out_fp, in_fp, inputs, outputs = \
            self.ev_adatper.get_results_from_avl(x)

        # outputs:
        Clb_Cnr_over_Clr_Cnb = float(out_fp.get_value("Clb Cnr / Clr Cnb"))
        Cma = float(out_fp.get_value("Cma"))
        Cnb = float(out_fp.get_value("Cnb"))
        Clb = float(out_fp.get_value("Clb"))
        CLtot = float(out_fp.get_value("CLtot"))
        CDtot = float(out_fp.get_value("CDtot"))
        Cmtot = float(out_fp.get_value("Cmtot"))

        condicoes = [
            {
                "cond": Clb_Cnr_over_Clr_Cnb > 1,
                "erro": "falha de estabilidade - Clb_Cnr_over_Clr_Cnb <= 1"
            },
            {
                "cond": Cma < 0,
                "erro": "falha de estabilidade - Cma >= 0"
            },
            {
                "cond": Clb < 0,
                "erro": "falha de estabilidade - Clb >= 0"
            },
            {
                "cond": Cnb > 0,
                "erro": "falha de estabilidade - Cnb <= 0"
            },
        ]

        for cond in condicoes:
            if cond["cond"] != True:
                erros.append(cond["erro"])

        # if len(erros) > 0:
        #     return 0, erros

        P = \
            -0.1*math.fabs(Cma - 0.675) +\
            -0.1*math.fabs(Cnb - 0.07) +\
            -0.1*math.fabs(Clb - 0.07) +\
            +0.1*CLtot +\
            -0.1*CDtot +\
            -0.1*Cmtot

        print("============== GOT NEW SCORE!!!!", P)
        return P, None


class Evaluator:
    max_iter_count: int
    limit_iter_count: int
    limit_variation_factor: float
    interval_steps: int
    score_precision: float
    scorer: Scorer

    def __init__(
            self,
            max_iter_count: int,
            limit_iter_count: int,
            limit_variation_factor: float,
            interval_steps: int,
            score_precision: float,
            scorer: Scorer
    ):
        self.max_iter_count = max_iter_count
        self.limit_iter_count = limit_iter_count
        self.limit_variation_factor = limit_variation_factor
        self.interval_steps = interval_steps
        self.score_precision = score_precision
        self.scorer = scorer

    # Implementation of Newton-Raphson method
    # https://en.wikipedia.org/wiki/Newton%27s_method
    def evaluate_derivative(self, inputs: list[Input]) -> list[float]:
        iter_count = 0

        step_sizes = []
        for inp in inputs:
            if inp.min_variation:
                step_sizes.append(inp.min_variation)
            else:
                step_sizes.append(
                    inp.get_interval_amplitude()/self.interval_steps
                )

        min_variation = step_sizes.copy()

        x_next = [float(inp.curr) for inp in inputs]

        while True:
            if iter_count > self.max_iter_count:
                return x_next
            iter_count += 1
            x_changed = False

            for inp in inputs:
                i = inputs.index(inp)

                # increase so that a convergance is eventually forced
                if iter_count > self.limit_iter_count:
                    step_sizes[i] = step_sizes[i] * self.limit_variation_factor

                fx, errors = self.scorer.get_score_from_outfile(x_next)
                if errors is not None:
                    print("ERROS!\n", errors)
                variation = step_sizes[i]

                # d = f(x) / (x1 - x2)
                d = fx / -variation

                # if math.fabs(d*step_sizes[i]) < min_variation[i]:
                #     continue
                # el
                if d > 0:
                    x_next[i] = x_next[i] + step_sizes[i]*d
                else:
                    x_next[i] = x_next[i] - step_sizes[i]*d
                x_changed = False

            if not x_changed:
                break

        return x_next


class EvaluatorAdapter():

    res_memo: dict[tuple[float], AVLResultParser]

    def __init__(
        self,
        avl: AVL,
        input_file: AVLFileParser,
        inputs: list[Input],
        outputs: list[Output],
        evaluator: Evaluator,
        scorer: Scorer
    ):
        self.avl = avl
        self.input_file = input_file
        self.inputs = inputs
        self.outputs = outputs
        self.evaluator = evaluator
        self.in_fp_base = input_file
        self.scorer = scorer
        self.res_memo = {}

        scorer.set_ev_adapter(self)

    def optimize(self) -> tuple[AVLFileParser, AVLResultParser]:
        vals = self.evaluator.evaluate_derivative(self.inputs)
        return self.get_avl_file_from_inputs(vals), self.get_results_from_avl(vals)

    def get_results_from_avl(self, x: list[float]) -> tuple[AVLFileParser, AVLResultParser, list[Input], list[Output]]:
        in_fp = self.get_avl_file_from_inputs(x)
        if tuple(x) in self.res_memo:
            return self.res_memo[tuple(x)], in_fp, self.inputs, self.outputs

        self.avl.analyse(in_fp)
        out_fp = AVLResultParser(read_file("env/out.txt"))

        self.res_memo[tuple(x)] = out_fp
        return out_fp, in_fp, self.inputs, self.outputs

    def get_avl_file_from_inputs(self, x: list[float]) -> AVLFileParser:
        new_fp = AVLFileParser(str)
        new_fp.set_template(self.in_fp_base)

        for i in range(len(self.inputs)):
            new_fp.set_value(self.inputs[i].key, x[i])

        return new_fp


def test():
    with open("geometria.avl") as f:
        x = AVLFileParser(arquivo=f.read())
        print(json.dumps(x.structure, indent=4))
        # base_lbl = "children.surfaces"
        # for i in list(x.get_value(base_lbl)):
        #     lbl = base_lbl + "." + str(i)
        #     print(f"'{lbl}' = '{x.get_value(lbl)}'\n")


def main():
    cfg = read_json(CONFIG_FILE)

    input_fp = AVLFileParser(arquivo=read_file(cfg["base_input_file"]))

    avl = AVL(cfg["avl_env_path"], cfg["avl_input_file"],
              cfg["avl_output_file"], overwrite_any=True)

    inputs = [
        Input(key=k, value=v, curr=input_fp.get_value(k))
        for k, v in cfg["inputs"].items()
    ]

    outputs = [Output(key=k) for k, v in cfg["outputs"].items()]

    scorer = V1Scorer()

    evaluator = Evaluator(
        max_iter_count=cfg["max_iter_count"],
        limit_iter_count=cfg["limit_iter_count"],
        limit_variation_factor=cfg["limit_variation_factor"],
        interval_steps=cfg["interval_steps"],
        score_precision=cfg["score_precision"],
        scorer=scorer
    )

    evaluator_adapter = EvaluatorAdapter(
        avl, input_fp, inputs, outputs, evaluator, scorer)

    # otimizar ->
    # - cn
    # - 3 momentos
    out_avl_fp, out_analysis_fp = evaluator_adapter.optimize()

    write_file(cfg["final_input_file"], out_avl_fp.parse_into_file())
    copy_file(cfg["avl_env_path"] + cfg["avl_output_file"],
              cfg["final_output_file"])


if __name__ == "__main__":
    main()
    # test()
