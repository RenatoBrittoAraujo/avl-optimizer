import subprocess
from os.path import exists
import copy
import json
import math
import threading
import hashlib
import json


from avl_parse_util import (
    parse_avl_file,
    build_avl_file,
    parse_avl_out_file,
    build_out_avl_file,
)

CONFIG_FILE = "config.json"


def get_dict_hash(d):
    # Serialize the dictionary to a JSON string
    json_str = json.dumps(d, sort_keys=True)

    # Calculate the hash of the JSON string
    hash_object = hashlib.md5(json_str.encode())
    return hash_object.hexdigest()


def to_float(v: str) -> tuple[bool, float]:
    try:
        return True, float(v)
    except Exception:
        return False, 0.0


def write_file(target, content, clear=False):
    with open(target, "w") as f:
        if clear:
            f.truncate(0)
        f.write(content)


def copy_file(source, target):
    subprocess.run("cp " + source + " " + target, shell=True)


def read_file(target):
    with open(target) as f:
        return f.read()


def read_json(file):
    with open(file, "r") as f:
        return json.load(f)


def delete_file(target):
    try:
        subprocess.call(["rm", target])
    except Exception as e:
        print("ERROR! tried to delete file", target, "but failed because", e)
        pass


class FileParser:
    structure: dict

    def __init__(self, arquivo: str = None, structure: dict = None):
        if arquivo:
            self.structure = self.parse_from_file(arquivo)
        elif structure:
            if type(structure) != dict:
                if isinstance(structure, FileParser):
                    structure = structure.structure
                else:
                    raise Exception("error: structure must be a dict")
            self.structure = copy.deepcopy(structure)
        else:
            raise Exception("error: no file or structure provided")

    def set_template(self, fp: "FileParser"):
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

    def flatten(self) -> dict:
        return self.__flatten(self.structure, "")

    def __flatten(self, o: dict, s: str) -> dict:
        if o is None:
            o = self.structure
        if type(o) != dict and type(o) != list:
            return {s: o}
        res = {}
        for k, v in o.items():
            vs = s
            if vs != "":
                vs += "."
            vs += k
            vs = vs.replace(" ", "\_")

            if type(v) == dict:
                f = self.__flatten(v, vs)
                res = {**res, **f}
            elif type(v) == list:
                for i in range(len(v)):
                    ts = vs + f".{i}"
                    f = self.__flatten(v[i], ts)
                    res = {**res, **f}
            else:
                res[vs] = v
        return res

    def __get(self, obj: any, key: str) -> any:
        if type(obj) == list:
            return obj[int(key)]
        else:
            if key not in obj:
                if "\\_" in key:
                    other_possible = key.replace("\\_", " ")
                    if other_possible in obj:
                        return obj[other_possible]
                raise KeyError
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


class AVL:
    command_list: str
    avl_folder_path: str
    input_file: str
    output_file: str
    overwrite_any: bool
    io_file_indx: int

    def __init__(
        self,
        avl_folder_path: str,
        input_file: str,
        output_file: str,
        overwrite_any: bool,
    ):
        if not avl_folder_path.endswith("/"):
            avl_folder_path += "/"
        if not input_file.endswith(".avl"):
            raise Exception("error: file '" + input_file + "' is not .avl")

        self.avl_folder_path = avl_folder_path
        self.input_file = input_file
        self.output_file = output_file
        self.overwrite_any = overwrite_any
        self.command_list = ""
        self.io_file_indx = 0

    def add_command(self, cmd: str):
        self.command_list += cmd + "\n"

    def get_new_id(self) -> int:
        self.io_file_indx += 1
        return self.io_file_indx - 1

    def analyse_for_thread(
        self,
        thread_id: int,
        thread_label: str,
        thread_results: dict,
        thread_end_event: threading.Event,
        in_fp: AVLFileParser,
    ) -> AVLResultParser:
        nid = self.get_new_id()
        print(f"[id: {thread_id}, label: {thread_label}, nid: {nid}] thread started")
        res_fp = self.analyse_from_fp(in_fp, nid=nid)
        thread_results[thread_id] = res_fp
        thread_end_event.set()
        print(f"[id: {thread_id}, label: {thread_label}, nid: {nid}] thread ended")
        return res_fp

    def analyse_from_fp(self, in_fp: AVLFileParser, nid: int = None) -> AVLResultParser:
        if nid is None:
            nid = self.get_new_id()
        self.command_list = ""

        inf, ouf = self.create_in_out_file(nid)
        write_file(inf, in_fp.parse_into_file())
        delete_file(ouf)

        process = self.analyse_v1(str(nid), inf, ouf)

        res_str = read_file(ouf)
        res_fp = AVLResultParser(arquivo=res_str)
        # delete_file(ouf)

        return res_fp

    def analyse_v1(self, label: int, inf: str, ouf: str) -> str:
        commands = """
        LOAD {input_file}
        OPER
        X
        ST
        {output_file}
        """.format(
            input_file=inf.replace("env/", ""), output_file=ouf.replace("env/", "")
        )

        for line in commands.split("\n"):
            line = line.strip()
            if line:
                self.add_command(line.strip())

        return self.exec(label)

    def exec(self, label: str):
        if not self.command_list:
            raise Exception("error: no avl commands have been provided")

        print(f"[{label}] running a thread....")

        process = subprocess.run(
            ["./avl"],
            cwd="env",
            input=bytes(self.command_list, encoding="utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        write_file("dev_files/commands_sent.txt", self.command_list)
        write_file("dev_files/stdout.txt", process.stdout.decode("utf-8"))
        write_file("dev_files/stderr.txt", process.stderr.decode("utf-8"))

        print(f"[{label}] thread ended!")
        err = process.stderr.decode("utf-8")
        err.replace(
            "At line 145 of file ../src/userio.f (unit = 5, file = 'stdin')\nFortran runtime error: End of file\n",
            "",
        )

        # [TODO] parse de erros aqui
        # if err != "":
        #     print("AVL ERROR!")
        #     print(f"[{label}] stderr: {process.stderr}")

        return process

    def create_in_out_file(self, nid: int):
        inp = self.avl_folder_path + self.input_file.replace(".avl", f"_{nid}.avl")
        out = self.avl_folder_path + self.output_file.replace(".txt", f"_{nid}.txt")
        return inp, out


class ThreadQueue:
    max_threads: int
    last_tid: int
    running_threads: list[dict[str, any]]
    thread_end_event: threading.Event
    results: dict[int, any]
    label_map: dict[str, int]

    def __init__(
        self,
        max_threads: int = 1,
    ):
        self.max_threads = max_threads
        self.running_threads = []
        self.last_tid = 0
        self.thread_end_event = threading.Event()
        self.results = {}
        self.label_map = {}

    def add_new_thread_blocking(
        self, procedure: any, args: tuple, label: str = None
    ) -> int:
        self.wait_queue_space_if_any()
        thread_id = self.create_new_thread_id()
        if label is None:
            label = str(thread_id)
        args = (thread_id, label, self.results, self.thread_end_event, *args)
        t = threading.Thread(
            target=procedure,
            args=args,
            daemon=True,
        )
        self.running_threads.append(t)
        t.start()
        return thread_id, label

    def get_thread_result_blocking(self, thread_label: str):
        tid = None
        if thread_label in self.label_map:
            tid = self.label_map[thread_label]
        if thread_label.isdigit():
            tid = int(thread_label)
        if tid is None or tid >= self.last_tid:
            print(f"thread_id for label '{thread_label}' not found")
            raise KeyError

        if tid in self.results:
            result = self.results[tid]
            del self.results[tid]
            return result

        print(
            f"thread_id {tid} for label '{thread_label}' has not given set any result"
        )
        return None

    def create_new_thread_id(self) -> int:
        c_tid = self.tid
        self.tid += 1
        return c_tid

    def wait_queue_space_if_any(self):
        if len(self.running_threads) < self.max_threads:
            return
        self.thread_end_event.wait()

    def wait_all_threads(self):
        for t in self.running_threads:
            t.join()


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
        if "INPUT_INDEX" not in globals():
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
    ev_adatper: "EvaluatorAdapter"

    def set_ev_adapter(self, ev_adapter: "EvaluatorAdapter"):
        self.ev_adatper = ev_adapter

    def get_score_from_outfile(
        self, x: list[float]
    ) -> tuple[float, list[str], AVLResultParser]:
        raise NotImplementedError

    def get_score(
        self,
        in_fp: AVLFileParser,
        out_fp: AVLResultParser,
        inputs: list[Input],
        outputs: list[Output],
    ) -> tuple[float, list[str], AVLResultParser]:
        raise NotImplementedError


class SumScorer(Scorer):
    # [TODO] Implementar uma função de score real
    def get_score_from_outfile(
        self, x: list[float]
    ) -> tuple[float, list[str], AVLResultParser]:
        out_fp, in_fp, inputs, outputs = self.ev_adatper.get_results_from_avl(x)

        vals_sum = 0
        for out in outputs:
            val = float(out_fp.get_value(out.key))
            vals_sum += val
        print("============== GOT NEW SCORE!!!!", vals_sum)
        return vals_sum, None, out_fp


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

    def validate(self, input_fp: AVLFileParser):
        corda = 180  # mm
        corda = 180  # mm

        return True

    # retorna score e os erros que ocorreram
    def get_score_from_outfile(
        self, x: list[float]
    ) -> tuple[float, list[str], AVLResultParser]:
        # print("feeding avl this: ...")
        # print(self.ev_adatper.get_avl_file_from_inputs(x).structure)

        out_fp, in_fp, inputs, outputs = self.ev_adatper.get_results_from_avl(x)
        return self.get_score(in_fp, out_fp, inputs, outputs)

    def get_score(
        self,
        in_fp: AVLFileParser,
        out_fp: AVLResultParser,
        inputs: list[Input],
        outputs: list[Output],
    ) -> tuple[float, list[str], AVLResultParser]:
        # [TODO] NOTA: Não tem garantias e limites de tamanho!
        # - to fazendo
        # - limitado direito no input
        # [TODO] NOTA: Não inclue garantias de controle ainda!
        # - depois
        # [TODO] NOTA: LIMITAÇõES POSICIONAMENTO - onde a cauda pode ficar, por exemplo
        # - aleef vai manda
        # - não é prioridade

        erros = []

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
                "erro": "falha de estabilidade - Clb_Cnr_over_Clr_Cnb <= 1",
            },
            {"cond": Cma < 0, "erro": "falha de estabilidade - Cma >= 0"},
            {"cond": Clb < 0, "erro": "falha de estabilidade - Clb >= 0"},
            {"cond": Cnb > 0, "erro": "falha de estabilidade - Cnb <= 0"},
        ]

        for cond in condicoes:
            if cond["cond"] != True:
                # [TODO] NOTA: Não acontece nada quando uma condição é quebrada
                # - encontre sua própria solução, o resultado final precisa atender elas
                # - já ta parametrizado na formula alguns desses
                # - talvez adicionar um relu
                # - deixa rodar e fodase
                erros.append(cond["erro"])

        P = (
            -0.1 * math.fabs(Cma - 0.675)
            + -0.1 * math.fabs(Cnb - 0.07)
            + -0.1 * math.fabs(Clb - 0.07)
            + +0.1 * CLtot
            + -0.1 * CDtot
            + -0.1 * Cmtot
        )

        print("============== GOT NEW SCORE!!!!", P)
        return P, erros, out_fp


class Evaluator:
    max_iter_count: int
    limit_iter_count: int
    limit_variation_factor: float
    interval_steps: int
    score_precision: float
    scorer: Scorer
    empty_changeset: dict
    thread_queue: ThreadQueue
    avl: AVL

    def __init__(
        self,
        max_iter_count: int,
        limit_iter_count: int,
        limit_variation_factor: float,
        interval_steps: int,
        score_precision: float,
        scorer: Scorer,
        thread_queue: ThreadQueue,
        avl: AVL,
    ):
        self.max_iter_count = max_iter_count
        self.limit_iter_count = limit_iter_count
        self.limit_variation_factor = limit_variation_factor
        self.interval_steps = interval_steps
        self.score_precision = score_precision
        self.scorer = scorer
        self.empty_changeset = {"changed": False, "out_fp": None}
        self.thread_queue = thread_queue
        self.avl = avl

    # Implementation of Newton-Raphson method
    # https://en.wikipedia.org/wiki/Newton%27s_method
    def evaluate_derivative(self, inputs: list[Input]) -> list[float]:
        iter_count = 0

        step_sizes = []
        for inp in inputs:
            if inp.min_variation:
                step_sizes.append(inp.min_variation)
            else:
                step_sizes.append(inp.get_interval_amplitude() / self.interval_steps)

        x_next = [float(inp.curr) for inp in inputs]

        print("[main] Getting initial out file...")
        self.
        self.avl.analyse_from_fp()
        _, _, init_out_fp = self.scorer.get_score_from_outfile(x_next)

        while True:
            if iter_count > self.max_iter_count:
                return x_next
            iter_count += 1
            x_new = x_next.copy()
            x_out_changes = False
            x_inp_changes = False

            print(f"[main]  CREATING {len(inputs)} NEW THREADS")

            for inp in inputs:
                indx = inputs.index(inp)

                self.thread_queue.add_new_thread_blocking(
                    self.get_new_variations,
                    (x_new, step_sizes[indx]),
                    label=int(indx),
                )

            self.thread_queue.wait_all_threads()

            for i in range(len(x_new)):
                if x_next[i] != x_new[i]:
                    print(
                        f"Changed the input {inputs[i].key} in x_next versus x_new from {x_next[i]} to {x_new[i]}"
                    )
                    x_inp_changes = True

                out_fp = self.thread_queue.get_thread_result_blocking(int(i))

                if out_fp is None:
                    print("out_fp is None!")
                    continue

                max_err = 1e-6
                for key, orig_val in init_out_fp.flatten().items():
                    res_val = float(out_fp.get_value(key))
                    orig_val = float(orig_val)
                    if math.fabs(res_val - orig_val) > max_err:
                        x_out_changes = True
                        print(f"[main] the attribute '{key}' is changed")
                        print(f"[main] intial value is: {orig_val}")
                        print(f"[main] new value is:    {res_val}")

            if not x_out_changes and not x_inp_changes:
                print("no changes detecting, ending the evalutor")
                break

        return x_next

    def get_new_variations(
        self,
        indx:int,
        inp: list[float],
        variation: float,
    ) -> tuple[float, bool]:
        # # increase so that a convergance is eventually forced
        # if iter_count > self.limit_iter_count:
        #     step_sizes[indx] = step_sizes[indx] * self.limit_variation_factor

        fx, errors, out_fp = self.scorer.get_score_from_outfile(inp)
        if errors is not None:
            print("ERROS!\n", errors)

        # d = f(x) / (x1 - x2)
        d = fx / -variation

        # if math.fabs(d*step_sizes[indx]) < min_variation[indx]:
        #     continue
        # el
        print(f"d = {d}")
        if d > 0:
            nv = inp[indx] + variation * d
            inp[indx] = nv
        else:
            nv = inp[indx] - variation * d
            inp[indx] = nv

        return out_fp


class EvaluatorAdapter:
    res_memo: dict[tuple[float], AVLResultParser]

    def __init__(
        self,
        avl: AVL,
        input_file: AVLFileParser,
        inputs: list[Input],
        outputs: list[Output],
        scorer: Scorer,
    ):
        self.avl = avl
        self.input_file = input_file
        self.inputs = inputs
        self.outputs = outputs
        self.in_fp_base = input_file
        self.scorer = scorer
        self.res_memo = {}

        scorer.set_ev_adapter(self)

    def optimize(self) -> tuple[AVLFileParser, AVLResultParser]:
        vals = self.evaluator.evaluate_derivative(self.inputs)
        return self.get_avl_file_from_inputs(vals), self.get_results_from_avl(vals)

    def get_results_from_avl(
        self, x: list[float]
    ) -> tuple[AVLFileParser, AVLResultParser, list[Input], list[Output]]:
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

    def get_score_from_file(self, file: str):
        return self.scorer.get_score_from_outfile(file)


class AppState:
    cfg: dict
    input_fp: AVLFileParser
    avl: AVL
    inputs: list[Input]
    outputs: list[Output]
    scorer: Scorer
    evaluator: Evaluator
    evaluator_adapter: EvaluatorAdapter
    avl_thread_queue: ThreadQueue

    def __init__(self):
        pass

    def init_prod(self):
        self.cfg = read_json(CONFIG_FILE)

        self.input_fp = AVLFileParser(arquivo=read_file(self.cfg["base_input_file"]))

        self.avl_thread_queue = ThreadQueue(max_threads=self.cfg["max_threads"])

        self.avl = AVL(
            self.cfg["avl_env_path"],
            input_file=self.cfg["avl_input_file"],
            output_file=self.cfg["avl_output_file"],
            overwrite_any=True,
        )

        self.inputs = [
            Input(key=k, value=v, curr=self.input_fp.get_value(k))
            for k, v in self.cfg["inputs"].items()
        ]

        self.outputs = [Output(key=k) for k, v in self.cfg["outputs"].items()]

        self.scorer = V1Scorer()

        self.evaluator_adapter = EvaluatorAdapter(
            self.avl,
            self.input_fp,
            self.inputs,
            self.outputs,
            self.scorer,
        )

        self.evaluator = Evaluator(
            max_iter_count=self.cfg["max_iter_count"],
            limit_iter_count=self.cfg["limit_iter_count"],
            limit_variation_factor=self.cfg["limit_variation_factor"],
            interval_steps=self.cfg["interval_steps"],
            score_precision=self.cfg["score_precision"],
            scorer=self.scorer,
            thread_queue=self.avl_thread_queue,
            avl=self.avl,
        )




def test():
    app = AppState()
    app.init_prod()

    if1 = AVLFileParser(structure=app.input_fp)
    if2 = AVLFileParser(structure=app.input_fp)

    # vamos dobrar a corda de todas as surfaces mais longe do centro (Y=0) da asa
    # no caso seria o flap e asa (basta ver o arquivo dev_files/estrutura_flat_input.json)
    # pra ver como isso afeta o resultado, como teste

    v1 = if2.get_value(
        "children.surfaces.Right_Wing.children.sections.SECTION_3.children.Chord"
    )
    v2 = if2.get_value(
        "children.surfaces.Right_flap.children.sections.SECTION_3.children.Chord"
    )

    v1 = str(float(v1) * 2.0)
    v2 = str(float(v2) * 2.0)

    if2.set_value(
        "children.surfaces.Right_Wing.children.sections.SECTION_3.children.Chord", v1
    )
    if2.set_value(
        "children.surfaces.Right_flap.children.sections.SECTION_3.children.Chord", v2
    )

    f2 = app.avl.analyse(if2)
    f1 = app.avl.analyse(if1)

    s1, _ = app.scorer.get_score(None, f1, None, None)
    s2, _ = app.scorer.get_score(None, f2, None, None)

    print("score f1 = ", s1)
    print("score f2 = ", s2)


def prod():
    app = AppState()
    app.init_prod()

    out_avl_fp, out_analysis = app.evaluator_adapter.optimize()
    out_fp, in_fp, inputs, outputs = out_analysis

    write_file(app.cfg["final_input_file"], out_avl_fp.parse_into_file())

    write_file(
        app.cfg["avl_env_path"] + app.cfg["avl_output_file"],
        out_fp.parse_into_file(),
    )


if __name__ == "__main__":
    prod()
    # test()
