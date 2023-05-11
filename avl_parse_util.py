import json


def gettokens(s):
    t = ""
    tokens = []

    for line in s.split('\n'):
        if len(line) == 0:
            continue
        else:
            if line[1:].startswith('---') or line[1:].startswith('==='):
                line = '#---'
            t += line + "\n"

    for line in t.split('\n'):
        line: str = line
        line = line.strip()
        if len(line) == 0:
            continue
        for i in ['\t', '\n', '\r', '\v', '\f']:
            line = line.replace(i, ' ')
        line = line.replace("\t", " ")
        comment = False
        if line[0] == '#':
            comment = True
            line = line[1:]
        tks = line.split(' ')
        cks = []
        for item in tks:
            if len(item) > 0:
                cks.append(item)
        if len(cks) == 0:
            continue
        item = {
            "comment": comment,
            "value": line,
            "tokens": cks
        }
        tokens.append(item)

    return tokens


keyword_rules: dict[str, dict] = json.load(open('parse_rules.json'))


def parse_avl_file(avl_file: str):
    tokens_dict = gettokens(avl_file)

    structure = {
        "type": "root",
        "title": "",
        "children": {}
    }
    expecting_title = True
    expecting_value_labels = []
    curr = structure
    queue = [structure]
    q_ptr = 0

    for token_obj in tokens_dict:
        tokens = token_obj["tokens"]
        value = token_obj["value"]
        is_comment = token_obj["comment"]

        rule = None
        if len(tokens) == 1:
            rule = keyword_rules.get(tokens[0])

        if value.startswith('---'):
            if expecting_title:
                raise Exception(
                    "error: expected title, none provided. line: " + str(i))
            continue

        if expecting_title:
            if is_comment:
                continue
            else:
                old_title = curr["title"]
                if curr["type"] != "root":
                    children_node = queue[q_ptr -
                                          1]["children"][relationship_name]

                    children_node[value] = curr
                    del children_node[old_title]

                    queue[q_ptr -
                          1]["children"][relationship_name] = children_node
                curr["title"] = value
                expecting_title = False
                continue

        if len(tokens) == 1 and rule and rule.get("is_root"):
            while queue[q_ptr]["type"] != rule["root_parent"]:
                q_ptr -= 1
                queue.pop()
            parent = queue[-1:][0]
            curr = {
                "type": tokens[0],
                "title": tokens[0],
                "children": {}
            }
            queue.append(curr)
            q_ptr += 1
            relationship_name = rule["parent_relationship_name"]

            if parent["children"].get(relationship_name) is None:
                parent["children"][relationship_name] = {}

            i = 0
            while i == 0 or parent["children"][relationship_name].get(curr["title"]) is not None:
                i += 1
                curr["title"] = tokens[0] + "_" + str(i)

            parent["children"][relationship_name][curr["title"]] = curr

            if rule.get("expect_title"):
                expecting_title = True
            else:
                del curr["title"]

            continue

        if not rule and len(expecting_value_labels) > 0 and not is_comment:
            for token in tokens:
                try:
                    label = expecting_value_labels.pop()
                except:
                    raise Exception(
                        "error: provided unexpected value. line: " + str(i))
                b = curr["children"].get(label)
                if b is None:
                    b = token
                elif type(b) is list:
                    b.append(token)
                else:
                    b = [b, token]
                curr["children"][label] = b
            continue

        exp = []
        for j in range(len(tokens)-1, -1, -1):
            el = tokens[j]
            if keyword_rules.get(el):
                expect_values = keyword_rules[el]["expects"]
                for i in range(expect_values):
                    exp.append(el)
            else:
                exp = []
                break
        if len(exp) > 0:
            expecting_value_labels += exp

    return structure


def build_avl_file(st: dict, depth: int = 0) -> str:
    # procedure:
    #   se tem titulo:
    #     printa titulo
    #   senao se tiver tipo e não é root:
    #     printa tipo
    #   para cada filho
    #       se filho é dict:
    #           procedure(filho)
    #       senao:
    #           printa nomes atributos
    #           printa valores atributos

    def tab():
        return "\t" * depth

    def end():
        return '\n' + tab()

    out = ""

    if depth > 0:
        out += "#---------------" + end()

    if st.get("type") and st["type"] != "root":
        out += st["type"] + end()

    if st.get("title"):
        out += st["title"] + end()

    if st.get("children"):
        for name, child in st["children"].items():
            if type(child) == dict:
                for sc in child.values():
                    out += build_avl_file(sc, depth+1) + end()
            else:
                if keyword_rules.get(name):
                    out += '#' + name + end()
                else:
                    out += '#' + name + end()
                l = child
                if type(child) != list:
                    l = [child]
                for i in l:
                    out += f"{i} "
                out += end()

    return out


def parse_avl_out_file(out_file: str):
    tokens = ' \n '.join(out_file.split("\n"))
    tokens = ' = '.join(tokens.split("="))
    tokens = ' '.join(tokens.split("\t"))
    tokens = ' '.join(tokens.split("#"))
    tokens = tokens.split(" ")

    c_tokens = []
    for e in tokens:
        if e == '\n':
            c_tokens.append('\n')
            continue
        e = e.strip()
        if len(e) == 0 or e.startswith("--"):
            continue
        c_tokens.append(e)

    i = 0
    final = {}
    for t in c_tokens:
        if t == '=':
            j = i
            l = []
            while c_tokens[j-1] != '\n':
                l += [c_tokens[j-1]]
                j -= 1
            l = reversed(l)
            label = ' '.join(l)
            if label == "Clb Cnr / Clr Cnb":
                final[label] = c_tokens[i+1]
            elif label == "Neutral point Xnp":
                final[label] = c_tokens[i+1]
            else:
                final[c_tokens[i-1]] = c_tokens[i+1]
        i += 1

    return final


def build_out_avl_file(st: dict) -> str:
    template = ""
    with open("template_out.txt") as f:
        template = f.read()
    for k, v in st.items():
        print(k, v)
        template = template.replace('{' + k + '}', v)
    return template


def to_structure():
    with open('g2.avl') as f:
        structure = parse_avl_file(f.read())
        out = json.dumps(structure, indent=4)
        f = open("t2.json", "w")
        f.write(out)
        f.close()


def to_out_structure():
    with open('o3.txt') as f:
        st_out = parse_avl_out_file(f.read())
        out = json.dumps(st_out, indent=4)
        f = open("to3.json", "w")
        f.write(out)
        f.close()


def to_out_file():
    with open('to3.json') as f:
        st_out = json.loads(f.read())
        out = build_out_avl_file(st_out)
        f = open("io3.json", "w")
        f.write(out)
        f.close()


def to_avl():
    with open('t2.json') as f:
        st = json.loads(f.read())
        avl_f = build_avl_file(st)
        f2 = open("g2.avl", "w")
        f2.write(avl_f)
        f2.close()


if __name__ == "__main__":
    to_out_structure()
    to_out_file()
