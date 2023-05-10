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
            queue = queue[:1]
            curr = queue[0]
            continue

        if expecting_title:
            if is_comment:
                continue
            else:
                old_title = curr["title"]
                if curr["type"] != "root":
                    queue[-2:][0]["children"][relationship_name][value] = curr
                    del queue[-2:][0]["children"][relationship_name][old_title]
                curr["title"] = value
                expecting_title = False
                continue

        if len(tokens) == 1 and rule and rule.get("is_root"):
            parent = curr
            curr = {
                "type": tokens[0],
                "title": tokens[0],
                "children": {}
            }
            queue.append(curr)
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


if __name__ == "__main__":
    with open('geometria.txt') as f:
        structure = parse_avl_file(f.read())
        out = json.dumps(structure, indent=4)
        f = open("out.json", "w")
        f.write(out)
        f.close()
