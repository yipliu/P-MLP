import json

def loadjson(p):
    with open(p, 'r') as f:
        data = json.load(f)
    return data


def dumpjson(data, p):
    with open(p, 'w') as f:
        json.dump(data, f, indent=2)