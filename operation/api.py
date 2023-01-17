import json

def get_command(key):
    with open('./operation/controller.json') as json_file:
        data = json.load(json_file)
        return data[key]