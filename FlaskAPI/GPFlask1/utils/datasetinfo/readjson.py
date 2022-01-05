import json
import os
from Configuration.Configuration import JsonFile


def read_json_file():
    file_path = os.path.join(JsonFile.json_directory, JsonFile.json_name)
    f_json = open(file_path)
    data = json.load(f_json)
    return data