import os
import json
from Configuration.Configuration import Config, FileInfo, JsonFile, Dataset
from app.app import Predict


def get_dataset_info(dataset_directory):
    dataset_info = {}
    prediction_list =[]
    Prediction_path = "app/static/predictions"
    for root, sub_dir, files in os.walk(Prediction_path):
        for file in files:
            prediction_list.append(file)

    for root, sub_dir, files in os.walk(dataset_directory):
        for filename in files:
            if filename.split('.')[-1] != Config.extension:
                continue
            file_info = {}

            f_path = os.path.join(root, filename)
            dataset_name = f_path.split('\\')[-3]
            if dataset_name not in Dataset.dataset_info.keys():
                continue

            f_name = filename.split('.')[0]
            file_info[FileInfo.dataset_name] = dataset_name
            file_info[FileInfo.dataset_annotated] = Dataset.dataset_info[dataset_name]["Annotated"]
            file_info[FileInfo.file_path] = f_path
            if f_name+".mp4" in prediction_list:
                file_info[FileInfo.pretrained_path] = Prediction_path + '/' + f_name + ".mp4"
            else:
                file_info[FileInfo.pretrained_path] = ""
            f_class = f_path.split('\\')[-2]
            if f_class in Config.classes.keys():
                file_info[FileInfo.file_type] = "Abnormal"
                file_info[FileInfo.file_class] = "UnSpecified yet"
            else:
                file_info[FileInfo.file_type] = "Normal"
                file_info[FileInfo.file_class] = "Normal"
            dataset_info[f_name] = file_info

    json_dict = {}
    json_dict["dataset"] = dataset_info
    return json_dict


def generate_json_file(json_dict):
    file_dir = os.path.join(JsonFile.json_directory, JsonFile.json_name)
    with open(file_dir, "w") as outfile:
        json.dump(json_dict, outfile, indent=4)