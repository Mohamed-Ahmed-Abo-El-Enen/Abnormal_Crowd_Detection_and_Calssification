
import os

prediction_list =[]
Prediction_path = "app/static/predictions"
for root, sub_dir, files in os.walk(Prediction_path):
    prediction_list.append(files)
    for file in files :
        prediction_list.append(file)
