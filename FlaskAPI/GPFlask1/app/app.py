from flask import Flask, render_template, request
#from Configuration.Configuration import Config, FileInfo
from Configuration.Configuration import Config, FileInfo
from utils.datasetinfo.readjson import read_json_file
from utils.utils import copy_file, clear_folder
from model.ModelPredict import predict


app = Flask(__name__)
pages = {
    "index": "index.html"
}
app.config['videos_info'] = []
app.config['videos_static_path'] = "app/static/sample"
app.config['current_video'] = ""
app.config['current_video_dataset'] = ""
app.config['current_video_is_annotated'] = False
app.config['current_video_type'] = ""
app.config['current_video_class'] = ""


@app.route('/Predict', methods=["GET", "POST"])
def Predict():
    current_video = app.config['current_video']
    if app.config['videos_info'][current_video.split('.')[0]]['pretrained_path'] == "":
        anomaly_detection_res, predicted_video_type, predicted_video_class = predict(app.config["classifier_model"],
                                                                                 app.config["C3D_model"],
                                                                                 app.config['videos_static_path'],
                                                                                 app.config['current_video'],
                                                                                 app.config['videos_static_path'])
    else:
        copy_file(app.config['videos_info'][current_video.split('.')[0]]['pretrained_path'],
                  app.config['videos_static_path'] + "/{}_result.mp4".format(current_video))
        anomaly_detection_res = "/{}_result.mp4".format(current_video)
        predicted_video_type = app.config['videos_info'][current_video.split('.')[0]]['file_type']
        predicted_video_class = app.config['videos_info'][current_video.split('.')[0]]['file_class']
    return render_template(pages["index"],
                           videos_info=app.config['videos_info'],
                           video_name=app.config['current_video'],
                           result_visibility="visible",
                           disable_result=False,
                           video_dataset=app.config['current_video_dataset'],
                           video_is_annotated=app.config['current_video_is_annotated'],
                           video_type=app.config['current_video_type'],
                           video_class=app.config['current_video_class'],
                           anomaly_detection_res=anomaly_detection_res,
                           predicted_video_type=predicted_video_type,
                           predicted_video_class=predicted_video_class)


@app.route('/Getvideo', methods=["GET", "POST"])
def Getvideo():
    video_name = request.args.get('video_name')
    video_path = app.config["videos_info"][video_name][FileInfo.file_path]
    clear_folder(app.config['videos_static_path'])
    copy_file(video_path, app.config['videos_static_path'])
    app.config['current_video'] = "{}.{}".format(video_name, Config.extension)
    app.config['current_video_dataset'] = app.config['videos_info'][video_name][FileInfo.dataset_name]
    app.config['current_video_is_annotated'] = app.config['videos_info'][video_name][FileInfo.dataset_annotated]
    app.config['current_video_type'] = app.config['videos_info'][video_name][FileInfo.file_type]
    app.config['current_video_class'] = app.config['videos_info'][video_name][FileInfo.file_class]
    return render_template(pages["index"],
                           videos_info=app.config['videos_info'],
                           video_name=app.config['current_video'],
                           result_visibility="hidden",
                           disable_result=True,
                           video_dataset=app.config['current_video_dataset'],
                           video_is_annotated=app.config['current_video_is_annotated'],
                           video_type=app.config['current_video_type'],
                           video_class=app.config['current_video_class'])


@app.route("/GetData", methods=["GET", "POST"])
def GetData():
    data = read_json_file()
    app.config['videos_info'] = data["dataset"]

    clear_folder(app.config['videos_static_path'])
    keys = list(app.config['videos_info'].keys())
    copy_file(app.config['videos_info'][keys[0]][FileInfo.file_path], app.config['videos_static_path'])
    app.config['current_video_dataset'] = app.config['videos_info'][keys[0]][FileInfo.dataset_name]
    app.config['current_video_is_annotated'] = app.config['videos_info'][keys[0]][FileInfo.dataset_annotated]
    app.config['current_video_type'] = app.config['videos_info'][keys[0]][FileInfo.file_type]
    app.config['current_video_class'] = app.config['videos_info'][keys[0]][FileInfo.file_class]
    app.config['current_video'] = "{}.{}".format(keys[0], Config.extension)
    return render_template(pages["index"],
                           videos_info=app.config['videos_info'],
                           video_name=app.config['current_video'],
                           result_visibility="hidden",
                           disable_result=True,
                           video_dataset=app.config['current_video_dataset'],
                           video_is_annotated=app.config['current_video_is_annotated'],
                           video_type=app.config['current_video_type'],
                           video_class=app.config['current_video_class'])


@app.route('/', methods=["GET", "POST"])
def run():
    request_type_str = request.method
    if request_type_str == "GET":
        return GetData()

