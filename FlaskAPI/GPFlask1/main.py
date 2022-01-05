from Configuration.Configuration import Config
from utils.datasetinfo.generatejson import get_dataset_info, generate_json_file
from model.CS3D import C3D
from model.Classfier import classifier_model, load_weights
from app.app import *

if __name__ == "__main__":
    json_dict = get_dataset_info(Config.DATASET_DIR)

    app.config["classifier_model"] = classifier_model()
    app.config["classifier_model"] = load_weights(app.config["classifier_model"],
                                                  weights_file="model/weight/weights.mat")
    app.config["C3D_model"] = C3D(weights='model/weight/C3D_Sport1M_weights_keras_2.2.4.h5')

    generate_json_file(json_dict)
    app.run()

