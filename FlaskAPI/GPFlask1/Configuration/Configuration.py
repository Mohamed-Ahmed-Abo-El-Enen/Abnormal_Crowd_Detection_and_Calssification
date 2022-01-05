class Config:
    EPOCHS = 1
    num_frames_per_second = 10
    SEQUENCE_SIZE = 16
    H = 256
    W = 256
    C = 1
    TO_GRAY = True
    DATASET_DIR = "Dataset"
    save_npz_dir = "./npz_files"
    types = {"Normal": 0, "Abnormal": 1}
    classes = {"Explosion": 1, 'Burglary': 2, 'Fighting': 3, 'Assault': 4, 'Arrest': 5, 'Arson': 6,
               'Abuse': 7}
    extension = "mp4"
    MODEL_WEIGHTS_DIRECTORY = "./model_weights"
    COMBINE_MODEL_PATH = "combined_model_weights.hdf5"
    GENERATOR_MODEL_PATH = "generator_model_weights.hdf5"
    DISCRIMINATOR_MODEL_PATH = "discriminator_model_weights.hdf5"
    CLASSIFIER_MODEL_PATH = "classifier_model_weights.hdf5"


class FileInfo:
    file_name = "file_name"
    dataset_name = "dataset_name"
    dataset_annotated = "dataset_annotated"
    file_path = "file_path"
    file_class = "file_class"
    file_type = "file_type"
    pretrained_path = "pretrained_path"


class JsonFile:
    json_name = "dataset.json"
    json_directory = "utils\datasetinfo"


class Dataset:
    dataset_info = {
        "UCF": {"Annotated": True, "Num_types": 2, "Num_classes": 8},
        "UCSD": {"Annotated": False, "Num_types": 0, "Num_classes": 0}
    }


class WeaklySupervisedConfig:
    frame_height = 256
    frame_width = 256
    channels = 3
    frame_count = 5
    extension = "mp4"
    features_per_bag = 20