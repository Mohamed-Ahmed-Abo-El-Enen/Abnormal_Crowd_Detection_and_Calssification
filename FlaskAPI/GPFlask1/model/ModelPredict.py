import os
import numpy as np
from Configuration.Configuration import WeaklySupervisedConfig
from model.preprocessing import get_video_clips, c3d_feature_extractor, preprocess_input, interpolate, extrapolate
from model.visualization import visualize_predictions
import time


def predict(classifier_model, C3D_model, directory, file_name, destination):
    start_time = time.time()
    sample_video_path = os.path.join(directory, file_name).replace("\\", "/")
    video_clips, num_frames = get_video_clips(sample_video_path)
    print("Number of clips in the video : ", len(video_clips))
    # build models
    feature_extractor = c3d_feature_extractor(C3D_model)
    print("Models initialized")

    # extract features
    rgb_features = []
    for i, clip in enumerate(video_clips):
        clip = np.array(clip)
        if len(clip) < WeaklySupervisedConfig.frame_count:
            continue

        clip = preprocess_input(clip)
        rgb_feature = feature_extractor.predict(clip)[0]
        rgb_features.append(rgb_feature)

        print("Processed clip : ", i)

    rgb_features = np.array(rgb_features)
    rgb_feature_bag = interpolate(rgb_features, WeaklySupervisedConfig.features_per_bag)

    # classify using the trained classifier model
    predictions = classifier_model.predict(rgb_feature_bag)

    predictions = np.array(predictions).squeeze()

    predictions = extrapolate(predictions, num_frames)

    prediction_time = (time.time() - start_time)
    print("%s seconds: " % prediction_time)
    print("frame per second: %s" % (num_frames/prediction_time))

    anomaly_detection_res = "/{}_result.mp4".format(file_name)

    result_path = os.path.join(destination, anomaly_detection_res).replace("\\", "/")
    # visualize predictions
    print('Executed Successfully - ' + anomaly_detection_res + ' saved')
    visualize_predictions(sample_video_path, predictions, result_path)
    predicted_video_class = ""
    predicted_video_type = ""
    return anomaly_detection_res, predicted_video_type, predicted_video_class

