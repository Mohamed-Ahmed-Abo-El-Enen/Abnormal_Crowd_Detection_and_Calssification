from os import listdir
from os.path import join
import cv2
import numpy as np
from Configuration.Configuration import WeaklySupervisedConfig
from tensorflow.keras.models import Model
from tensorflow.keras.utils import get_file


#C3D_MEAN_PATH = 'https://github.com/adamcasson/c3d/releases/download/v0.1/c3d_mean.npy'
mean_path = "model/weight/c3d_mean.npy"


def get_video_frames(video_path):
    if WeaklySupervisedConfig.extension == "mp4":
        return get_mp4_video_frames(video_path)
    return get_tif_video_frames(video_path)


def get_mp4_video_frames(video_path):
    """Reads the video given a file path
    :param video_path: Path to the video
    :returns: Video as an array of frames
    :rtype: np.ndarray
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            break
    cap.release()
    return frames


def get_tif_video_frames(video_path):
    """Reads the video given a file path
    :param video_path: Path to the video
    :returns: Video as an array of frames
    :rtype: np.ndarray
    """
    frames=[]
    for c in sorted(listdir(video_path)):
        img_path = join(video_path, c).replace("\\", '/')
        if str(img_path)[-3:] == "tif":
            img = cv2.imread(img_path)
           # img = np.array(img, dtype=np.float32) / 256.0
            frames.append(img)

    return frames


def get_video_clips(video_path):
    """Divides the input video into non-overlapping clips
    :param video_path: Path to the video
    :returns: Array with the fragments of video
    :rtype: np.ndarray
    """
    frames = get_video_frames(video_path)
    clips = sliding_window(frames, WeaklySupervisedConfig.frame_count, WeaklySupervisedConfig.frame_count)
    return clips, len(frames)


def sliding_window(arr, size, stride):
    """Apply sliding window to an array, getting chunks of
    of specified size using the specified stride
    :param arr: Array to be divided
    :param size: Size of the chunks
    :param stride: Number of frames to skip for the next chunk
    :returns: Tensor with the resulting chunks
    :rtype: np.ndarray
    """
    num_chunks = int((len(arr) - size) / stride) + 2
    result = []
    for i in range(0,  num_chunks * stride, stride):
        if len(arr[i:i + size]) > 0:
            result.append(arr[i:i + size])
    return np.array(result)


def get_video_clips(video_path):
    """Divides the input video into non-overlapping clips
    :param video_path: Path to the video
    :returns: Array with the fragments of video
    :rtype: np.ndarray
    """
    frames = get_video_frames(video_path)
    clips = sliding_window(frames, WeaklySupervisedConfig.frame_count, WeaklySupervisedConfig.frame_count)
    return clips, len(frames)


def c3d_feature_extractor(C3D_model):
    """Creation of the feature extraction architecture. This network is
    formed by a subset of the original C3D architecture (from the
    beginning to fc6 layer)
    :returns: Feature extraction model
    :rtype: keras.model
    """
    layer_name = 'fc6'
    feature_extractor_model = Model(inputs=C3D_model.input,
                                    outputs=C3D_model.get_layer(layer_name).output)
    return feature_extractor_model


def preprocess_input(video):
    """Preprocess video input to make it suitable for feature extraction.
    The video is resized, cropped, resampled and training mean is substracted
    to make it suitable for the network
    :param video: Video to be processed
    :returns: Preprocessed video
    :rtype: np.ndarray
    """

    intervals = np.ceil(np.linspace(0, video.shape[0] - 1, 16)).astype(int)
    frames = video[intervals]

    # Reshape to 128x171
    reshape_frames = np.zeros((frames.shape[0], 128, 171, frames.shape[3]))
    for i, img in enumerate(frames):
        img = cv2.resize(img, (171, 128), cv2.INTER_CUBIC)
        reshape_frames[i, :, :, :] = img

    # mean_path = get_file('c3d_mean.npy',
    #                      C3D_MEAN_PATH,
    #                      cache_subdir='models',
    #                      md5_hash='08a07d9761e76097985124d9e8b2fe34')

    mean = np.load(mean_path)
    reshape_frames -= mean
    # Crop to 112x112
    reshape_frames = reshape_frames[:, 8:120, 30:142, :]
    # Add extra dimension for samples
    reshape_frames = np.expand_dims(reshape_frames, axis=0)

    return reshape_frames


def interpolate(features, features_per_bag):
    """Transform a bag with an arbitrary number of features into a bag
    with a fixed amount, using interpolation of consecutive features
    :param features: Bag of features to pad
    :param features_per_bag: Number of features to obtain
    :returns: Interpolated features
    :rtype: np.ndarray
    """
    feature_size = np.array(features).shape[1]
    interpolated_features = np.zeros((features_per_bag, feature_size))
    interpolation_indices = np.round(np.linspace(0, len(features) - 1, num=features_per_bag + 1))
    count = 0
    for index in range(0, len(interpolation_indices)-1):
        start = int(interpolation_indices[index])
        end = int(interpolation_indices[index + 1])

        assert end >= start

        if start == end:
            temp_vect = features[start, :]
        else:
            temp_vect = np.mean(features[start:end+1, :], axis=0)

        temp_vect = temp_vect / np.linalg.norm(temp_vect)

        if np.linalg.norm(temp_vect) == 0:
            print("Error")

        interpolated_features[count,:]=temp_vect
        count = count + 1

    return np.array(interpolated_features)


def extrapolate(outputs, num_frames):
    """Expand output to match the video length
    :param outputs: Array of predicted outputs
    :param num_frames: Expected size of the output array
    :returns: Array of output size
    :rtype: np.ndarray
    """

    extrapolated_outputs = []
    extrapolation_indices = np.round(np.linspace(0, len(outputs) - 1, num=num_frames))
    for index in extrapolation_indices:
        extrapolated_outputs.append(outputs[int(index)])
    return np.array(extrapolated_outputs)