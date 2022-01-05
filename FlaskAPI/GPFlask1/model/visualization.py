import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import animation
import numpy as np


def visualize_predictions(video_path, predictions, save_path):
    fig = plt.figure()

    fig_prediction = plt.subplot(1, 1, 1)
    fig_prediction.set_xlim(0, len(predictions))
    fig_prediction.set_ylim(0, 1.15)

    def update(i):
        x = range(0, i)
        y = predictions[0:i]
        fig_prediction.plot(x, y, '-')

        return plt

    # FuncAnimation will call the 'update' function for each frame; here
    # animating over 10 frames, with an interval of 20ms between frames.

    anim = animation.FuncAnimation(fig, update, frames=np.arange(0, len(predictions), 10), interval=1, repeat=False)
    #f = "C:/Users/Administrator/Desktop/GP Project/output_folder/animation_test.gif"
    #writergif = animation.PillowWriter(fps=16)
    #anim.save(f, writer=writergif)

    #f = "app/static/sample/animation_vid.mp4"
    writervideo = animation.FFMpegWriter(fps=16)
    anim.save(save_path, writer=writervideo)