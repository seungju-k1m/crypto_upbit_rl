from os import truncate
from pandas.core.algorithms import isin
from Simulator.pipeline_v1 import DataPipeLine_Sim
from datetime import datetime, timedelta
from typing import List
from configuration import RENDER_MODE
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import time
from copy import deepcopy

matplotlib.use("Agg")
template = {'time':[], 'midpoint':[], 'acc_volume':[]}


class Renderer:
    plt.style.use("dark_background")
    plt.axis("off")

    def __init__(
        self,
        pipeline:DataPipeLine_Sim,
        which_plot=None
    ):
        self.pipe = pipeline

        self.which_plot = which_plot
        self.offset = 24

        # x, y -> 4개
        self.x_vec = {}
        self.y_vec = {}
        self.y2_vec = {}
        self.size = self.pipe.offset * 24

        # plot !!
        # 4개의 line이 있어야한다.
        self.axes = []
        self.axes_twin = []
        self.lines = []
        self.lines_2 = []
        self.interval = []
        self.y2_div = [1 / (20*unit) for unit in [1, 5, 15, 60]]
        # self.y2_div = [1, 1/25, 1/50, 1/100]
        # for i in range(4):
        #     fig = plt.figure(i, figsize=(20, 12))
        #     ax = fig.addd_subplot(111)
        #     self.axes.append(ax)
        
        # self.lines = []
        self.init_plot_configuration()

        self.bgs = []
    
    @staticmethod
    def get_figure(fig, plot=False) -> np.ndarray:
        m = time.time()
        fig.canvas.draw()
        # fig.canvas.flush_events()
        
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        fig_np = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        # fig_np = np.array(fig.canvas.renderer._renderer)
        fig_Image = Image.fromarray(fig_np).convert("L")
        fig_Image = fig_Image.resize((96, 72), Image.BOX)
        
        # fig_Image.show()
        if plot:
            fig_Image.show()
        return np.array(fig_Image)

    def render(self):
        image = []
        delta = [0.005, 0.02, 0.05, 0.1]
        # j = time.time()
        for i, unit in enumerate([1, 5, 15, 60]):
            x, y, y2 = self.x_vec[unit], self.y_vec[unit], self.y2_vec[unit]

            truncated_x = x[-self.offset:]
            truncated_y = y[-self.offset:]
            truncated_y2 = y2[-self.offset:]

            truncated_x = np.array([np.datetime64(i) for i in truncated_x])
            
            line1, line2 = self.lines[i], self.lines_2[i]
            ax, ax_twin = self.axes[i], self.axes_twin[i]

            fig = line1.figure

            # fig.canvas.restore_region(self.bgs[i])

            line1.set_ydata(truncated_y)
            line1.set_xdata(truncated_x)

            mean_y = truncated_y.mean()

            k = delta[i]
            
            ax.set_ylim(mean_y * (1- k), mean_y * (1 + k))

            # ax.set_ylim(min(truncated_y) * 0.9, max(truncated_y)* 1.1)
            ax.set_xlim(min(truncated_x), max(truncated_x))

            for l, y2_each in zip(line2, truncated_y2):
                l._height = y2_each * self.y2_div[i]
                # l._height = 100
                xx = l.get_x()
                l.set_x(xx+self.interval[i])
            
            # ax.draw_artist(line1)
            # ax_twin.draw_artist(line2)

            # image_tmp = self.figure_to_array(line1.figure)
            image_np = self.get_figure(fig, plot=False)
            # fig.canvas.blit(fig.bbox)
            # fig.canvas.flush_events()

            image.append(image_np)
        # print(time.time() - j)
        image = np.stack(image, axis=0)
        return image

    def init_plot_configuration(self):
        for i in range(4):
            fig = plt.figure(i, figsize=(4, 3))
            ax = fig.add_subplot(111)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            self.axes.append(ax)

    def reset(self):
        # self.x_vec.clear()
        # self.y_vec.clear()
        # self.y2_vec.clear()
        # self.lines.clear()
        # self.lines_2.clear()
        
        # self.axes.clear()
        # self.axes_twin.clear()
        # self.interval.clear()
        # self.bgs.clear()
        # self.close()

        data_dict = self.pipe.reset()
        # key : 1, 5, 15, 60
        # value: [time, midpoint, acc_volume in each np.ndarray]

        for unit, value in data_dict.items():
            time_np, midpoint_np, acc_volume_np = value
            self.x_vec[unit] = time_np
            self.y_vec[unit] = midpoint_np
            self.y2_vec[unit] = acc_volume_np
        
        # init plot configuration !!
        uu = [1, 5, 15, 60]
        for i, u in enumerate(uu):
            """
                ax_twin,

                line01,
                line02

                interval between acc_volume
            """
            ax = self.axes[i]

            if len(self.axes_twin) > 3:
                ax_twin = self.axes_twin[i]
            else:
                ax_twin = ax.twinx()
                ax_twin.set_ylim(0, 5)
                ax_twin.get_xaxis().set_visible(False)
                ax_twin.get_yaxis().set_visible(False)

                self.axes_twin.append(ax_twin)

            x, y, y2 = self.x_vec[u], self.y_vec[u], self.y2_vec[u]
            truncated_x = x[-self.offset:]
            truncated_y = y[-self.offset:]
            truncated_y2 = y2[-self.offset:]
            # truncated_x = np.datetime64(truncated_x)
            truncated_x = np.array([np.datetime64(i) for i in truncated_x])
            
            if len(self.lines) > 3:
                line = self.lines[i]
                line.set_ydata(truncated_y)
                line.set_xdata(truncated_x)
            else:
                line,  = ax.plot(truncated_x, truncated_y, '-', alpha=0.8)
                self.lines.append(line)

            y:np.ndarray
            # mean_y = truncated_y.mean()
            ax.set_ylim(min(truncated_y), max(truncated_y))
            ax.set_xlim(min(truncated_x), max(truncated_x))
            # ax.set_ylim(mean_y * 0.98, mean_y * 1.02)

            if len(self.lines_2) > 3:
                line2 = self.lines_2[i]
                for l, y2_each in zip(line2, truncated_y2):
                    l._height = y2_each * self.y2_div[i]
                    # l._height = 100
                    xx = l.get_x()
                    l.set_x(xx+self.interval[i])
            else:
                line2 = ax_twin.bar(truncated_x, truncated_y2 *self.y2_div[i], width=0.0005 * u, color='w')
            # 간격
                a1 = line2[0].get_x()
                a2 = line2[1].get_x()
                self.interval.append(a2 - a1)
                self.lines_2.append(line2)
        
        image = self.render()

        return (image, (deepcopy(self.y_vec), deepcopy(self.y2_vec)))
        
    def step(self, duration=15):
        # updating x_vec, y_vec, y2_vec
        # updating unit : 15min

        output, done = self.pipe.step(duration)

        def update(new_data:np.ndarray, prev_data:np.ndarray):
            len_data = len(prev_data)
            data = np.concatenate((prev_data, new_data), 0)
            data = data[-len_data:]
            return data

        for unit, value in output.items():
            time_np, midpoint_np, acc_volume_np = value

            prev_time_np = self.x_vec[unit]
            prev_midpoint_np = self.y_vec[unit]
            prev_acc_volume_np = self.y2_vec[unit]

            time_np = update(time_np, prev_time_np)
            midpoint_np = update(midpoint_np, prev_midpoint_np)
            acc_volume_np = update(acc_volume_np, prev_acc_volume_np)

            self.x_vec[unit] = time_np
            self.y_vec[unit] = midpoint_np
            self.y2_vec[unit] = acc_volume_np
        
        image = self.render()
        return (image, (deepcopy(self.y_vec), deepcopy(self.y2_vec))), done

    @staticmethod
    def close():
        plt.close()

    @property
    def count(self):
        return self.pipe.count
