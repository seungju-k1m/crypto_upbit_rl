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

from matplotlib import gridspec
from matplotlib import animation

matplotlib.use("Agg")
template = {'time':[], 'midpoint':[], 'acc_volume':[]}


class Animator:
    plt.style.use("dark_background")
    plt.axis("off")
    plt.ioff()

    def __init__(
        self,
        pipeline:DataPipeLine_Sim,
        plot_unit = 1
    ):
        self.pipe = pipeline
        self.plot_unit = 1

        self.x_vec, self.y_vec, self.y2_vec = [], [], []

        spec = gridspec.GridSpec(ncols=1, nrows=2,
                         width_ratios=[1], wspace=0.5,
                         hspace=0.5, height_ratios=[3, 1])
        
        self.fig = plt.figure(figsize=(4, 3))

        self.ax = self.fig.add_subplot(spec[0])
        self.ax02 = self.fig.add_subplot(spec[1])

        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)

        self.ax02.get_xaxis().set_visible(False)
        self.ax02.get_yaxis().set_visible(False)

        self.ax02.set_ylim(0, 5)
        self.plot_unit = plot_unit
        self.y2_div = [1 / (5*unit) for unit in [1, 5, 15, 60]]
        self._init()

    def _init(self):
        data_dict = self.pipe.reset()
        time_np, mindpoint_np, acc_volume_np = data_dict[self.plot_unit]

        self.line, = self.ax.plot(time_np, mindpoint_np, '-', alpha=0.8, linewidth=2, animated=True)
        i = [1, 5, 15, 60].index(self.plot_unit)
        self.line2, = self.ax02.plot(time_np, acc_volume_np * self.y2_div[i], '-', alpha=0.8, linewidth=2, animated=True)

    def plot(self):
        i = [1, 5, 15, 60].index(self.plot_unit)
        delta = [0.01/2, 0.02/2, 0.05/2, 0.1/2]
        k = delta[i]
        def update(x):
            t_line_1 = self.line.get_xdata()
            mid_line = self.line.get_ydata()
            acc_line = self.line2.get_ydata()

            len_data =len(t_line_1)

            output, done = self.pipe.step(self.plot_unit)
            output = output[self.plot_unit]
            t, mid, vol = output

            vol = vol * self.y2_div[i]

            # t_line_1 = np.concatenate((t_line_1, t), 0)
            # t_line_1 = t_line_1[-len_data:]

            mid_line = np.concatenate((mid_line, mid), 0)
            mid_line = mid_line[-len_data:]

            acc_line = np.concatenate((acc_line, vol), 0)
            acc_line = acc_line[-len_data:]

            # self.line.set_xdata(t_line_1)
            self.line.set_ydata(mid_line)
            

            # self.line2.set_xdata(t_line_1)
            self.line2.set_ydata(acc_line)

            mean_y = float(mid_line.mean())

            lim_info = self.ax.get_ylim()
            # print(linewidth)
            
            # self.line2.set_linewidth(linewidth)

            if np.min(mid_line) <= lim_info[0]:
                self.ax.set_ylim(np.min(mid_line), lim_info[1])
                
            lim_info = self.ax.get_ylim()
            if np.max(mid_line) >= lim_info[1]:
                self.ax.set_ylim(lim_info[0], max(mid_line))
            
            lim_info = self.ax.get_ylim()

            delta = (lim_info[1] - lim_info[0]) / mean_y

            linewidth = delta * 100
            self.line.set_linewidth(linewidth)


            # self.ax.set_xlim(min(t_line_1), max(t_line_1))
            # self.ax02.set_xlim(min(t_line_1), max(t_line_1))

            return self.line, self.line2
        
        offset = self.pipe.offset
        size = self.pipe.size
        data_len = self.pipe.data.length_info[self.plot_unit]

        m = data_len - int(offset/self.plot_unit) - size
        zxz = [i for i in range(m-10)]
        anime = animation.FuncAnimation(
            self.fig, update, frames=zxz, blit=True
        )
        anime.save('./test.gif')


class Renderer:
    plt.style.use("dark_background")
    plt.axis("off")
    plt.ioff()

    def __init__(
        self,
        pipeline:DataPipeLine_Sim,
        which_plot=None
    ):
        self.pipe = pipeline

        self.which_plot = which_plot
        self.offset = self.pipe.size

        # x, y -> 4개
        self.x_vec = {}
        self.y_vec = {}
        self.y2_vec = {}
        self.size = self.pipe.offset * self.offset

        # plot !!
        # 4개의 line이 있어야한다.
        self.axes = []
        self.axes_twin = []
        self.lines = []
        self.lines_fill = []
        self.lines_2 = []
        self.interval = []
        self.y2_div = [1 / (5*unit) for unit in [1, 5, 15, 60]]

        self.init_plot_configuration()

        self.bgs = []
        self.Image = [False for i in range(4)]
    
    @staticmethod
    def get_figure(fig, plot=False) -> np.ndarray:
        m = time.time()
        # fig.canvas.draw()
        fig.canvas.flush_events()
        
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        fig_np = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        # fig_np = np.array(fig.canvas.renderer._renderer)
        fig_Image = Image.fromarray(fig_np).convert("L")
        fig_Image = fig_Image.resize((96, 72), Image.BOX)
        # fig_Image.show()

        if plot:
            fig_Image.show()
        return np.array(fig_Image)

    def render(self, cond):
        image = []
        j = time.time()
        for i, unit in enumerate([1, 5, 15, 60]):
            if not cond[i]:
                
                image_np = self.Image[i]
                image.append(image_np)
                continue

            x, y, y2 = self.x_vec[unit], self.y_vec[unit], self.y2_vec[unit]

            truncated_x = x[-self.offset:]
            truncated_y = y[-self.offset:]
            truncated_y2 = y2[-self.offset:]

            truncated_x = np.array([np.datetime64(i) for i in truncated_x])
            
            line1, line2 = self.lines[i], self.lines_2[i]
            ax, ax_twin = self.axes[i], self.axes_twin[i]
            # line_fill = self.lines_fill[i]

            fig = line1.figure
            fig.clear()
            bg = self.bgs[i]
            fig.canvas.restore_region(bg)

            line1.set_ydata(truncated_y)
            line1.set_xdata(truncated_x)

            line2.set_ydata(truncated_y2 * self.y2_div[i])
            line2.set_xdata(truncated_x)

            # l_fill = ax.fill_between(truncated_x, truncated_y, 0,  facecolor = 'C0', alpha = 0.2)
            # l_fill_2 = ax_twin.fill_between(truncated_x, truncated_y2 * self.y2_div[i], 0, facecolor= 'C0', alpha=0.4)
            mean_y = truncated_y.mean()
            lim_info = ax.get_ylim()
            # if np.min(truncated_y) <= lim_info[0]:
            #     ax.set_ylim(np.min(truncated_y), lim_info[1])
            # lim_info = ax.get_ylim()
            # if np.max(truncated_y) >= lim_info[1]:
            #     ax.set_ylim(lim_info[0], max(truncated_y))
            
            
            if np.min(truncated_y) <= lim_info[0]:
                self.ax.set_ylim(np.min(truncated_y), lim_info[1])
                
            lim_info = self.ax.get_ylim()
            if np.max(truncated_y) >= lim_info[1]:
                self.ax.set_ylim(lim_info[0], max(truncated_y))
            
            lim_info = self.ax.get_ylim()

            delta = (lim_info[1] - lim_info[0]) / mean_y

            linewidth = delta * 100
            self.line.set_linewidth(linewidth)
            
            ax.set_xlim(min(truncated_x), max(truncated_x))

            ax_twin.set_xlim(min(truncated_x), max(truncated_x))

            ax.draw_artist(line1)
            ax_twin.draw_artist(line2)

            # ax.draw_artist(l_fill)
            # ax_twin.draw_artist(l_fill_2)
            
            fig.canvas.blit(fig.bbox)
            image_np = self.get_figure(fig, plot=False)
            self.Image[i] = deepcopy(image_np)
            image.append(image_np)
        # print(time.time() - j)
        image = np.stack(image, axis=0)
        return image

    def init_plot_configuration(self):
        spec = gridspec.GridSpec(ncols=1, nrows=2,
                         width_ratios=[1], wspace=0.5,
                         hspace=0.5, height_ratios=[3, 1])
        for i in range(4):
            fig = plt.figure(i, figsize=(4, 3))
            ax = fig.add_subplot(spec[0])
            ax02 = fig.add_subplot(spec[1])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            ax02.get_xaxis().set_visible(False)
            ax02.get_yaxis().set_visible(False)
            ax02.set_ylim(0, 12)
            self.axes.append(ax)
            self.axes_twin.append(ax02)

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
        delta = [0.005/2, 0.02/2, 0.05/2, 0.1/2]
        for i, u in enumerate(uu):
            """
                ax_twin,

                line01,
                line02

                interval between acc_volume
            """
            ax = self.axes[i]
            ax02 = self.axes_twin[i]
            fig = ax.figure
            

            x, y, y2 = self.x_vec[u], self.y_vec[u], self.y2_vec[u]
            truncated_x = x[-self.offset:]
            truncated_y = y[-self.offset:]
            truncated_y2 = y2[-self.offset:]
            truncated_x = np.array([np.datetime64(i) for i in truncated_x])
            
            if len(self.lines) > 3:
                line = self.lines[i]
                line.set_ydata(truncated_y)
                line.set_xdata(truncated_x)
            else:
                ax.plot([0], [0])
                
                fig.canvas.draw()
                bg = fig.canvas.copy_from_bbox(fig.bbox)
                self.bgs.append(bg)
                line,  = ax.plot(truncated_x, truncated_y, '-', alpha=0.8, animated=True)
                # line_fill = ax.fill_between(truncated_x, truncated_y, animated=True)
                # line,  = ax.plot(truncated_x, truncated_y, '-', alpha=0.8)
                self.lines.append(line)
                # self.lines_fill.append(line_fill)
                
                

            y:np.ndarray
            mean_y = float(truncated_y.mean())
            min_y_ = float(truncated_y.min())
            max_y_ = float(truncated_y.max())

            k = delta[i]

            min_y = min(mean_y * (1 - k), min_y_)

            max_y = max(mean_y * (1 + k), max_y_)
            
            ax.set_ylim(min_y, max_y)
            ax.set_xlim(min(truncated_x), max(truncated_x))
            # ax.set_ylim(mean_y * 0.98, mean_y * 1.02)

            if len(self.lines_2) > 3:
                pass
                
            else:
                line2,  = ax02.plot(truncated_x, truncated_y2 * self.y2_div[i], '-', alpha=0.8, animated=True)
                self.lines_2.append(line2)
                ax.draw_artist(line)
                ax02.draw_artist(line2)

                fig.canvas.blit(fig.bbox)
                # fig.canvas.draw()
        cond = [True for i in range(4)]
        image = self.render(cond)

        return (image, (deepcopy(self.y_vec), deepcopy(self.y2_vec)))
        
    def step(self, duration=15):
        # updating x_vec, y_vec, y2_vec
        # updating unit : 15min

        output, done = self.pipe.step(duration)
        cond = [False for i in range(4)]

        def update(new_data:np.ndarray, prev_data:np.ndarray):
            len_data = len(prev_data)
            data = np.concatenate((prev_data, new_data), 0)
            data = data[-len_data:]
            return data
        j = 0
        for unit, value in output.items():
            time_np, midpoint_np, acc_volume_np = value

            prev_time_np = self.x_vec[unit]
            prev_midpoint_np = self.y_vec[unit]
            prev_acc_volume_np = self.y2_vec[unit]
            if len(time_np) > 0:
                cond[j] = True
            j += 1
            time_np = update(time_np, prev_time_np)
            midpoint_np = update(midpoint_np, prev_midpoint_np)
            acc_volume_np = update(acc_volume_np, prev_acc_volume_np)

            self.x_vec[unit] = time_np
            self.y_vec[unit] = midpoint_np
            self.y2_vec[unit] = acc_volume_np
        
        image = self.render(cond)
        return (image, (deepcopy(self.y_vec), deepcopy(self.y2_vec))), done

    @staticmethod
    def close():
        plt.close()

    @property
    def count(self):
        return self.pipe.count
