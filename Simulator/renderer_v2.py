from os import truncate
from pandas.core.algorithms import isin
from Simulator.pipeline_v2 import DataPipeLine_Sim
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
        self.size = self.pipe.size

        self.time_vec = {}
        self.opening_vec = {}
        self.high_vec = {}
        self.low_vec = {}
        self.trade_vec = {}
        self.acc_volume_vec = {}

        self.init_plot_configuration()
    
    @staticmethod
    def get_figure(fig, plot=False) -> np.ndarray:
        # fig.canvas.draw()
        fig.canvas.flush_events()
        
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        fig_np = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        # fig_np = np.array(fig.canvas.renderer._renderer)
        fig_Image = Image.fromarray(fig_np)
        fig_Image = fig_Image.resize((96* 2, 72 * 2))
        # fig_Image.show()

        if plot:
            fig_Image.show()
        x = np.array(fig_Image)
        x = np.transpose(x, (2, 0, 1))
        return x
    
    def init_plot_configuration(self):

        self.fig, axes = plt.subplots(2, 2)
        self.axes = []
        self.axes.append(axes[0][0])
        self.axes.append(axes[0][1])
        self.axes.append(axes[1][0])
        self.axes.append(axes[1][1])
        self.lines = {}
        self.lines_2 = {}
        for i in [1, 5, 15, 60]:
            self.lines[i] = []
            self.lines_2[i] = []
        for ax, i in zip(self.axes, [1, 5, 15, 60]):
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            for _ in range(self.size):
                line, = ax.plot([0], [0], lw=1, animated=True)
                self.lines[i].append(line)

                line2, = ax.plot([0], [0], lw=3, animated=True)
                self.lines_2[i].append(line2)
        self.fig.canvas.draw()
        self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)

    def reset(self):

        data_dict = self.pipe.reset()
        for unit, value in data_dict.items():
            time_np, opening_price_np, high_price_np, low_price_np, trade_price_np, acc_volume_np = value
            self.time_vec[unit] = time_np
            self.opening_vec[unit] = opening_price_np
            self.high_vec[unit] = high_price_np
            self.low_vec[unit] = low_price_np
            self.trade_vec[unit] = trade_price_np
            self.acc_volume_vec[unit] = acc_volume_np
        
        uu = [1, 5, 15, 60]
        for i, u in enumerate(uu):
            truncated_time = self.time_vec[u][-self.size:]
            truncated_time = np.array([np.datetime64(i) for i in truncated_time])
            truncated_opening_vec = self.opening_vec[u][-self.size:]
            truncated_high_vec = self.high_vec[u][-self.size:]
            truncated_low_vec = self.low_vec[u][-self.size:]
            truncated_trade_vec = self.trade_vec[u][-self.size:]
            line01 = self.lines[u]
            line02 = self.lines_2[u]

            # setting ylim and xlim
            ax = self.axes[i]
            ax.set_xlim(min(truncated_time), max(truncated_time))
            ax.set_ylim(min(truncated_low_vec), max(truncated_high_vec))
            lim_info = ax.get_ylim()

            mean_y = truncated_opening_vec.mean()

            delta = min((lim_info[1] - lim_info[0]) / mean_y * 100, 2)
            delta = max(delta, 0.5)
            for l1, l2, tt, ov, hv, lv, tv in zip(
                line01, line02, truncated_time, truncated_opening_vec, truncated_high_vec, truncated_low_vec, truncated_trade_vec):
                if ov > tv:
                    clr = 'red'
                else:
                    clr = 'green'
                l1.set_xdata([tt, tt])
                l2.set_xdata([tt, tt])

                l1.set_ydata([ov, tv])
                l2.set_ydata([lv, hv])

                l1.set_color(clr)
                l2.set_color(clr)

                l1.set_linewidth(delta * 3)
                l2.set_linewidth(delta * 1)
            
                ax.draw_artist(l1)
                ax.draw_artist(l2)
        
        self.fig.canvas.blit(self.bg)
        fig = self.get_figure(self.fig)
        return fig
    
    def render(self):
        # self.fig.clear()
        self.fig.canvas.restore_region(self.bg)
        for i, u in enumerate([1, 5, 15, 60]):
            truncated_time = self.time_vec[u][-self.size:]
            truncated_time = np.array([np.datetime64(i) for i in truncated_time])
            truncated_opening_vec = self.opening_vec[u][-self.size:]
            truncated_high_vec = self.high_vec[u][-self.size:]
            truncated_low_vec = self.low_vec[u][-self.size:]
            truncated_trade_vec = self.trade_vec[u][-self.size:]
            line01 = self.lines[u]
            line02 = self.lines_2[u]

            # setting ylim and xlim
            ax = self.axes[i]
            
            lim_info = ax.get_ylim()

            # if lim_info[0] > min(truncated_low_vec):
            #     ax.set_ylim(min(truncated_low_vec), lim_info[1])
            # lim_info = ax.get_ylim()
            # if lim_info[1] < max(truncated_high_vec):
            #     ax.set_ylim(lim_info[0], max(truncated_high_vec))
            ax.set_ylim(min(truncated_low_vec), max(truncated_high_vec))
            ax.set_xlim(min(truncated_time), max(truncated_time))

            mean_y = truncated_opening_vec.mean()
            lim_info = ax.get_ylim()


            delta = min((lim_info[1] - lim_info[0]) / mean_y * 100, 2)
            delta = max(delta, 0.5)
            for l1, l2, tt, ov, hv, lv, tv in zip(
                line01, line02, truncated_time, truncated_opening_vec, truncated_high_vec, truncated_low_vec, truncated_trade_vec):
                if ov > tv:
                    clr = 'red'
                else:
                    clr = 'green'
                l1.set_xdata([tt, tt])
                l2.set_xdata([tt, tt])

                l1.set_ydata([ov, tv])
                l2.set_ydata([lv, hv])

                l1.set_color(clr)
                l2.set_color(clr)

                l1.set_linewidth(delta * 3)
                l2.set_linewidth(delta * 1)
            
                ax.draw_artist(l1)
                ax.draw_artist(l2)

        self.fig.canvas.blit(self.fig.bbox)
        fig = self.get_figure(self.fig)
        return fig

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
            time_np, opening_price_np, high_price_np, low_price_np, trade_price_np, acc_volume_np = value
            prev_time = self.time_vec[unit]
            prev_opening = self.opening_vec[unit]
            prev_high = self.high_vec[unit]
            prev_low = self.low_vec[unit]
            prev_trade = self.trade_vec[unit]
            prev_acc = self.acc_volume_vec[unit]

            if len(time_np) > 0:
                cond[j] = True
            j += 1
            self.time_vec[unit] = update(time_np, prev_time)
            self.opening_vec[unit] = update(opening_price_np, prev_opening)
            self.high_vec[unit] = update(high_price_np, prev_high)
            self.low_vec[unit] = update(low_price_np, prev_low)
            self.trade_vec[unit] = update(trade_price_np, prev_trade)
            self.acc_volume_vec[unit] = update(acc_volume_np, prev_acc)
        
        image = self.render()
        return (image, deepcopy(self.trade_vec)), done

    @staticmethod
    def close():
        plt.close()

    @property
    def count(self):
        return self.pipe.count
