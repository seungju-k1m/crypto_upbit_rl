from os import truncate
from pandas.core.algorithms import isin
from Simulator.pipeline_v2 import DataPipeLine_Sim
from datetime import datetime, timedelta
from typing import List
from configuration import RENDER_MODE
from PIL import Image

import PIL

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
        self.plot_unit = plot_unit
        self.size = self.pipe.size
        self.axes = []
        for i in range(4):
            fig = plt.figure(i, figsize=(4, 3))
            ax = fig.add_subplot()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            self.axes.append(ax)

        self.time_vec = {}
        self.opening_vec = {}
        self.high_vec = {}
        self.low_vec = {}
        self.trade_vec = {}
        self.acc_volume_vec = {}

      
        self._init()

    def _init(self):
        data_dict = self.pipe.reset()
        for unit, value in data_dict.items():
            time_np, opening_price_np, high_price_np, low_price_np, trade_price_np, acc_volume_np = value
            self.time_vec[unit] = time_np
            self.opening_vec[unit] = opening_price_np
            self.high_vec[unit] = high_price_np
            self.low_vec[unit] = low_price_np
            self.trade_vec[unit] = trade_price_np
            self.acc_volume_vec[unit] = acc_volume_np
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
            # if lim_info[0] > min(truncated_low_vec):
            #     ax.set_ylim(min(truncated_low_vec), lim_info[1])
            # lim_info = ax.get_ylim()
            # if lim_info[1] < max(truncated_high_vec):
            #     ax.set_ylim(lim_info[0], max(truncated_high_vec))
            ax.set_ylim(min(truncated_low_vec), max(truncated_high_vec))
            lim_info = ax.get_ylim()

            mean_y = truncated_opening_vec.mean()

            delta = min((lim_info[1] - lim_info[0]) / mean_y * 100 / u ** 0.5, 20)
            delta = max(delta, 0.25)
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
            
    def plot(self):

        def update(x):
            lines = []
            for i, u in enumerate([1, 5, 15, 60]):
                output, done = self.pipe.step(self.plot_unit)
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

                    self.time_vec[unit] = update(time_np, prev_time)
                    self.opening_vec[unit] = update(opening_price_np, prev_opening)
                    self.high_vec[unit] = update(high_price_np, prev_high)
                    self.low_vec[unit] = update(low_price_np, prev_low)
                    self.trade_vec[unit] = update(trade_price_np, prev_trade)
                    self.acc_volume_vec[unit] = update(acc_volume_np, prev_acc)
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
                lim_info = ax.get_ylim()
                ax.set_xlim(min(truncated_time), max(truncated_time))
                if lim_info[0] > min(truncated_low_vec) or lim_info[1] < max(truncated_high_vec):
                    ax.set_ylim(min(truncated_low_vec), max(truncated_high_vec))
                mean_y = truncated_opening_vec.mean()
                lim_info = ax.get_ylim()


                delta = min((lim_info[1] - lim_info[0]) / mean_y * 50 / u ** 0.5, 20)
                delta = max(delta, 0.25)
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

                    lines.append(l1)
                    lines.append(l2)

            # self.ax.set_xlim(min(t_line_1), max(t_line_1))
            # self.ax02.set_xlim(min(t_line_1), max(t_line_1))
            return lines
        
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
    
    def get_figure(self) -> np.ndarray:
        # fig.canvas.draw()
        xx = []
        for f in self.figs:
            f.canvas.flush_events()
        
            data = np.fromstring(f.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            fig_np = data.reshape(f.canvas.get_width_height()[::-1] + (3,))
        
            fig_Image = Image.fromarray(fig_np)
            # fig_Image = fig_Image.resize((96, 72), resample=PIL.Image.BICUBIC)
            # fig_Image = fig_Image.resize((96, 72), resample=PIL.Image.BOX)
            # fig_Image = fig_Image.resize((96, 72), resample=PIL.Image.NEAREST)
            fig_Image = fig_Image.resize((96, 72), resample=PIL.Image.HAMMING).convert("L")

            x = np.array(fig_Image)
            xx.append(x)
            # x = np.transpose(x, (2, 0, 1))
        return np.stack(xx, 0)
    
    def init_plot_configuration(self):

        # self.fig, axes = plt.subplots(2, 2)
        self.axes = []
        self.figs = []
        self.bgs = []
        for i in range(4):
            fig = plt.figure(i, figsize=(4,3))
            ax = fig.add_subplot()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            self.axes.append(ax)
            self.figs.append(fig)

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
        for f in self.figs:
            f.canvas.draw()
            self.bgs.append(
                f.canvas.copy_from_bbox(
            f.bbox
        )
            )

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

            delta = min((lim_info[1] - lim_info[0]) / mean_y * 50 / u ** 0.5, 20)
            delta = max(delta, 0.25)
            for l1, l2, tt, ov, hv, lv, tv in zip(
                line01, line02, truncated_time, truncated_opening_vec, truncated_high_vec, truncated_low_vec, truncated_trade_vec):
                if ov > tv:
                    # clr = 'red'
                    clr = 'white'
                else:
                    # clr = 'green'
                    clr = 'white'
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
        for f, g in zip(self.figs, self.bgs):
            f.canvas.blit(g)
        fig = self.get_figure()
        return fig
    
    def render(self):
        lim_infos = []
        for ax in self.axes:
            lim_infos.append(
                deepcopy(
                    ax.get_ylim()
                )
            )
        # self.fig.clear()
        for f, g in zip(self.figs, self.bgs):
            f.canvas.restore_region(g)
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
            
            lim_info = lim_infos[i]

            # if lim_info[0] > min(truncated_low_vec):
            #     ax.set_ylim(min(truncated_low_vec), lim_info[1])
            # lim_info = ax.get_ylim()
            # if lim_info[1] < max(truncated_high_vec):
            #     ax.set_ylim(lim_info[0], max(truncated_high_vec))
            if lim_info[0] > min(truncated_low_vec) or lim_info[1] < max(truncated_high_vec):
                    ax.set_ylim(min(truncated_low_vec), max(truncated_high_vec))
            ax.set_xlim(min(truncated_time), max(truncated_time))

            mean_y = truncated_opening_vec.mean()
            lim_info = ax.get_ylim()

            delta = min((lim_info[1] - lim_info[0]) / mean_y * 50 / u ** 0.5, 20)
            delta = max(delta, 0.25)
            for l1, l2, tt, ov, hv, lv, tv in zip(
                line01, line02, truncated_time, truncated_opening_vec, truncated_high_vec, truncated_low_vec, truncated_trade_vec):
                if ov > tv:
                    # clr = 'red'
                    clr = 'white'
                else:
                    # clr = 'green'
                    clr = 'white'
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
        
        for f in self.figs:
            f.canvas.blit(self.figs[0].bbox)
        fig = self.get_figure()
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
