from configuration import MARKET, RENDER_MODE, SECRETE_KEY, UNIT_MINUTE
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import PIL


class Renderer:
    plt.style.use("dark_background")
    plt.axis("off")

    def __init__(self, screen_size=48, unit=1):
        self.screen_size = screen_size
        self.line1 = []
        self.y_vec = None
        self.BOOL_SET_YLIM = False
        self.ylim = None
        self.width = 0.001 * unit
        self.unit = unit
        self.y2_div = 1 / (10 * unit)
        color = ['w', 'w', 'w', 'w']
        idx = UNIT_MINUTE.index(unit)
        self.color = color[idx]
        # self.x_vec = np.linspace(0, self.screen_size*10, self.screen_size + 1)[0:-1]
    
    def init_data(self, x_vec, y_vec, y2_vec=None):
        self.x_vec = x_vec
        self.y_vec = y_vec
        self.y2_vec = y2_vec
        # if y2_vec is not None:
        #     self.y2_vec = y2_vec
        self.line1 = []
    
    def set_ylim(self, ymin, ymax):
        self.ylim = [ymin, ymax]
        self.BOOL_SET_YLIM = True
    
    def render(self, state=None):
        if state is None:
            self.line1, obs = self.live_plotting(
                self.x_vec,
                self.y_vec,
                self.line1,
                ticker=MARKET,
                y2_data=self.y2_vec,
                mode='2'
            )
            return None
        tt, midpoint = state[:2]
        acc_volume = state[-2]
        self.line1, obs = self.live_plotting(
            self.x_vec,
            self.y_vec,
            self.line1,
            ticker=MARKET,
            y2_data=self.y2_vec,
            mode='2'
        )
        self.y_vec = np.append(self.y_vec[1:], midpoint)
        self.x_vec = np.append(self.x_vec[1:], tt)
        if acc_volume is not None:
            self.y2_vec = np.append(self.y2_vec[1:], acc_volume)
        return obs

    def live_plotting(self, x_vec, y1_data, line1, ticker='KRW-BTC', y2_data=None, mode="human"):
        
        if mode == "human":
            if not line1:
                plt.ion()
                fig = plt.figure(figsize=(20, 12))
                ax = fig.add_subplot(111)
                line1, = ax.plot(x_vec, y1_data/1e7, '-', alpha=0.8)
                plt.ylabel('Price [1e7KRW]')
                plt.legend()
                plt.title('Title: {}'.format(ticker))
                plt.show(block=False)
            
            line1.set_ydata(y1_data)
            line1.set_xdata(x_vec)

            if self.BOOL_SET_YLIM:
                plt.ylim(*self.ylim)
            else:
                if np.min(y1_data) <= line1.axes.get_ylim()[0] or \
                    np.max(y1_data) >= line1.axes.get_ylim()[1]:
                    plt.ylim(np.min(y1_data), np.max(y1_data))
                
            plt.xlim(np.min(x_vec), np.max(x_vec))
            plt.pause(0.00001)
        else:
            y1_data_ =  y1_data / 1e7
            # y1_data = (y1_data - y1_data.mean()) / y1_data.var()
            if not line1:
                plt.ion()
                fig = plt.figure(figsize=(4, 3))
                ax = fig.add_subplot(111)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                y1_data: np.ndarray
                
                line1, = ax.plot(x_vec, y1_data_, '-', alpha=0.8)
                
                if y2_data is not None:
                    axtwin = ax.twinx()
                    axtwin.set_ylim(0, 10)
                    axtwin.get_xaxis().set_visible(False)
                    axtwin.get_yaxis().set_visible(False)
                    self.line2 =axtwin.bar(x_vec, y2_data * self.y2_div, width=self.width, color=self.color)
                    self.k = 0
                    a1 = self.line2[0].get_x()
                    a2 = self.line2[1].get_x()
                    self.k = a2 - a1
                    # line2.set_ydata(y1_data)
                    # line2.set_xdata(x_vec)
            
            line1.set_ydata(y1_data_)
            line1.set_xdata(x_vec)
            if y2_data is not None:
                for l, y in zip(self.line2, y2_data):
                    l._height = y * self.y2_div
                    xx = l.get_x()
                    l.set_x(xx + self.k)
            
            obs = self.figure_to_array(line1.figure)
            obs = Image.fromarray(obs).convert("L")
            # obs = obs.resize((96, 72), PIL.Image.BILINEAR)
            # obs = obs.resize((96, 72), PIL.Image.BICUBIC)
            # obs = obs.resize((96, 72), PIL.Image.NEAREST)
            obs = obs.resize((96, 72), PIL.Image.BOX)
            obs = np.array(obs)
            if RENDER_MODE:
                plt.show(block=False)
                plt.pause(0.00001)
            
            if np.min(y1_data) <= line1.axes.get_ylim()[0] or \
                np.max(y1_data) >= line1.axes.get_ylim()[1]:
                ax = line1.axes
                ax.set_ylim(np.min(y1_data_) * .9, np.max(y1_data_))
            plt.xlim(np.min(x_vec), np.max(x_vec))
            
        return line1, obs
    
    def get_current_fig(self):
        a = self.figure_to_array(self.line1.figure)
        obs = Image.fromarray(a).convert("L")
        obs = obs.resize((96, 72), PIL.Image.BOX)
        obs = np.array(obs)
        return obs


    @staticmethod
    def close():
        plt.close()
    
    @staticmethod
    def figure_to_array(fig):
        fig.canvas.draw()
        return np.array(fig.canvas.renderer._renderer)