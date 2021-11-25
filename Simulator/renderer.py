from configuration import MARKET, SECRETE_KEY
import matplotlib.pyplot as plt
import numpy as np


class Renderer:
    plt.style.use("dark_background")

    def __init__(self, screen_size=100):
        self.screen_size = screen_size
        self.line1 = []
        self.y_vec = None
        self.BOOL_SET_YLIM = False
        self.ylim = None
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
    
    def render(self, midpoint=100,tt=None, mode="human", acc_volume=None):
        self.line1 = self.live_plotting(
            self.x_vec,
            self.y_vec,
            self.line1,
            ticker=MARKET,
            mode=mode,
            y2_data=self.y2_vec
        )

        self.y_vec = np.append(self.y_vec[1:], midpoint)
        self.x_vec = np.append(self.x_vec[1:], tt)
        if acc_volume is not None:
            self.y2_vec = np.append(self.y2_vec[1:], acc_volume*100)

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
                fig = plt.figure(figsize=(20, 12))
                ax = fig.add_subplot(111)
                y1_data: np.ndarray
                
                line1, = ax.plot(x_vec, y1_data_, '-', alpha=0.8)
                
                if y2_data is not None:
                    self.axtwin = ax.twinx()
                    self.axtwin.set_ylim(0, 10)
                    line2 =self.axtwin.bar(x_vec, y2_data/100, width=0.01, color='b')
                    # line2.set_ydata(y1_data)
                    # line2.set_xdata(x_vec)
            
            line1.set_ydata(y1_data_)
            line1.set_xdata(x_vec)
            if y2_data is not None:
                self.axtwin.bar(x_vec, y2_data/500, width=0.01, color='b')      
            plt.show(block=False)
            if np.min(y1_data) <= line1.axes.get_ylim()[0] or \
                np.max(y1_data) >= line1.axes.get_ylim()[1]:
                ax = line1.axes
                ax.set_ylim(np.min(y1_data_), np.max(y1_data_))
            plt.xlim(np.min(x_vec), np.max(x_vec))
            plt.pause(0.00001)
        return line1
    
    @staticmethod
    def close():
        plt.close()