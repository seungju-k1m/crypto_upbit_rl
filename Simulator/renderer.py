from configuration import MARKET
import matplotlib.pyplot as plt
import numpy as np


class Renderer:
    plt.style.use("dark_background")

    def __init__(self):
        self.screen_size = 100
        self.line1 = []
        self.y_vec = None
        self.x_vec = np.linspace(0, self.screen_size*10, self.screen_size + 1)[0:-1]
    
    def init_data(self, y_vec):
        self.y_vec = y_vec
        self.line1.clear()
    
    def render(self, midpoint=100, mode='human'):
        self.line1 = self.live_plotting(
            self.x_vec,
            self.y_vec,
            self.line1,
            ticker=MARKET
        )
        self.y_vec = np.append(self.y_vec[1:], midpoint)
    
    @staticmethod
    def live_plotting(x_vec, y1_data, line1, ticker='KRW-BTC'):
        if not line1:
            plt.ion()
            fig = plt.figure(figsize=(20, 12))
            ax = fig.add_subplot(111)
            line1, = ax.plot(x_vec, y1_data, '-', label='midpoint', alpha=0.8)
            plt.ylabel('Price [KRW]')
            plt.legend()
            plt.title('Title: {}'.format(ticker))
            plt.show(block=False)
        
        line1.set_ydata(y1_data)

        if np.min(y1_data) <= line1.axes.get_ylim()[0] or \
            np.max(y1_data) >= line1.axes.get_ylim()[1]:
            plt.ylim(np.min(y1_data), np.max(y1_data))
        plt.pause(0.00001)
        return line1
    
    @staticmethod
    def close():
        plt.close()