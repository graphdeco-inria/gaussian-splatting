from matplotlib import pyplot as plt
import matplotlib.colors as colors
from matplotlib.animation import FuncAnimation
import numpy as np

class GifRenderer:
    def __init__(self, num_rows, num_cols, figsize=(10, 5), gif_interval=1):
        self.fig, self.axes = plt.subplots(num_rows, num_cols, figsize=figsize)
        if num_rows == 1:
            self.axes = self.axes[np.newaxis, :]
        self.fig.subplots_adjust(hspace=0.6)
        self.ims = [[None] * num_cols for _ in range(num_rows)]
        self.images_series = [[None] * num_cols for _ in range(num_rows)]
        self.max_frames = [[0] * num_cols for _ in range(num_rows)]
        self.losses = [[None] * num_cols for _ in range(num_rows)]
        self.titles = [[""] * num_cols for _ in range(num_rows)]
        self.gif_interval = gif_interval

    def add_gt(self, row, col, image_torch):
        ax = self.axes[row, col]
        im = ax.imshow(image_torch.permute(1, 2, 0).cpu().numpy(), animated=True)
        ax.axis('off')
        ax.set_title("Ground Truth")
        self.ims[row][col] = im

    def add_series(self, row, col, images_torch, losses, title=""):
        ax = self.axes[row, col]
        self.titles[row][col] = title
        self.max_frames[row][col] = len(images_torch)
        self.images_series[row][col] = [images_torch[i].permute(1, 2, 0).cpu().numpy() for i in range(len(images_torch))]
        self.losses[row][col] = losses
        im = ax.imshow(self.images_series[row][col][0], animated=True)
        ax.axis('off')
        ax.set_title(f"Iteration {0}, Loss: {losses[0]:.6e}")
        self.ims[row][col] = im

    def animate(self, save_path, interval=200):
        def update(frame):
            for row in range(len(self.ims)):
                for col in range(len(self.ims[row])):
                    im = self.ims[row][col]
                    if im is not None and frame < self.max_frames[row][col] and frame % self.gif_interval == 0:
                        im.set_array(self.images_series[row][col][frame])
                        loss = self.losses[row][col][frame]
                        title = self.titles[row][col]
                        self.axes[row, col].set_title(f"{title}\nIteration {frame}, Loss: {loss:.3e}")
            return [im for row in self.ims for im in row if im is not None]

        ani = FuncAnimation(self.fig, update, frames=max([max(frames) for frames in self.max_frames]), 
                            interval=interval, blit=True)
        ani.save(save_path, writer='pillow')
