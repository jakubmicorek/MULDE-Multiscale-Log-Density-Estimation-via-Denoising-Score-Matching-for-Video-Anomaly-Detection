import numpy as np
import matplotlib.pyplot as plt

def plot_mesh(plt_, xx, yy, data, colorbar_label, cmap='coolwarm', plot_scatter=False, interpolation='bilinear'):
    plt_.imshow(data, extent=(xx.min(), xx.max(), yy.min(), yy.max()), origin='lower', cmap=cmap, interpolation=interpolation)
    plt_.colorbar(label=colorbar_label, ticks=np.linspace(data.min(), data.max(), 10))
    if plot_scatter:
        plt_.scatter(xx, yy, color='black', s=0.5, label="manifold")
