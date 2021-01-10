from imageio import imread
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm

from setup import create_designmatrix


# Load the terrain
terrain = imread('SRTM_data_Norway_1.tif')


N = 300

terrain = terrain[:int(N), :N]

# Creates mesh of image pixels
x1 = np.linspace(0, 300, np.shape(terrain)[0])
x2 = np.linspace(0, 300, np.shape(terrain)[1])
x1_mesh, x2_mesh = np.meshgrid(x1, x2)


def plot_terrain(x, y, z, title=""):
    plt.style.use("default")
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, z, cmap=cm.viridis, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax.view_init(30, 30)
    ax.set_xlabel("West - East")
    ax.set_ylabel("North - South")
    ax.set_zlim(0, 3000)
    ax.set_title(title)
    plt.show()

def Train_test_val_terrain(x1_mesh, x2_mesh, terrain, intv):
    """Splits the data set into training, validation and testing sets"""
    train0 = x1_mesh[::intv, ::intv]
    a = train0.shape[0]
    train = np.zeros((3, a, a))
    train[0, :, :] = train0
    train[1, :, :] = x2_mesh[::intv, ::intv]
    train[2, :, :] = terrain[::intv, ::intv]

    n = int(intv / 2)
    val0 = x1_mesh[n::intv, n::intv]
    b = val0.shape[0]
    val = np.zeros((3, b, b))
    val[0, :, :] = val0
    val[1, :, :] = x2_mesh[n::intv, n::intv]
    val[2, :, :] = terrain[n::intv, n::intv]

    n = int(intv / 3)
    intv = int(intv/2)
    test0 = x1_mesh[n::intv, n::intv]
    c = test0.shape[0]
    test = np.zeros((3, c, c))
    test[0, :, :] = test0
    test[1, :, :] = x2_mesh[n::intv, n::intv]
    test[2, :, :] = terrain[n::intv, n::intv]

    return train, test, val

train, test, val = Train_test_val_terrain(x1_mesh, x2_mesh, terrain, 22)

if __name__ == "__main__":
    plt.style.use('ggplot')
    plot_terrain(x1_mesh, x2_mesh, terrain)
    plot_terrain(train[0], train[1], train[2])
