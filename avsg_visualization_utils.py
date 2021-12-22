import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


######################################################################


def plot_poly_elems(ax, poly, facecolor='0.4', alpha=0.3, edgecolor='black', label='', is_closed=False, linewidth=1):
    first_plt = True
    for elem in poly:
        x = elem[0, :, 0].detach().cpu()
        y = elem[0, :, 1].detach().cpu()
        if first_plt:
            first_plt = False
        else:
            label = None
        if is_closed:
            ax.fill(x, y, facecolor=facecolor, alpha=alpha, edgecolor=edgecolor, linewidth=linewidth, label=label)
        else:
            ax.plot(x, y, alpha=alpha, color=edgecolor, linewidth=linewidth, label=label)


##############################################################################################


def plot_lanes(ax, left_lanes, right_lanes, facecolor='0.4', alpha=0.3, edgecolor='black', label='', linewidth=1):
    # assert len(left_lanes) == len(right_lanes)
    n_elems = min(len(left_lanes), len(right_lanes))
    first_plt = True
    for i in range(n_elems):
        x_left = left_lanes[i][0, :, 0]
        y_left = left_lanes[i][0, :, 1]
        x_right = right_lanes[i][0, :, 0]
        y_right = right_lanes[i][0, :, 1]
        x = torch.cat((x_left, torch.flip(x_right, [0]))).detach().cpu()
        y = torch.cat((y_left, torch.flip(y_right, [0]))).detach().cpu()
        if first_plt:
            first_plt = False
        else:
            label = None
        ax.fill(x, y, facecolor=facecolor, alpha=alpha, edgecolor=edgecolor, linewidth=linewidth, label=label)


##############################################################################################


def plot_rectangles(ax, centroids, extents, yaws, label, facecolor, alpha=0.7, edgecolor='black'):
    n_elems = len(centroids)
    first_plt = True
    for i in range(n_elems):
        if first_plt:
            first_plt = False
        else:
            label = None
        height = extents[i][0]
        width = extents[i][1]
        angle = yaws[i]
        angle_deg = float(np.degrees(angle))
        xy = centroids[i] \
             - 0.5 * height * np.array([np.cos(angle), np.sin(angle)]) \
             - 0.5 * width * np.array([-np.sin(angle), np.cos(angle)])
        rect = Rectangle(xy, height, width, angle_deg, facecolor=facecolor, alpha=alpha,
                         edgecolor=edgecolor, linewidth=1, label=label)

        ax.add_patch(rect)


##############################################################################################


def visualize_scene_feat(agents_feat, map_feat):
    centroids = np.stack([af['centroid'] for af in agents_feat])
    yaws = np.stack([af['yaw'] for af in agents_feat])
    speed = np.stack([af['speed'] for af in agents_feat])
    # print('agents types: ', [af['agent_label_id'] for af in agents_feat])
    X = centroids[:, 0]
    Y = centroids[:, 1]
    U = speed * np.cos(yaws)
    V = speed * np.sin(yaws)

    fig, ax = plt.subplots()
    plot_lanes(ax, map_feat['lanes_left'], map_feat['lanes_right'], facecolor='grey', alpha=0.3, edgecolor='black',
               label='Lanes')
    plot_poly_elems(ax, map_feat['lanes_mid'], facecolor='lime', alpha=0.4, edgecolor='lime', label='Lanes mid',
                    is_closed=False, linewidth=1)
    plot_poly_elems(ax, map_feat['crosswalks'], facecolor='orange', alpha=0.3, edgecolor='orange', label='Crosswalks',
                    is_closed=True)

    n_agents = len(agents_feat)
    if n_agents > 0:
        extents = [af['extent'] for af in agents_feat]
        plot_rectangles(ax, centroids[1:], extents[1:], yaws[1:], label='non-ego', facecolor='saddlebrown')
        plot_rectangles(ax, [centroids[0]], [extents[0]], [yaws[0]], label='ego', facecolor='red')

        valid = speed > 1e-4
        ax.quiver(X[valid], Y[valid], U[valid], V[valid], units='xy', color='black', width=0.5)

    ax.grid()
    plt.legend()
    # plt.show()

    canvas = plt.gca().figure.canvas
    canvas.draw()
    data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    image = data.reshape(canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return image
##############################################################################################
