import numpy as np
import reader as reader


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def animate_3d(points, frame_rate):

    def update_graph(num):
        graph._offsets3d = (points[num, 0], points[num, 1], points[num, 2])
        title.set_text('Skeleton, frame={}'.format(num))

    fig = plt.figure()
    ax = Axes3D(fig)
    lim = np.min(points), np.max(points)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_zlim(*lim)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    title = ax.set_title('Skeleton')

    graph = ax.scatter(points[0, 0], points[0, 1], points[0, 2])
    ani = animation.FuncAnimation(fig, update_graph, len(points), interval=1000//frame_rate, blit=False)

    plt.show()


def animate_2d(points, frame_rate):

    def update_graph(num):
        graph.set_offsets(points[num].T)
        title.set_text('Skeleton, frame={}'.format(num))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    lim = np.min(points), np.max(points)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    title = ax.set_title('Skeleton')

    graph = ax.scatter(points[0, 0], points[0, 1])
    ani = animation.FuncAnimation(fig, update_graph, len(points), interval=1000//frame_rate, blit=False)

    plt.show()


def extract_3d(filename, reorder=(0, 1, 2)):
    readInst = reader.MyReader(filename, True)
    dt, points, limits = readInst.read()
    points = points[:, reorder]
    return points


def rotation_m(phi, theta):
    phi, theta = map(np.deg2rad, (phi, theta))
    xy = np.array([
        [np.cos(phi), -np.sin(phi), 0],
        [np.sin(phi),  np.cos(phi), 0],
        [0,            0,           1],
    ])
    xz = np.array([
        [np.cos(theta),  0, np.sin(theta)],
        [0,              1, 0],
        [-np.sin(theta), 0, np.cos(theta)],
    ])
    return np.matmul(xy, xz)


def extract_2d(filename, reorder, phi, theta):
    # compute in radians
    phi, theta = map(np.deg2rad, (phi, theta))

    # extract 3d points [num_frames, 3, num_points]
    points = extract_3d(filename, reorder)
    num_frames = len(points)

    def normalize(points):
        points -= np.mean(points, axis=-1, keepdims=True)
        points /= np.max(points.reshape(num_frames, -1), axis=-1).reshape(num_frames, 1, 1) - \
                  np.min(points.reshape(num_frames, -1), axis=-1).reshape(num_frames, 1, 1)
        return points

    def rotate_3d(phi, theta):
        yz = np.array([
            [1, 0,             0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ])
        xy_inv = np.array([
            [np.cos(phi), np.sin(phi), 0],
            [-np.sin(phi), np.cos(phi), 0],
            [0, 0, 1],
        ])
        return np.matmul(yz, xy_inv)

    points = normalize(points)
    points = points.transpose((1, 0, 2)).reshape(3, -1)
    points = np.matmul(rotate_3d(phi, theta), points)
    points = points.reshape(3, num_frames, -1).transpose(1, 0, 2)

    # project to XZ
    points[:, 1] -= (points[:, 1].min() - 1)  # d == 1
    points = points[:, [0, 2]] / points[:, [1]]
    points = normalize(points)

    return points


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("filename", type=str)
    parser.add_argument("--frame_rate", type=int, required=False, default=25)
    parser.add_argument("--view", type=str, required=False)
    args = parser.parse_args()

    # this reorder needs for BerkeleyMHAD dataset
    reorder = 0, 2, 1

    if args.view is None:
        points = extract_3d(args.filename, reorder)
        animate_3d(points, args.frame_rate)
    else:
        phi, theta = map(float, args.view.split(','))
        phi = phi % 360
        theta = theta % 90
        points = extract_2d(args.filename, reorder, phi, theta)
        animate_2d(points, args.frame_rate)
