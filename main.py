"""Use taichi to visualize 2D wave equation."""
import numpy as np
import taichi as ti
ti.init(arch=ti.gpu)

N = 500
N0 = 10
h = 0.03

U = ti.field(dtype=float, shape=(N, N))   # 2D function
Ut = ti.field(dtype=float, shape=(N, N))  # derivative of U on time, i.e. speed of U
LU = ti.field(dtype=float, shape=(N, N))  # laplace of U


def init():
    """Initial condition of speed."""
    for _ in range(N0):
        i, j = np.random.randint(0, N, 2)
        Ut[i, j] = np.random.uniform()


@ti.kernel
def laplace():
    """Laplace of U. Refer
    - https://github.com/taichi-dev/taichi/blob/ceb1edc73bfdf08103b074b953a4d0798486a6b5/examples/algorithm/laplace.py#L12-L19
    - https://github.com/taichi-dev/taichi/blob/ceb1edc73bfdf08103b074b953a4d0798486a6b5/examples/algorithm/mgpcg.py#L58-L64
    """
    for i, j in ti.ndrange(N, N):
        # LU[i, j] = -4 * U[i, j] + U[i - 1, j] + U[i + 1, j] + U[i, j - 1] + U[i, j + 1]
        LU[i, j] = \
            -6 * U[i, j] + \
            0.5 * (U[i - 1, j - 1] + U[i + 1, j - 1] + U[i - 1, j + 1] + U[i + 1, j + 1]) + \
            (U[i - 1, j] + U[i + 1, j] + U[i, j - 1] + U[i, j + 1])


@ti.kernel
def update():
    """Iterative on U and Ut."""
    for i, j in ti.ndrange(N, N):
        U[i, j] = U[i, j] + Ut[i, j] * h
        Ut[i, j] = Ut[i, j] + LU[i, j] * h


def post_process(rng=[-0.1, 0.1]):
    """Post process of image. Refer
    - https://docs.taichi.graphics/docs/lang/articles/basic/external
    """
    u = U.to_numpy()
    u = (u - rng[0]) / (rng[1] - rng[0]) * 255
    u = np.clip(u, 0, 255).astype(np.uint8)
    return u


def show():
    with ti.GUI("2D wave equation", res=(N, N)) as gui:
        init()
        while gui.running:
            laplace()
            update()
            u = post_process()
            gui.set_image(u)
            gui.show()


def save():
    result_dir = "./data"
    video_manager = ti.VideoManager(output_dir=result_dir, framerate=24, automatic_build=False)

    init()
    for i in range(1000):
        laplace()
        update()
        u = post_process()
        video_manager.write_frame(u)
        print(f'\rFrame {i + 1}/1000 is recorded', end='')

    print()
    print('Exporting .mp4 and .gif videos...')
    video_manager.make_video(gif=True, mp4=True)
    print(f'MP4 video is saved to {video_manager.get_output_filename(".mp4")}')
    print(f'GIF video is saved to {video_manager.get_output_filename(".gif")}')


if __name__ == "__main__":
    show()
    # save()
