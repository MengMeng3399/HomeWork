import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

m, n, num = 5, 4, 160
all_high = 0.3
interval = 0.008
length = (1 - (m + 1) * interval) / m
half_length = length / 2
high = (all_high - (n + 1) * interval) / n
half_high = high / 2
dt = 2e-4

box = ti.Vector.field(2, float, shape=(m, n))
box_center = ti.Vector.field(2, float, shape=(m, n))
box_destroy = ti.field(int, shape=(m, n))
circle_v = ti.Vector.field(2, float, shape=())
circle_x = ti.Vector.field(2, float, shape=())
board_center = ti.Vector.field(2, float, shape=())
game_over = ti.field(int, ())
score = ti.field(int, ())
board_half_length = 0.08
circle_radius = 0.01
init_circle_high = circle_radius + 0.12

n_grid = 128
dx = 1 / n_grid
inv_dx = 1 / dx
p_rho = 1
p_vol = (dx * 0.5) ** 2
p_mass = p_vol * p_rho
E = 100
gravity = 4.8
bound = 3
# E, nu = 0.1e4, 0.2  # Young's modulus and Poisson's ratio
# mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame parameters

C = ti.Matrix.field(2, 2, float, (m, n, num))
J = ti.field(float, (m, n, num))
p_x = ti.Vector.field(2, float, shape=(m, n, num))
p_v = ti.Vector.field(2, float, shape=(m, n, num))
p_destroy = ti.field(int, shape=(m, n))
F = ti.Matrix.field(2, 2, dtype=float, shape=(m, n, num))  # deformation gradient
Jp = ti.field(dtype=float, shape=(m, n, num))  # plastic deformation

grid_v = ti.Vector.field(2, float, (n_grid, n_grid))
grid_m = ti.field(float, (n_grid, n_grid))

_normal = np.array(
    [
        [0.0, 1.0],  # ↑
        [1.0, 0.0],  # →
        [0.0, -1.0],  # ↓
        [-1.0, 0.0]  # ←
    ])
normal = ti.Vector.field(2, float, _normal.shape[:1])  # edge table (constant)
normal.from_numpy(_normal)

coefficient_v = 28


@ti.func
def ti_none(v):
    return v[None]


@ti.func
def ti_vector(u, v):
    return ti.Vector([u, v])


@ti.func
def clamp(difference, x, y):
    return max(x, min(y, difference))


@ti.func
def check_collision(i, j):
    aabb_half_offset = ti_vector(length / 2, high / 2)
    # 获取长方形的中心。
    aabb_center = box[i, j] + aabb_half_offset
    # 球心到中心的距离
    difference = ti_none(circle_x) - aabb_center
    # clamp
    clamped = clamp(difference, -aabb_half_offset, aabb_half_offset)
    # 求球心到长方形的最近点所在位置
    closest = aabb_center + clamped
    # 求球心到长方形边界的距离
    difference = closest - ti_none(circle_x)
    return [difference, difference.norm() < circle_radius]


@ti.func
def collision_edge(target):
    max_dot = 0.0
    match = -1
    for i in range(4):
        dot_product = target.normalized().dot(normal[i])
        if dot_product > max_dot:
            max_dot = dot_product
            match = i
    return match


@ti.func
def game_pass():
    temp = 1
    for I in ti.grouped(box_destroy):
        if box_destroy[I] == 0:
            temp = 0
    return temp


@ti.kernel
def move() -> int:
    circle_v[None] = ti_none(circle_v).normalized() * coefficient_v
    circle_x[None] += ti_none(circle_v) * dt
    new_x = ti_none(circle_x) + ti_none(circle_v) * dt
    collided = 0
    # 检测与边界的碰撞
    if new_x.x - circle_radius < 0 or new_x.x + circle_radius > 1:
        circle_v[None].x = -ti_none(circle_v).x
        collided = 1
    if new_x.y + circle_radius > 1:
        circle_v[None].y = -ti_none(circle_v).y
        collided = 1
    if new_x.y - circle_radius < 0.05:
        game_over[None] = 1

    # 检测与长方形的碰撞
    for i, j in box:
        if box_destroy[i, j] != 1:
            if check_collision(i, j)[1]:
                ti.atomic_add(score[None], 1)
                collided = 1
                edge = collision_edge(check_collision(i, j)[0])
                if edge == 0 or edge == 2:
                    circle_v[None].y = -ti_none(circle_v).y
                if edge == 1 or edge == 3:
                    circle_v[None].x = -ti_none(circle_v).x
                box_destroy[i, j] = 1

    # 检测球与挡板碰撞
    if abs(circle_x[None].x - board_center[None].x) <= board_half_length and new_x.y <= init_circle_high:
        percentage = abs(circle_x[None].x - board_center[None].x) / board_half_length
        circle_v[None].x = clamp(circle_v[None].x * percentage * 4.02, -20, 20)
        circle_v[None].y = abs(circle_v[None].y)
        collided = 1
    if collided == 0:
        circle_x[None] = new_x
    return game_pass()


@ti.kernel
def init():
    score[None] = 0
    circle_x[None] = ti_vector(0.5, init_circle_high + 0.005)
    board_center[None] = ti_vector(0.5, init_circle_high - circle_radius)
    for i, j in box:
        position_x = interval * (i + 1) + i * length
        position_y = (1 - (interval * (j + 1) + (j + 1) * high))
        box[i, j] = ti_vector(position_x, position_y)
        p_destroy[i, j] = 0
        box_destroy[i, j] = 0
    for i, j, k in p_x:
        p_x[i, j, k] = [ti.random() * length * 0.95 + (interval * (i + 1) + i * length),
                        ti.random() * high * 0.95 + (1 - (interval * (j + 1) + (j + 1) * high))]
        p_v[[i, j, k]] = [0, 0]
        J[[i, j, k]] = 1


def game_state(state):
    return state % 3


@ti.kernel
def init_v(x: ti.f32, y: ti.f32):
    circle_v[None] = (ti_vector(x, y) - circle_x[None]).normalized() * coefficient_v


delete_time = 50000


@ti.kernel
def P2G():
    for i, j in box:
        if box_destroy[i, j] == 1 and p_destroy[i, j] < delete_time:
            p_destroy[i, j] += 1
    for i, j in grid_m:
        grid_v[i, j] = [0, 0]
        grid_m[i, j] = 0
    for i, j, k in p_x:
        if box_destroy[i, j] == 1 and p_destroy[i, j] < delete_time:
            Xp = p_x[i, j, k] / dx
            p_v[i, j, k].y -= dt * gravity
            base = int(Xp - 0.5)
            fx = Xp - base
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            stress = -dt * 4 * E * p_vol * (J[i, j, k] - 1) / dx ** 2
            affine = ti.Matrix([[stress, 0], [0, stress]]) + p_mass * C[i, j, k]
            for _i, _j in ti.static(ti.ndrange(3, 3)):
                offset = ti.Vector([_i, _j])
                dpos = (offset - fx) * dx
                weight = w[_i].x * w[_j].y
                grid_v[base + offset] += weight * (p_mass * p_v[i, j, k] + affine @ dpos)
                grid_m[base + offset] += weight * p_mass


@ti.kernel
def grid_operator():
    for i, j in grid_m:
        if grid_m[i, j] > 0:
            grid_v[i, j] /= grid_m[i, j]
        if i < bound and grid_v[i, j].x < 0:
            grid_v[i, j].x = 0
        if i > n_grid - bound and grid_v[i, j].x > 0:
            grid_v[i, j].x = 0
        if j < bound and grid_v[i, j].y < 0:
            grid_v[i, j].y = 0
        if j > n_grid - bound and grid_v[i, j].y > 0:
            grid_v[i, j].y = 0


@ti.kernel
def G2P():
    for i, j, k in p_x:
        if box_destroy[i, j] == 1:
            Xp = p_x[i, j, k] / dx
            base = int(Xp - 0.5)
            fx = Xp - base
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            new_v = ti.Vector.zero(float, 2)
            new_C = ti.Matrix.zero(float, 2, 2)
            for _i, _j in ti.static(ti.ndrange(3, 3)):
                offset = ti.Vector([_i, _j])
                dpos = (offset - fx) * dx
                weight = w[_i].x * w[_j].y
                g_v = grid_v[base + offset]
                new_v += weight * g_v
                new_C += 4 * weight * g_v.outer_product(dpos) / dx ** 2
            p_v[i, j, k] = new_v
            p_x[i, j, k] += dt * p_v[i, j, k]
            J[i, j, k] *= 1 + dt * new_C.trace()
            C[i, j, k] = new_C


init()
box_origin = box.to_numpy()
is_run = 0

gui = ti.GUI("Breakout Game", res=(600, 500), background_color=0x000000)
state = ''
_pass = 0
while gui.running:
    mouse_x, mouse_y = gui.get_cursor_pos()
    for e in gui.get_events(ti.GUI.PRESS):
        state = e.key
        if state == 'r':
            _pass = 0
            init()
            game_over[None] = 0
            state = 'q'
    if state == 'q':
        circle_x[None].x = mouse_x
        board_center[None].x = mouse_x
    if state == 'w':
        gui.line(circle_x.to_numpy(), gui.get_cursor_pos(), radius=2, color=0x068587)
    if state == 'e':
        init_v(mouse_x, mouse_y)
        is_run = 1
        board_center[None].x = mouse_x
        state = ''
    if is_run:
        board_center[None].x = mouse_x
        _pass = move()
    if _pass:
        is_run = 0
        state = ''
        gui.text(content='Congratulations on completing the game!',
                 pos=(0.15, 0.65),
                 font_size=25,
                 color=0xffffff)
        gui.text(content='Click the \'R\' to start again!',
                 pos=(0.25, 0.45),
                 font_size=30,
                 color=0xffffff)
    if game_over[None] == 1:
        is_run = 0
        state = ''
        gui.text(content='Game Over!',
                 pos=(0.32, 0.65),
                 font_size=45,
                 color=0xffffff)
        gui.text(content='Click the \'R\' to start again!',
                 pos=(0.22, 0.45),
                 font_size=30,
                 color=0xffffff)
    if is_run == 1:
        for s in range(50):
            P2G()
            grid_operator()
            G2P()
    board_center_np = board_center.to_numpy()
    board_left = [board_center_np[0] - board_half_length, board_center_np[1]]
    board_right = [board_center_np[0] + board_half_length, board_center_np[1]]
    circle_pos = circle_x.to_numpy()
    gui.line(board_left, board_right, radius=3.5, color=0xFFFFFF)
    pos_x = p_x.to_numpy()
    gui.text(content=f'score is {score[None]}',
             pos=(0.03, 0.2),
             font_size=20,
             color=0xffffff)
    for i, j in ti.ndrange(m, n):
        if p_destroy[i, j] < delete_time:
            gui.circles(pos_x[i, j], radius=1.5, color=0x87CEEB)
        origin_x = box_origin[i, j][0]
        origin_y = box_origin[i, j][1]
        if box_destroy[i, j] == 0:
            gui.rect([origin_x, origin_y + high], [origin_x + length, origin_y], radius=1.5, color=0xFFFAFA)
    gui.circle(circle_pos, radius=5, color=0xDC143C)
    gui.show()  # Change to gui.show(f'{frame:06d}.png') to write images to disk
