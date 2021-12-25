# 游戏设计 1.从右边打出小球
#       2.当箭射到小球，或者出界，箭回到原点
#       3.当箭射到小球，小球消失


import taichi as ti
import numpy as np
import time
import math

ti.init(arch=ti.cpu)
# 屏幕大小

dim = 2
res = 800
dx = 1.0
inv_dx = 1.0 / dx
half_inv_dx = 0.5 * inv_dx
dt = 0.009
p_jacobi_iters = 30
# 注入能量需要用到的参数
f_strength = 10000.0
# 密度消散
dye_decay = 0.99
# 箭需要的参数
num_arrowLine = 3
arrow_position = ti.Vector.field(dim, float)
ti.root.dense(ti.i, num_arrowLine).place(arrow_position)
line_radius = 3.0
speed_arrow = 1500
line_color = 0x00BFFF
fraction = ti.field(int)
ti.root.place(fraction)
# 设置一个箭头的方向，其实就是速度的方向，但是最后碰到靶子后，会将速度设为0，但是箭应该留在靶子上
arrow_dir = ti.Vector.field(dim, float)
ti.root.place(arrow_dir)

# 鼠标点击需要的参数
mouse_data = ti.Vector.field(dim, float)
ti.root.place(mouse_data)
# 为了得到鼠标点击之后，箭的速度方向，需要一个固定点
const_data = ti.Vector.field(dim, float)
ti.root.place(const_data)

paused = ti.field(int)
ti.root.dense(ti.i, 2).place(paused)

# 小人需要的属性，小人的位置不变
man_position = ti.Vector.field(dim, float)
ti.root.place(man_position)

man_radius = 15.0
man_color = 0xBC8F8F
num_line = 4
line_radius = 2.0
man_line_start = ti.Vector.field(dim, float)
man_line_end = ti.Vector.field(dim, float)
ti.root.dense(ti.i, num_line).place(man_line_start, man_line_end)

# 从右边射出小球的设置,先设置10个球做测试
num_ball = 10
ball_position = ti.Vector.field(dim, float)
ball_velocity = ti.Vector.field(dim, float)
ti.root.dense(ti.i, num_ball).place(ball_position, ball_velocity)
ball_radius = ti.field(float)
ti.root.dense(ti.i, num_ball).place(ball_radius)

game_over = ti.field(int, ())
# 设置一个粒子，以该粒子的运动更新速度场，密度场
particle_position = ti.Vector.field(dim, float)
particle_velocity = ti.Vector.field(dim, float)
ti.root.place(particle_position)
ti.root.place(particle_velocity)

_velocities = ti.Vector.field(2, dtype=float, shape=(res, res))
_new_velocities = ti.Vector.field(2, dtype=float, shape=(res, res))
# 速度的散度
velocity_divs = ti.field(dtype=float, shape=(res, res))
_pressures = ti.field(dtype=float, shape=(res, res))
_new_pressures = ti.field(dtype=float, shape=(res, res))
color_buffer = ti.Vector.field(3, dtype=float, shape=(res, res))
_dye_buffer = ti.Vector.field(3, dtype=float, shape=(res, res))
_new_dye_buffer = ti.Vector.field(3, dtype=float, shape=(res, res))


class TexPair:
    def __init__(self, cur, nxt):
        self.cur = cur
        self.nxt = nxt

    def swap(self):
        self.cur, self.nxt = self.nxt, self.cur


velocities_pair = TexPair(_velocities, _new_velocities)
pressures_pair = TexPair(_pressures, _new_pressures)
dyes_pair = TexPair(_dye_buffer, _new_dye_buffer)


@ti.func
def sample(qf, u, v):
    i, j = int(u), int(v)
    i = max(0, min(res - 1, i))
    j = max(0, min(res - 1, j))
    return qf[i, j]


# 线性插值
@ti.func
def lerp(vl, vr, frac):
    # frac: [0.0, 1.0]
    return vl + frac * (vr - vl)


# 双线性插值
@ti.func
def bilerp(vf, u, v):
    s, t = u - 0.5, v - 0.5
    # floor
    iu, iv = int(s), int(t)
    # fract
    fu, fv = s - iu, t - iv
    a = sample(vf, iu + 0.5, iv + 0.5)
    b = sample(vf, iu + 1.5, iv + 0.5)
    c = sample(vf, iu + 0.5, iv + 1.5)
    d = sample(vf, iu + 1.5, iv + 1.5)
    return lerp(lerp(a, b, fu), lerp(c, d, fu), fv)


@ti.kernel
def advect(vf: ti.template(), qf: ti.template(), new_qf: ti.template()):
    for i, j in vf:
        # 向后追踪一个dt
        coord = ti.Vector([i, j]) + 0.5 - dt * vf[i, j]
        new_qf[i, j] = bilerp(qf, coord[0], coord[1])


# ---------------------------------------------------------------------------------------
force_radius = res / 3.0
inv_force_radius = 1.0 / force_radius
inv_dye_denom = 4.0 / (res / 15.0) ** 2
f_strength_dt = f_strength * dt


@ti.kernel
def apply_impulse(vf: ti.template(), dyef: ti.template()):
    for i, j in vf:
        # omx，omy当前粒子的位置
        omx, omy = particle_position[None][0], particle_position[None][1]
        # mdir 当前粒子移动的方向
        mdir = particle_velocity[None]
        dx, dy = (i + 0.5 - omx), (j + 0.5 - omy)
        # 中心网格到当前粒子位置的距离
        d2 = (dx * dx + dy * dy) * 500.0
        # dv = F * dt
        # 以下是注入能量的步骤：
        # 注入速度：跟网格信息点与当前粒子的距离有关。
        factor = ti.exp(-d2 * inv_force_radius)
        momentum = mdir * f_strength_dt * factor * 2
        v = vf[i, j]
        vf[i, j] = v + momentum
        # add dye
        dc = dyef[i, j]
        if mdir.norm() > 0.5:
            dc += ti.exp(-d2 * inv_dye_denom) * ti.Vector(
                [1.0, 1.0, 1.0]) * 20.0
        # 密度消散。
        dc *= dye_decay
        dyef[i, j] = dc


# 求速度的散度，边界处理为固体
@ti.kernel
def divergence(vf: ti.template()):
    for i, j in vf:
        vl = sample(vf, i - 1, j)[0]
        vr = sample(vf, i + 1, j)[0]
        vb = sample(vf, i, j - 1)[1]
        vt = sample(vf, i, j + 1)[1]
        vc = sample(vf, i, j)
        if i == 0:
            vl = -vc[0]
        if i == res - 1:
            vr = -vc[0]
        if j == 0:
            vb = -vc[1]
        if j == res - 1:
            vt = -vc[1]

        velocity_divs[i, j] = (vr - vl + vt - vb) * half_inv_dx


p_alpha = -dx * dx


@ti.kernel
def pressure_jacobi(pf: ti.template(), new_pf: ti.template()):
    for i, j in pf:
        # 把边界当固体处理了，即与边界的压强梯度为0
        pl = sample(pf, i - 1, j)
        pr = sample(pf, i + 1, j)
        pb = sample(pf, i, j - 1)
        pt = sample(pf, i, j + 1)
        div = velocity_divs[i, j]
        # 求解压力。根据速度的散度
        new_pf[i, j] = (pl + pr + pb + pt + p_alpha * div) * 0.25


@ti.kernel
def subtract_gradient(vf: ti.template(), pf: ti.template()):
    for i, j in vf:
        pl = sample(pf, i - 1, j)
        pr = sample(pf, i + 1, j)
        pb = sample(pf, i, j - 1)
        pt = sample(pf, i, j + 1)
        v = sample(vf, i, j)
        # 根据压力场投影速度
        v = v - half_inv_dx * ti.Vector([pr - pl, pt - pb])
        vf[i, j] = v


@ti.kernel
def fill_color_v3(vf: ti.template()):
    for i, j in vf:
        v = vf[i, j]
        color_buffer[i, j] = ti.Vector([abs(v[0]), abs(v[1]), abs(v[2])])


def step():
    advect(velocities_pair.cur, velocities_pair.cur, velocities_pair.nxt)
    advect(velocities_pair.cur, dyes_pair.cur, dyes_pair.nxt)
    velocities_pair.swap()
    dyes_pair.swap()

    apply_impulse(velocities_pair.cur, dyes_pair.cur)
    divergence(velocities_pair.cur)

    for _ in range(p_jacobi_iters):
        pressure_jacobi(pressures_pair.cur, pressures_pair.nxt)
        pressures_pair.swap()

    subtract_gradient(velocities_pair.cur, pressures_pair.cur)
    fill_color_v3(dyes_pair.cur)


def vec2_npf32(m):
    return np.array([m[0], m[1]], dtype=np.float32)


# ----------------箭的操作-------------------------

@ti.kernel
def init_state():
    # 固定点的位置
    const_data[None] = ti.Vector([70, 365])
    particle_position[None] = const_data[None] + ti.Vector([50.0, 0.0])
    # #初始化箭的方向
    arrow_dir[None] = ti.Vector([1.0, 0.0])
    # #初始化线
    cal_arrowLine()
    # 鼠标初始数据,为了确保箭的初始速度是水平向右的，当然也可以是别的。
    mouse_data[None] = ti.Vector([50, 275])
    # 小人的位置
    man_position[None] = ti.Vector([50.0, 400.0])
    man_line_start[0] = man_position[None] + ti.Vector([0.0, -man_radius])
    man_line_start[1] = man_position[None] + ti.Vector([0.0, -man_radius])
    man_line_end[0] = man_line_start[0] + ti.Vector([-20.0, -20.0])
    man_line_end[1] = man_line_start[1] + ti.Vector([20.0, -20.0])
    man_line_start[2] = man_line_start[0] + ti.Vector([-10.0, -10.0])
    man_line_start[3] = man_line_start[1] + ti.Vector([10.0, -10.0])
    man_line_end[2] = man_line_start[2] + ti.Vector([0.0, -50.0])
    man_line_end[3] = man_line_start[3] + ti.Vector([0.0, -50.0])

    # 右边打出的小球的设置
    min_x = 1000
    max_x = 1300
    min_vel = 100
    max_vel = 400
    min_radius = 10
    max_radius = 20
    for i in range(num_ball):
        ball_position[i][0] = (max_x - min_x) * ti.random() + min_x
        ball_position[i][1] = (res - 100 - 100) * ti.random() + 100
        # 设置小球的速度
        ball_velocity[i][0] = -((max_vel - min_vel) * ti.random() + min_vel)
        # 设置小球的半径
        ball_radius[i] = (max_radius - min_radius) * ti.random() + min_radius
    paused[0] = True
    paused[1] = True


@ti.func
def cal_arrowLine():
    # 计算线的位置
    dir = arrow_dir[None]
    arrow_position[0] = particle_position[None] - dir * 50
    # 将-dir旋转
    x = -dir[0]
    y = -dir[1]
    x1 = math.cos(50) * x - math.sin(50) * y
    y1 = math.sin(50) * x + math.cos(50) * y
    dir2 = (ti.Vector([x1, y1])).normalized()
    arrow_position[1] = particle_position[None] + dir2 * 20

    x1 = math.cos(-50) * x - math.sin(-50) * y
    y1 = math.sin(-50) * x + math.cos(-50) * y
    dir3 = (ti.Vector([x1, y1])).normalized()
    arrow_position[2] = particle_position[None] + dir3 * 20


@ti.func
def confine_ball_position(p, v, r):
    dis = (p - particle_position[None]).norm()
    if dis <= r + 5.0:
        # r=0.0
        fraction[None] = fraction[None] + 10
        min_x = 1000
        max_x = 1300
        p[0] = (max_x - min_x) * ti.random() + min_x
        p[1] = (res - 100 - 100) * ti.random() + 100
    if p[0] < const_data[None][0]:
        paused[0] = True
        paused[1] = True
        game_over[None] = 1

    return r, p


@ti.kernel
def move_ball():
    # 更新小球的位置
    for i in range(num_ball):
        ball_position[i] += ball_velocity[i] * dt
        ball_radius[i], ball_position[i] = confine_ball_position(ball_position[i], ball_velocity[i], ball_radius[i])


@ti.kernel
def move_particle():
    particle_position[None] += particle_velocity[None] * dt
    # 对粒子进行限制
    # particle_position[None],particle_velocity[None]=confine_position_to_boundary(particle_position[None],particle_velocity[None])
    cal_arrowLine()


# -----------以上为箭的操作--------------------------------------------------

def reset():
    velocities_pair.cur.fill(ti.Vector([0, 0]))
    pressures_pair.cur.fill(0.0)
    dyes_pair.cur.fill(ti.Vector([0, 0, 0]))
    color_buffer.fill(ti.Vector([0, 0, 0]))
    particle_position[None] = ti.Vector([90.0, 275])
    init_state()
    fraction[None] = 0


def render(gui):
    # 画一条线，表示小球超过这个线游戏结束
    x = const_data[None][0]
    gui.line((x / res, 0.0), (x / res, 1.0), radius=0.6, color=0x33FF99)
    man_pos = man_position.to_numpy() / res
    man_line1 = man_line_start.to_numpy()
    man_line2 = man_line_end.to_numpy()
    for j in range(dim):
        man_line1[:, j] /= res
        man_line2[:, j] /= res
    # 头部
    gui.circle(man_pos, radius=man_radius, color=man_color)
    # 身子
    gui.lines(man_line1, man_line2, radius=line_radius, color=man_color)
    # 画出箭，三条线
    star_point = particle_position.to_numpy() / res
    arr_line = arrow_position.to_numpy()

    for i in range(num_arrowLine):
        arr_line[i] /= res
        gui.line(star_point, arr_line[i], radius=line_radius, color=line_color)

    # 画出小球
    pos_np = ball_position.to_numpy()
    pos_radius = ball_radius.to_numpy()
    for i in range(num_ball):
        pos_np[i] /= res
    gui.circles(pos_np, radius=pos_radius, color=0xCC00FF)


def print_inf(gui):
    gui.text("(S) Shoot an arrow", pos=(0.60, 0.95), font_size=25, color=0x98FB98)
    # gui.text('(O) One more arrow ', pos=(0.65, 0.90), font_size=20,color=0x98FB98)
    gui.text('(R) Restart', pos=(0.60, 0.90), font_size=25, color=0x98FB98)
    gui.text(f' score:{fraction}', pos=(0.05, 0.90), font_size=30, color=0xFF4500)
    if game_over[None] == 1:
        gui.text(content='Game Over!',
                 pos=(0.32, 0.65),
                 font_size=45,
                 color=0xffffff)
        gui.text(content='Click the \'R\' to start again!',
                 pos=(0.25, 0.45),
                 font_size=30,
                 color=0xffffff)


@ti.kernel
def arrow_change():
    the_start = const_data[None]
    the_end = mouse_data[None]
    dir = (the_end - the_start).normalized()
    particle_velocity[None] = dir * speed_arrow
    particle_position[None] = the_start + dir * 50
    arrow_dir[None] = (particle_velocity[None]).normalized()
    cal_arrowLine()


def mouse_op(gui):
    if gui.is_pressed(ti.GUI.LMB):
        mouse_data[None] = ti.Vector([gui.get_cursor_pos()[0], gui.get_cursor_pos()[1]]) * res
        if mouse_data[None][0] < const_data[None][0] + 10.0:
            mouse_data[None][0] = const_data[None][0] + 10.0
        arrow_change()


def main():
    gui = ti.GUI('Game', (res, res))
    init_state()
    while True:
        if gui.get_event(ti.GUI.PRESS):
            e = gui.event
            if e.key == ti.GUI.ESCAPE:
                break
            elif e.key == 'a':
                paused[1] = False
            elif e.key == 'r':
                paused[0] = True
                reset()
                move_particle()
                game_over[None] = 0
            # 按下S，放箭
            elif e.key == 's' and paused[0]:
                paused[0] = False
        if not paused[0]:
            move_particle()
            step()
        if not paused[1]:
            move_ball()
        mouse_op(gui)
        img = color_buffer.to_numpy()
        gui.set_image(img)
        render(gui)
        print_inf(gui)
        gui.show()


if __name__ == '__main__':
    main()
