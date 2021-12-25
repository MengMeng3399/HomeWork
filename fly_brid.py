# 问题：1.循环停止条件的密度限制
#     2.iisph是无条件稳定的吗？ 为什么去掉边界碰撞的速度处理会炸


import taichi as ti
import numpy as np
import math
import random
import threading
import time
import datetime

ti.init(arch=ti.cpu)

t = time.time()

delta_t = t

# 以下为flybirds实现

wallPositions = ti.Vector.field(n=2, dtype=ti.f32, shape=(10000))
wallVelocity = ti.field(float, shape=())

wallPoint = ti.field(int, shape=())
upDownDis = ti.field(float, shape=(10000))
leftToRight = ti.field(float, shape=())
midDis = ti.field(float, shape=())
otherP = ti.Vector.field(n=2, dtype=ti.f32, shape=())

birdPosition = ti.Vector.field(n=2, dtype=ti.f32, shape=())

# 游戏状态
gameState = ti.field(int, shape=())

# 游戏得分
fraction = ti.field(float, shape=())

# 游戏得分系数
fraction_k = ti.field(int, shape=())

# 标识游戏是否结束
isEnd = ti.field(int, shape=())

# 检测障碍是否越界
isSlipted = ti.field(int, shape=())
seed = ti.field(int, shape=())
# 保存越界障碍的索引
wallIndex = ti.field(int, shape=())


@ti.kernel
def initGame():
    seed[None] = 0
    isEnd[None] = 0
    gameState[None] = 0
    fraction[None] = 0.0
    fraction_k[None] = 1
    wallPoint[None] = 6.0
    leftToRight[None] = 80 / (wallPoint[None] - 1)
    leftToRight[None] -= 2

    midDis[None] = 18
    wallIndex[None] = -1
    otherP[None].x = 80
    wallVelocity[None] = 0.1
    for i in range((wallPoint[None] // 2)):
        wallPositions[i].x = i * leftToRight[None] * 2 + 2
        wallPositions[i].y = 0
        temp = np.random.random_sample()
        k = 0
        if temp < 0.5:
            k = -1
        else:
            k = 1

        upDownDis[i] = 15 * temp * 2

        # 初始化小鸟的位置
        birdPosition[None].x = 30
        birdPosition[None].y = upDownDis[0] + midDis[None] / 1.2


@ti.func
def emitParticles():
    temp = ti.random()
    if temp > 0.222 and temp < 0.225:
        num = ti.random() * (num_particles * 0.8)
        a = num - 5
        b = num + 5
        if (a < 0):
            a = 0
        if (b > num_particles - 1):
            b = num_particles - 1

        for i in range(a, b):
            forces[i].y += 100


@ti.kernel
def moveWall():
    for i in range(int(wallPoint[None] / 2)):
        wallPositions[i].x -= wallVelocity[None]

        if wallPositions[i].x <= 0:
            wallPositions[i].x = 80 - leftToRight[None]

            # 保存当前障碍的索引，于是单独绘制这部分，直到障碍合并
            wallIndex[None] = i
        random.seed(seed[None])
    if wallIndex[None] != -1:
        if otherP[None].x > 80 - leftToRight[None]:
            otherP[None].x -= wallVelocity[None]
        else:
            wallPositions[wallIndex[None]].x = otherP[None].x
            otherP[None].x = 80

            m = (seed[None])

            # 在这里实现随机障碍生成

            t = ti.random()
            upDownDis[wallIndex[None]] = 5 + 25 * t

            wallIndex[None] = -1


@ti.kernel
def updateBirdPosition():
    birdPosition[None].y -= 0.5
    if birdPosition[None].y < 0:
        birdPosition[None].y = 0


@ti.kernel
def keyUp():
    if gameState[None] == 0:
        birdPosition[None].y += 1.0


@ti.kernel
def keySpeedUp():
    wallVelocity[None] += 0.01


@ti.kernel
def keySolwDown():
    wallVelocity[None] -= 0.01
    if wallVelocity[None] <= 0.1:
        wallVelocity[None] = 0.1


@ti.kernel
def keyBirdsMoveToRight():
    birdPosition[None].x += 0.4


@ti.kernel
def keyBirdsMoveToLeft():
    birdPosition[None].x -= 0.4


@ti.kernel
def flyBirdsCollision():
    for i in range(int(wallPoint[None] / 2)):
        lowP1_x = wallPositions[i].x
        lowP1_y = wallPositions[i].y

        highP1_x = lowP1_x + leftToRight[None]
        highP1_y = upDownDis[i]

        if birdPosition[None].x > lowP1_x and birdPosition[None].x < highP1_x and birdPosition[None].y > lowP1_y and \
                birdPosition[None].y < highP1_y:
            gameState[None] = 1

        lowP2_x = lowP1_x
        lowP2_y = upDownDis[i] + midDis[None]

        highP2_x = lowP2_x + leftToRight[None]
        highP2_y = 40

        if birdPosition[None].x > lowP2_x and birdPosition[None].x < highP2_x and birdPosition[None].y > lowP2_y and \
                birdPosition[None].y < highP2_y:
            gameState[None] = 1

    for i in range(num_particles):
        dis = (positions[i] - birdPosition[None]).norm()
        if dis < 2:
            gameState[None] = 1
            break


# 以上为flybirds实现


screen_res = (800, 400)
screen_to_world_ratio = 10.0
boundary = (screen_res[0] / screen_to_world_ratio,
            screen_res[1] / screen_to_world_ratio)
cell_size = 2.51
cell_recpr = 1.0 / cell_size


def round_up(f, s):
    return (math.floor(f * cell_recpr / s) + 1) * s


grid_size = (round_up(boundary[0], 1), round_up(boundary[1], 1))

dragCoefficient = 0.4
h = 1.1
mass = 1.0
dim = 2
bg_color = 0x112f41
particle_color = 0x068587
boundary_color = 0xebaca2
num_particles_x = 60
num_particles = num_particles_x * 18
max_num_particles_per_cell = 100
max_num_neighbors = 100
time_delta = 1.0 / 20.0
epsilon = 1e-5
particle_radius = 3.0
particle_radius_in_world = particle_radius / screen_to_world_ratio

# 碰撞后处理速度需要的参数
restitutionCoefficient = 0.8
frictionCoeffient = 1000

# 迭代范围：(迭代需要用的参数)
minIterations = 2
maxIterations = 20
maxError = 0.01
# 静止密度
rho0 = 1.0
# 松弛系数
w = 0.5

neighbor_radius = h * 1.05
poly6_factor = 315.0 / 64.0 / math.pi
spiky_grad_factor = -45.0 / math.pi

forces = ti.Vector.field(dim, float)
a_ii = ti.field(float)
d_ii = ti.Vector.field(dim, float)
add_d_ij_pj = ti.Vector.field(dim, float)
density_adv = ti.field(float)
pressure = ti.field(float)
density = ti.field(float)
positions = ti.Vector.field(dim, float)
velocities = ti.Vector.field(dim, float)
grid_num_particles = ti.field(int)
grid2particles = ti.field(int)
particle_num_neighbors = ti.field(int)
particle_neighbors = ti.field(int)
board_states = ti.Vector.field(2, float)

ti.root.dense(ti.i, num_particles).place(positions, velocities, forces, d_ii, add_d_ij_pj)
grid_snode = ti.root.dense(ti.ij, grid_size)
grid_snode.place(grid_num_particles)
grid_snode.dense(ti.k, max_num_particles_per_cell).place(grid2particles)

nb_node = ti.root.dense(ti.i, num_particles)
nb_node.place(particle_num_neighbors)
nb_node.dense(ti.j, max_num_neighbors).place(particle_neighbors)

ti.root.dense(ti.i, num_particles).place(density, pressure, density_adv, a_ii)
ti.root.place(board_states)  # 0维张量


@ti.kernel
def move_board():
    # probably more accurate to exert force on particles according to hooke's law.
    b = board_states[None]
    b[1] += 1.0
    period = 90
    vel_strength = 8.0
    if b[1] >= 2 * period:
        b[1] = 0
    b[0] += -ti.sin(b[1] * np.pi / period) * vel_strength * time_delta
    board_states[None] = b


@ti.func
def poly6_value(s, h):
    result = 0.0
    if 0 < s and s < h:
        x = (h * h - s * s) / (h * h * h)
        result = poly6_factor * x * x * x
    return result


@ti.func
def spiky_gradient(r, h):
    result = ti.Vector([0.0, 0.0])
    r_len = r.norm()
    if 0 < r_len and r_len < h:
        x = (h - r_len) / (h * h * h)
        g_factor = spiky_grad_factor * x * x
        result = r * g_factor / r_len
    return result


@ti.func
def get_cell(pos):
    return int(pos * cell_recpr)


@ti.func
def is_in_grid(c):
    # @c: Vector(i32)
    return 0 <= c[0] and c[0] < grid_size[0] and 0 <= c[1] and c[
        1] < grid_size[1]


@ti.func
def confine_velocity_to_boundary(p, v):
    bmin = particle_radius_in_world
    bmax = ti.Vector([board_states[None][0], boundary[1]
                      ]) - particle_radius_in_world

    # 边界法向量（4个）
    norm_left = ti.Vector([1.0, 0.0])
    norm_down = ti.Vector([0.0, 1.0])
    norm_right = ti.Vector([-1.0, 0])
    norm_top = ti.Vector([0.0, -1.0])

    # 因为要处理速度，需要知道发生碰撞的是那条边界（4种）
    # 速度处理：(采用流体书的处理P153)
    # 左边界
    if p[0] <= bmin:
        # 速度处理
        normalDotRelativeVel = norm_left.dot(v)
        relativeVelN = normalDotRelativeVel * norm_left
        relativeVelT = v - relativeVelN
        if normalDotRelativeVel < 0.0:
            deltaRelativeVelN = (-restitutionCoefficient - 1.0) * relativeVelN
            relativeVelN *= -restitutionCoefficient
            if relativeVelT.norm() > 0.0:
                frictionScale = max(1.0 - frictionCoeffient * deltaRelativeVelN.norm() / relativeVelT.norm(), 0.0)
                relativeVelT *= frictionScale
        v = relativeVelN + relativeVelT
    # 下边界
    if p[1] <= bmin:
        # 速度处理
        normalDotRelativeVel = norm_down.dot(v)
        relativeVelN = normalDotRelativeVel * norm_down
        relativeVelT = v - relativeVelN
        if normalDotRelativeVel < 0.0:
            deltaRelativeVelN = (-restitutionCoefficient - 1.0) * relativeVelN
            relativeVelN *= -restitutionCoefficient
            if relativeVelT.norm() > 0.0:
                frictionScale = max(1.0 - frictionCoeffient * deltaRelativeVelN.norm() / relativeVelT.norm(), 0.0)
                relativeVelT *= frictionScale
        v = relativeVelN + relativeVelT
    # 右边界
    if p[0] >= bmax[0]:
        # 速度处理
        normalDotRelativeVel = norm_down.dot(v)
        relativeVelN = normalDotRelativeVel * norm_right
        relativeVelT = v - relativeVelN
        if normalDotRelativeVel < 0.0:
            deltaRelativeVelN = (-restitutionCoefficient - 1.0) * relativeVelN
            relativeVelN *= -restitutionCoefficient
            if relativeVelT.norm() > 0.0:
                frictionScale = max(1.0 - frictionCoeffient * deltaRelativeVelN.norm() / relativeVelT.norm(), 0.0)
                relativeVelT *= frictionScale
        v = relativeVelN + relativeVelT
    # 上边界
    if p[1] >= bmax[1]:
        # 速度处理
        normalDotRelativeVel = norm_down.dot(v)
        relativeVelN = normalDotRelativeVel * norm_top
        relativeVelT = v - relativeVelN
        if normalDotRelativeVel < 0.0:
            deltaRelativeVelN = (-restitutionCoefficient - 1.0) * relativeVelN
            relativeVelN *= -restitutionCoefficient
            if relativeVelT.norm() > 0.0:
                frictionScale = max(1.0 - frictionCoeffient * deltaRelativeVelN.norm() / relativeVelT.norm(), 0.0)
                relativeVelT *= frictionScale
        v = relativeVelN + relativeVelT
    return v


@ti.func
def confine_position_to_boundary(p):
    bmin = particle_radius_in_world
    bmax = ti.Vector([board_states[None][0], boundary[1]
                      ]) - particle_radius_in_world
    for i in ti.static(range(dim)):
        # Use randomness to prevent particles from sticking into each other after clamping
        if p[i] <= bmin:
            p[i] = bmin + epsilon * ti.random()
        elif bmax[i] <= p[i]:
            p[i] = bmax[i] - epsilon * ti.random()
    return p


@ti.kernel
def init_particles():
    for i in range(num_particles):
        delta = h * 0.8
        offs = ti.Vector([(boundary[0] - delta * num_particles_x) * 0.5,
                          boundary[1] * 0.02])
        positions[i] = ti.Vector([i % num_particles_x, i // num_particles_x
                                  ]) * delta + offs + ti.Vector([0.0, 20.0]) - ti.Vector([0, 30])

        for c in ti.static(range(dim)):
            velocities[i][c] = (ti.random() - 0.5) * 4
    board_states[None] = ti.Vector([boundary[0] - epsilon, -0.0])


def render(gui):
    gui.clear(bg_color)
    pos_np = positions.to_numpy()
    for j in range(dim):
        pos_np[:, j] *= screen_to_world_ratio / screen_res[j]
    gui.circles(pos_np, radius=particle_radius, color=particle_color)
    gui.rect((0, 0), (board_states[None][0] / boundary[0], 1),
             radius=1.5,
             color=boundary_color)

    gui.circle([birdPosition[None].x / 80, birdPosition[None].y / 40], radius=particle_radius * 2, color=0xebaca2)

    for i in range((wallPoint[None] // 2)):
        if wallIndex[None] != i:
            p1 = wallPositions[i]
            p2_x = wallPositions[i].x + leftToRight[None]
            p2_y = wallPositions[i].y + upDownDis[i]
            # 绘制障碍的下半部分
            gui.rect((p1.x / 80, (p1.y + upDownDis[i]) / 40), (p2_x / 80, (p2_y - upDownDis[i]) / 40), 2, 0xFFFFFF)

            # 绘制障碍的上半部分
            p3_x = p1.x
            p3_y = 40
            p4_x = p1.x + leftToRight[None]
            p4_y = p1.y + midDis[None] + upDownDis[i]
            gui.rect((p3_x / 80, p3_y / 40), (p4_x / 80, p4_y / 40), 2, 0xFFFFFF)

    if wallIndex[None] != -1:

        k = wallIndex[None]
        p5_x = otherP[None].x
        p5_y = 40
        p6_x = 80
        p6_y = upDownDis[k] + midDis[None]
        gui.rect((p5_x / 80, p5_y / 40), (80 / 80, p6_y / 40), 2, 0xFFFFFF)

        p5_xs = otherP[None].x
        p5_ys = upDownDis[k]
        p6_xs = 80
        p6_ys = 0
        gui.rect((p5_xs / 80, p5_ys / 40), (p6_xs / 80, p6_ys / 40), 2, 0xFFFFFF)

        p7_x = 0
        p7_y = 40
        p8_x = leftToRight[None] - (80 - p5_x)

        if p8_x < 0:
            p8_x = 0

        p8_y = p6_y
        gui.rect((p7_x / 80, p7_y / 40), (p8_x / 80, p8_y / 40), 2, 0xFFFFFF)

        p7_xs = 0
        p7_ys = upDownDis[k]
        p8_xs = p8_x
        p8_ys = 0

        gui.rect((p7_xs / 80, p7_ys / 40), (p8_xs / 80, p8_ys / 40), 2, 0xFFFFFF)
    #
    # gui.text(f' score:{fraction[None]}', pos=(0.05, 0.90), font_size=30, color=0xFF4500)

    if gui.is_pressed('e'):
        keySpeedUp()
        fraction_k[None] += 1
    if gui.is_pressed('q'):
        fraction_k[None] -= 1
        if fraction_k[None] <= 1:
            fraction_k[None] = 1
        keySolwDown()

    if gameState[None] != 1:
        fraction[None] += 0.1 * fraction_k[None]

    if gameState[None] == 1:
        isEnd[None] = 1
        gui.text(content='Game Over!',
                 pos=(0.32, 0.65),
                 font_size=45,
                 color=0xffffff)
        gui.text(content='Click the \'R\' to start again!`',
                 pos=(0.25, 0.45),
                 font_size=30,
                 color=0xffffff)

        if gui.is_pressed('r'):
            isEnd[None] = 0

    gui.show()


@ti.kernel
def prologue():
    # 常规操作，建立邻居搜索
    # 清除网格
    for I in ti.grouped(grid_num_particles):
        grid_num_particles[I] = 0
    for I in ti.grouped(particle_neighbors):
        particle_neighbors[I] = -1
    # 更新网格，计算每个粒子的归属网格
    for p_i in positions:
        cell = get_cell(positions[p_i])
        offs = ti.atomic_add(grid_num_particles[cell], 1)
        grid2particles[cell, offs] = p_i
    # 建立每个粒子的邻居的搜索
    for p_i in positions:
        pos_i = positions[p_i]
        cell = get_cell(pos_i)
        nb_i = 0
        for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2)))):
            cell_to_check = cell + offs
            if is_in_grid(cell_to_check):
                for j in range(grid_num_particles[cell_to_check]):
                    p_j = grid2particles[cell_to_check, j]
                    if nb_i < max_num_neighbors and p_j != p_i and (
                            pos_i - positions[p_j]).norm() < neighbor_radius:
                        particle_neighbors[p_i, nb_i] = p_j
                        nb_i += 1
        particle_num_neighbors[p_i] = nb_i


@ti.kernel
def advenction():
    # 遍历所有的粒子,更新密度 计算d_ii，预测速度；
    for p_i in range(num_particles):
        pos_i = positions[p_i]
        temp_density = 0.0
        temp_add_kernel = ti.Vector([0.0, 0.0])
        for j in range(particle_num_neighbors[p_i]):
            # p_j 当前粒子的邻居
            p_j = particle_neighbors[p_i, j]
            if p_j < 0:
                break
            pos_ji = pos_i - positions[p_j]
            temp_density += mass * poly6_value(pos_ji.norm(), h)
            temp_add_kernel += spiky_gradient(pos_ji, h)
        if temp_density > 0:
            density[p_i] = temp_density
        # 如果没有邻居，暂时按0.1计算！！！！！
        else:
            density[p_i] = 0.1
        # 因为0不能做除数，所以保证密度不能为0
        d_ii[p_i] = -time_delta ** 2 * mass / density[p_i] ** 2 * temp_add_kernel
        # 计算预测速度,预测只考虑重力，阻力
        # 首先清除力
        forces[p_i] = ti.Vector([0.0, 0.0])
        # 加入重力
        forces[p_i] += ti.Vector([0.0, -9.8])
        # #加入阻力
        var = velocities[p_i]
        temp_force = var * (-dragCoefficient)
        forces[p_i] += temp_force
        # 计算预测速度
        velocities[p_i] += time_delta * (forces[p_i] / mass)

    # 计算预测密度，初始压强，a_ii
    for p_i in range(num_particles):
        # 初始压强
        pressure[p_i] = 0.5 * pressure[p_i]
        pos_i = positions[p_i]
        veli = velocities[p_i]
        temp_density_adv = 0.0
        temp_aii = 0.0
        for j in range(particle_num_neighbors[p_i]):
            # p_j 当前粒子的邻居
            p_j = particle_neighbors[p_i, j]
            velj = velocities[p_j]
            pos_ji = pos_i - positions[p_j]
            pos_ij = positions[p_j] - pos_i
            grad_ij = spiky_gradient(pos_ji, h)
            grad_ji = spiky_gradient(pos_ij, h)
            temp_density_adv += time_delta * mass * (veli - velj).dot(grad_ij)
            dji = -time_delta ** 2 * mass / density[p_i] ** 2 * grad_ji
            temp_aii += mass * (d_ii[p_i] - dji).dot(grad_ij)
        density_adv[p_i] = density[p_i] + temp_density_adv
        # 避免出现分母为0的情况;
        if temp_aii == 0:
            temp_aii = -0.01
        # print("cuole")
        # print(temp_aii)
        a_ii[p_i] = temp_aii


@ti.kernel
def Pressure_Solve():
    l = 0
    # 这个终止条件暂时不知道怎么用：先随便设置一个参数
    # 初步理解：计算当前密度的平均值
    eta = maxError * 0.01 * rho0
    chk = False
    while l < minIterations or (l < maxIterations and ~chk):
        # 计算add_d_ij_pj
        chk = True
        for p_i in range(num_particles):
            temp_dij_pj = ti.Vector([0.0, 0.0])
            pos_i = positions[p_i]
            for j in range(particle_num_neighbors[p_i]):
                # p_j 当前粒子的邻居
                p_j = particle_neighbors[p_i, j]
                pressurej = pressure[p_j]
                posji = pos_i - positions[p_j]
                temp_dij_pj += -time_delta ** 2 * mass / density[p_j] ** 2 * pressurej * spiky_gradient(posji, h)
            add_d_ij_pj[p_i] = temp_dij_pj
        density_avg = 0.0
        for p_i in range(num_particles):
            pos_i = positions[p_i]
            apart_pressure = 0.0
            apart_density = 0.0
            for j in range(particle_num_neighbors[p_i]):
                # p_j 当前粒子的邻居
                p_j = particle_neighbors[p_i, j]
                pressurej = pressure[p_j]
                posji = pos_i - positions[p_j]
                posij = positions[p_j] - pos_i
                d_ji = -time_delta ** 2 * mass / density[p_i] ** 2 * spiky_gradient(posij, h)
                k_notequal_i = add_d_ij_pj[p_j] - d_ji * pressure[p_i]
                # 这块是点乘？？因为压强是一维
                apart_pressure += mass * (
                    (add_d_ij_pj[p_i] - d_ii[p_j] * pressurej - k_notequal_i).dot(spiky_gradient(posji, h)))
                apart_density += mass * (
                    (d_ii[p_i] * pressure[p_i] + add_d_ij_pj[p_i] - d_ii[p_j] * pressurej - k_notequal_i).dot(
                        spiky_gradient(posji, h)))
            pressure[p_i] = (1.0 - w) * pressure[p_i] + w / a_ii[p_i] * (rho0 - density_adv[p_i] - apart_pressure)

            if pressure[p_i] < 0:
                pressure[p_i] = 0
            if pressure[p_i] == 0.0:
                density_avg += rho0
            else:
                density_avg += apart_density + density_adv[p_i]
        # 终止条件，按照论文说法是计算平均密度，这步暂时不知道对不对
        density_avg /= num_particles
        # print( density_avg)
        chk = chk and (density_avg - rho0 <= eta)
        l += 1


@ti.kernel
def epilogue():
    # 计算压力
    for p_i in range(num_particles):
        den_pi = density[p_i]
        pressure_pi = pressure[p_i]
        pos_i = positions[p_i]
        pressureForces = ti.Vector([0.0, 0.0])
        for j in range(particle_num_neighbors[p_i]):
            # p_j 当前粒子的邻居
            p_j = particle_neighbors[p_i, j]
            if p_j < 0:
                break
            pos_ji = pos_i - positions[p_j]
            grad_ji = spiky_gradient(pos_ji, h)
            pressure_pj = pressure[p_j]
            den_pj = density[p_j]
            pressureForces -= mass * mass * (
                    pressure_pi / (den_pi * den_pi) + pressure_pj / (den_pj * den_pj)) * grad_ji
        # 此时力用来修正速度，只有压力作用。所以可以直接在force里只放压力
        forces[p_i] = pressureForces
        emitParticles()
    # 这步更新速度位置
    for i in range(num_particles):
        a = forces[i] / mass
        velocities[i] += a * time_delta
        positions[i] += velocities[i] * time_delta
        velocities[i] = confine_velocity_to_boundary(positions[i], velocities[i])
        positions[i] = confine_position_to_boundary(positions[i])


def run_iisph():
    prologue()
    advenction()
    Pressure_Solve()

    epilogue()


def run_flybirds():
    move_board()
    run_iisph()
    moveWall()

    updateBirdPosition()
    flyBirdsCollision()


def main():
    init_particles()
    initGame()
    print(f'boundary={boundary} grid={grid_size} cell_size={cell_size}')
    gui = ti.GUI('IISPH2D', screen_res)
    l = 0
    while gui.running and not gui.get_event(gui.ESCAPE):
        if isEnd[None] == 0:
            run_flybirds()

        if gui.is_pressed('r'):
            init_particles()
            initGame()

        if gui.is_pressed(ti.GUI.LMB):
            keyUp()

        if gui.is_pressed('d') and isEnd[None] == 0:
            keyBirdsMoveToRight()
        if gui.is_pressed('a') and isEnd[None] == 0:
            keyBirdsMoveToLeft()

        global delta_t

        delta_t = int(time.time() - t)

        render(gui)


if __name__ == '__main__':
    main()
