# 太极图形课S1-大作业 ---三个基于流体模拟的小游戏
stable_fluid+MPM+IISPH


# 作业来源

主要是根据流体模拟中比较常见的几个方法的2D版本做了修改，写了三个小游戏。

“射箭小游戏”是根据games201中给的stable fluid代码进行的修改。基本过程，使用鼠标控制箭的方向，按键进行箭的释放。以箭的速度为基准向矢量场中注入能量。使用Stable Fluid的相关操作对矢量场进行处理。产生烟雾的模拟效果。使用AABB-圆的碰撞检测技术，完成箭与球体的碰撞检测方案。当箭碰到小球后，小球消失。分数累加。如果小球超过基准线，但是并未被射中。游戏结束。

“飞翔的小鸟小游戏”流体部分采用隐式不可压缩的光滑粒子动力学方法（IISPH），基于连续方程对密度进行限制。除了IISPH本身对流体的模拟，这里人为加入一些噪声，具体思路为：随机选取靠近空气的流体粒子，并赋予其一个速度，达到一种沸腾的流体的效果，增加对小鸟飞翔过程中的干扰。其中的移动边界也会改变流体的状态，达到一种干扰的目的

“打方块小游戏”使用MPM框架模拟流体，初始时流体在方块内部，球体消灭方块后，流体突破方块的约束通过自由落体运动下落到地面，产生简单的流体落在地面的效果。

参考资料：

Stam, Jos. "Stable fluids." Proceedings of the 26th annual conference on Computer graphics and interactive techniques. 1999.

Ihmsen, Markus, et al. "Implicit incompressible SPH." IEEE transactions on visualization and computer graphics 20.3 (2013): 426-435.

Jiang, Chenfanfu, et al. "The material point method for simulating continuum materials." ACM SIGGRAPH 2016 Courses. 2016. 1-52.

https://www.bilibili.com/video/BV1ZK411H7Hc?spm_id_from=333.999.0.0   （Games201）

# 运行方式

运行环境：taichi

运行

三个游戏是互相独立的，选择要运行的游戏就可


# 效果展示

![Image](https://github.com/MengMeng3399/HomeWork/blob/main/%E6%95%88%E6%9E%9C%E5%9B%BE--1.gif)

![Image](https://github.com/MengMeng3399/HomeWork/blob/main/%E6%95%88%E6%9E%9C%E5%9B%BE--2.gif)

![Image](https://github.com/MengMeng3399/HomeWork/blob/main/%E6%95%88%E6%9E%9C%E5%9B%BE--3.gif)


# 整体结构
cv2.py

fly_brid.py

game.py

# 实现细节

整体流程：

1.“射箭小游戏” 


![image](https://user-images.githubusercontent.com/86776127/147383025-6f63402f-2194-437d-baec-d1cb2cc10da3.png)

1） 添加外力

2） 对流项。由于速度场的存在，流体不仅有自发的扩散运动，也有因速度带来的流动，也就是对流项。在此采用后向追溯的方法，使用半拉格朗日，通过当前单元格的速度来计算该处状态在前一dt时的位置

3） 求解粘度效应，等效为扩散方程

4） 投影。由于经历以上三步可能导致速度场有散度。所以需要将其投影到无散的速度场。

2.“飞翔的小鸟小游戏”

通过光滑核函数近似物理场，因此可以用于求解N-S方程。由于模拟的是液体，于是不考虑粘度项的计算。

1）确定好粒子的初始位置，利用插值公式计算出粒子的密度

2）计算预测密度

3）求解线性系统，（此处使用松弛-雅可比迭代）

![RCLF9YM`TE6QQZAE`E{ 1B6](https://user-images.githubusercontent.com/86776127/147383141-e597b4a2-ff92-4db8-a1c1-7b11cbaf3bed.png)


4）将求解得到的压强代入下式，求出压力。

![MJ3DG28 {{UE2 V@ OU@Y 6](https://user-images.githubusercontent.com/86776127/147383157-df6c8dbd-99d2-4e6d-878d-27e50408c9f1.png)


5）使用半隐式的方法更新粒子的最终位置


3.“打方块小游戏”

1）P2G 将粒子携带信息转移到网格上

2）Gop 网格操作

3）G2G 在网格进行完操作后转移回粒子

4）粒子移动


# 代码细节：

1.“射箭小游戏” 

射箭小游戏中以以箭的速度为基准向矢量场中注入能量，使用Stable Fluid的相关操作对矢量场进行处理。产生烟雾的模拟效果。

~~~
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
        # 密度消散
        dc *= dye_decay
        dyef[i, j] = dc、
   ~~~
      
      
2.相关碰撞检测

~~~
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
~~~
