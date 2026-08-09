"""Voxel World — a mini-Minecraft built from scratch in Python (pyglet + OpenGL).

Controls:
  WASD          move
  Mouse         look around
  Space         jump
  Left click    break block
  Right click   place block
  1-5           choose block: 1 grass, 2 dirt, 3 stone, 4 wood, 5 leaves
  Esc           release the mouse cursor

Requires:  pip install pyglet
Run:       python voxel_world.py
"""
import math
import random
import sys

import pyglet
from pyglet.gl import *
from pyglet.window import key, mouse

W, H = 1024, 640
WORLD_W, WORLD_D, WORLD_H = 32, 32, 24
SEED = 1337

AIR, GRASS, DIRT, STONE, WOOD, LEAVES = 0, 1, 2, 3, 4, 5

# per-block colors: (top, side, bottom)
COLORS = {
    GRASS: ((0.45, 0.80, 0.30), (0.42, 0.62, 0.28), (0.35, 0.25, 0.15)),
    DIRT: ((0.55, 0.38, 0.22), (0.50, 0.35, 0.20), (0.42, 0.30, 0.18)),
    STONE: ((0.55, 0.55, 0.58), (0.50, 0.50, 0.52), (0.42, 0.42, 0.45)),
    WOOD: ((0.60, 0.45, 0.28), (0.52, 0.38, 0.24), (0.45, 0.33, 0.21)),
    LEAVES: ((0.30, 0.60, 0.25), (0.26, 0.52, 0.22), (0.20, 0.42, 0.18)),
}

# face normals + corner vertices (unit cube)
FACES = [
    ((0, 1, 0), ((0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1))),      # top
    ((0, -1, 0), ((0, 0, 1), (1, 0, 1), (1, 0, 0), (0, 0, 0))),     # bottom
    ((1, 0, 0), ((1, 0, 0), (1, 0, 1), (1, 1, 1), (1, 1, 0))),      # +x
    ((-1, 0, 0), ((0, 0, 1), (0, 0, 0), (0, 1, 0), (0, 1, 1))),     # -x
    ((0, 0, 1), ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))),      # +z
    ((0, 0, -1), ((1, 0, 0), (0, 0, 0), (0, 1, 0), (1, 1, 0))),     # -z
]

# --------------------------------------------------------------------------- #
# World + terrain (layered value noise)
# --------------------------------------------------------------------------- #
world = [[[AIR] * WORLD_D for _ in range(WORLD_H)] for _ in range(WORLD_W)]


def _hash(x, z):
    n = (x * 374761393 + z * 668265263 + SEED * 1274126177) & 0xFFFFFFFF
    n = (n ^ (n >> 13)) * 1274126177 & 0xFFFFFFFF
    return ((n ^ (n >> 16)) & 0xFFFF) / 65535.0


def _smooth(t):
    return t * t * (3 - 2 * t)


def height(x, z):
    h = 0.0
    amp, freq = 1.0, 0.06
    for _ in range(3):
        xi, zi = x * freq, z * freq
        x0, z0 = int(math.floor(xi)), int(math.floor(zi))
        fx, fz = _smooth(xi - x0), _smooth(zi - z0)
        v00, v10 = _hash(x0, z0), _hash(x0 + 1, z0)
        v01, v11 = _hash(x0, z0 + 1), _hash(x0 + 1, z0 + 1)
        top = v00 + (v10 - v00) * fx
        bot = v01 + (v11 - v01) * fx
        h += (top + (bot - top) * fz) * amp
        amp *= 0.5
        freq *= 2.2
    return max(3, min(18, 3 + int(h * 12)))


def set_block(x, y, z, b):
    if 0 <= x < WORLD_W and 0 <= y < WORLD_H and 0 <= z < WORLD_D:
        world[x][y][z] = b


def get_block(x, y, z):
    if x < 0 or x >= WORLD_W or z < 0 or z >= WORLD_D:
        return STONE  # solid world borders
    if y < 0:
        return STONE
    if y >= WORLD_H:
        return AIR
    return world[x][y][z]


def build_world():
    heights = {}
    for x in range(WORLD_W):
        for z in range(WORLD_D):
            h = height(x, z)
            heights[(x, z)] = h
            for y in range(h + 1):
                b = STONE
                if y == h:
                    b = GRASS
                elif y >= h - 3:
                    b = DIRT
                set_block(x, y, z, b)
    rng = random.Random(SEED)
    for _ in range(22):
        x = rng.randrange(3, WORLD_W - 3)
        z = rng.randrange(3, WORLD_D - 3)
        h = heights.get((x, z), 8)
        for y in range(h + 1, h + 5):
            set_block(x, y, z, WOOD)
        for dy in range(3):
            r = 2 if dy == 0 else 1
            for dx in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    if dx * dx + dz * dz <= r * r and not (dx == 0 and dz == 0 and dy == 2):
                        set_block(x + dx, h + 4 + dy, z + dz, LEAVES)


# --------------------------------------------------------------------------- #
# Mesh (display list, rebuilt when blocks change)
# --------------------------------------------------------------------------- #
mesh_list = None


def rebuild_mesh():
    global mesh_list
    if mesh_list:
        glDeleteLists(mesh_list, 1)
    mesh_list = glGenLists(1)
    glNewList(mesh_list, GL_COMPILE)
    for x in range(WORLD_W):
        for y in range(WORLD_H):
            for z in range(WORLD_D):
                b = world[x][y][z]
                if not b:
                    continue
                for (dx, dy, dz), verts in FACES:
                    if get_block(x + dx, y + dy, z + dz):
                        continue
                    col = COLORS[b][0] if dy == 1 else (COLORS[b][2] if dy == -1 else COLORS[b][1])
                    glColor3f(*col)
                    glBegin(GL_QUADS)
                    for vx, vy, vz in verts:
                        glVertex3f(x + vx, y + vy, z + vz)
                    glEnd()
    glEndList()


# --------------------------------------------------------------------------- #
# Player
# --------------------------------------------------------------------------- #
PW, PH = 0.30, 1.8
px, py, pz = 0.0, 0.0, 0.0
vy = 0.0
yaw, pitch = 45.0, -25.0
on_ground = False
keys_down = set()
selected = GRASS

SENS = 0.15
GRAVITY = -28.0
JUMP = 9.0
SPEED = 7.0


def collides(x, y, z):
    for dx in (-PW, PW):
        for dz in (-PW, PW):
            for dy in (0.0, PH - 0.05):
                if get_block(int(x + dx), int(y + dy), int(z + dz)):
                    return True
    return False


def try_move(dx, dy, dz):
    global px, py, pz, on_ground
    if not collides(px + dx, py, pz):
        px += dx
    if not collides(px, py + dy, pz):
        py += dy
        on_ground = False
    else:
        on_ground = dy < 0
    if not collides(px, py, pz + dz):
        pz += dz


def update(dt):
    global vy, on_ground
    dt = min(dt, 0.05)
    fx = fz = 0.0
    if key.W in keys_down:
        fx += math.sin(math.radians(yaw))
        fz += math.cos(math.radians(yaw))
    if key.S in keys_down:
        fx -= math.sin(math.radians(yaw))
        fz -= math.cos(math.radians(yaw))
    if key.A in keys_down:
        fx += math.sin(math.radians(yaw - 90))
        fz += math.cos(math.radians(yaw - 90))
    if key.D in keys_down:
        fx -= math.sin(math.radians(yaw - 90))
        fz -= math.cos(math.radians(yaw - 90))
    n = math.hypot(fx, fz)
    if n:
        fx, fz = fx / n * SPEED, fz / n * SPEED
    vy = max(vy + GRAVITY * dt, -20.0)
    if on_ground and key.SPACE in keys_down:
        vy = JUMP
        on_ground = False
    try_move(fx * dt, 0.0, fz * dt)
    steps = max(1, int(abs(vy * dt) / 0.4) + 1)
    for _ in range(steps):
        try_move(0.0, vy * dt / steps, 0.0)


def look_dir():
    r = math.radians(pitch)
    a = math.radians(yaw)
    return (math.cos(r) * math.sin(a), -math.sin(r), math.cos(r) * math.cos(a))


def raycast(dist=6.0):
    ox, oy, oz = px, py + 1.5, pz
    dx, dy, dz = look_dir()
    prev = None
    t = 0.0
    while t < dist:
        bx, by, bz = int(ox + dx * t), int(oy + dy * t), int(oz + dz * t)
        if get_block(bx, by, bz):
            return (bx, by, bz), prev
        prev = (bx, by, bz)
        t += 0.02
    return None, None


# --------------------------------------------------------------------------- #
# Window + events
# --------------------------------------------------------------------------- #
window = pyglet.window.Window(W, H, caption="Voxel World — mini-Minecraft by Flipps V0.1",
                              resizable=True, vsync=True)


def setup_3d():
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(70, window.width / max(1, window.height), 0.1, 200)
    glMatrixMode(GL_MODELVIEW)


@window.event
def on_resize(w, h):
    glViewport(0, 0, w, h)
    setup_3d()


@window.event
def on_draw():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    glRotatef(-pitch, 1, 0, 0)
    glRotatef(-yaw, 0, 1, 0)
    glTranslatef(-px, -py, -pz)
    glCallList(mesh_list)


@window.event
def on_mouse_motion(x, y, dx, dy):
    global yaw, pitch
    yaw += dx * SENS
    pitch += dy * SENS
    pitch = max(-89, min(89, pitch))


@window.event
def on_mouse_press(x, y, button, modifiers):
    hit, prev = raycast()
    if not hit:
        return
    bx, by, bz = hit
    if button == mouse.LEFT:
        set_block(bx, by, bz, AIR)
        rebuild_mesh()
    elif button == mouse.RIGHT and prev:
        bx, by, bz = prev
        if collides(bx + 0.5, by + 0.5, bz + 0.5):
            return  # don't place a block inside the player
        set_block(bx, by, bz, selected)
        rebuild_mesh()


@window.event
def on_key_press(symbol, modifiers):
    global selected
    keys_down.add(symbol)
    if symbol == key.ESCAPE:
        window.set_exclusive_mouse(False)
        window.set_mouse_visible(True)
    for num, b in ((key._1, GRASS), (key._2, DIRT), (key._3, STONE),
                   (key._4, WOOD), (key._5, LEAVES)):
        if symbol == num:
            selected = b


@window.event
def on_key_release(symbol, modifiers):
    keys_down.discard(symbol)


def main():
    build_world()
    rebuild_mesh()
    global px, py, pz
    px = WORLD_W // 2 + 0.5
    pz = WORLD_D // 2 + 0.5
    py = height(int(px), int(pz)) + 2.0
    setup_3d()
    window.set_exclusive_mouse(True)
    pyglet.clock.schedule(update)
    pyglet.app.run()


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        build_world()
        solid = sum(1 for x in range(WORLD_W) for y in range(WORLD_H)
                    for z in range(WORLD_D) if world[x][y][z])
        trees = sum(1 for x in range(WORLD_W) for y in range(WORLD_H)
                    for z in range(WORLD_D) if world[x][y][z] == WOOD)
        print(f"voxel_world.py selftest OK — {solid} solid blocks, {trees} wood blocks, "
              f"spawn height {height(WORLD_W // 2, WORLD_D // 2)}")
    else:
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glClearColor(0.62, 0.78, 0.95, 1.0)
        main()
