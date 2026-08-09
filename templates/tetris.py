"""Tetris — classic puzzle game in pure Python (tkinter). No dependencies.
Controls: Left/Right move, Up rotate, Down soft-drop, Space hard-drop, P pause.
Run:  python tetris.py
"""
import random
import sys
import tkinter as tk

COLS, ROWS, CELL = 10, 20, 28
W, H = COLS * CELL, ROWS * CELL

SHAPES = [
    ([[1, 1, 1, 1]], "#06b6d4"),            # I
    ([[1, 1], [1, 1]], "#facc15"),          # O
    ([[0, 1, 0], [1, 1, 1]], "#a855f7"),    # T
    ([[1, 0, 0], [1, 1, 1]], "#f97316"),    # L
    ([[0, 0, 1], [1, 1, 1]], "#3b82f6"),    # J
    ([[0, 1, 1], [1, 1, 0]], "#22c55e"),    # S
    ([[1, 1, 0], [0, 1, 1]], "#ef4444"),    # Z
]


def rotate(m):
    return [list(r) for r in zip(*m[::-1])]


root = tk.Tk()
root.title("Tetris — made by Flipps V0.1")
cv = tk.Canvas(root, width=W, height=H, bg="#111111")
cv.pack()

grid = [[0] * COLS for _ in range(ROWS)]
piece = None  # [shape, color, x, y]
score = 0
level = 1
paused = False
over = False


def new_piece():
    global piece
    shape, color = random.choice(SHAPES)
    piece = [shape, color, COLS // 2 - len(shape[0]) // 2, 0]


def fits(shape, x, y):
    for r, row in enumerate(shape):
        for c, v in enumerate(row):
            if v:
                bx, by = x + c, y + r
                if bx < 0 or bx >= COLS or by >= ROWS:
                    return False
                if by >= 0 and grid[by][bx]:
                    return False
    return True


def lock():
    global piece, over
    shape, color, x, y = piece
    for r, row in enumerate(shape):
        for c, v in enumerate(row):
            if v and 0 <= y + r < ROWS and 0 <= x + c < COLS:
                grid[y + r][x + c] = color
    clear_lines()
    new_piece()
    if not fits(piece[0], piece[2], piece[3]):
        over = True


def clear_lines():
    global score, level
    full = [r for r in range(ROWS) if all(grid[r])]
    for r in full:
        del grid[r]
        grid.insert(0, [0] * COLS)
    if full:
        score += [0, 100, 300, 500, 800][min(len(full), 4)] * level
        level = score // 1000 + 1


def draw():
    cv.delete("all")
    for r in range(ROWS):
        for c in range(COLS):
            if grid[r][c]:
                cv.create_rectangle(c * CELL, r * CELL, c * CELL + CELL, r * CELL + CELL,
                                    fill=grid[r][c], outline="#222222")
    shape, color, x, y = piece
    for r, row in enumerate(shape):
        for c, v in enumerate(row):
            if v and y + r >= 0:
                cv.create_rectangle((x + c) * CELL, (y + r) * CELL,
                                    (x + c + 1) * CELL, (y + r + 1) * CELL,
                                    fill=color, outline="#222222")
    status = f"Score: {score}   Level: {level}"
    if paused:
        status += "   PAUSED"
    if over:
        status += "   GAME OVER"
    cv.create_text(W // 2, 12, text=status, fill="white", font=("Arial", 12))


def step():
    if over or paused:
        return
    if fits(piece[0], piece[2], piece[3] + 1):
        piece[3] += 1
    else:
        lock()


def hard_drop():
    while fits(piece[0], piece[2], piece[3] + 1):
        piece[3] += 1
    lock()


def on_key(e):
    global paused
    if over:
        return
    k = e.keysym
    shape, color, x, y = piece
    if k == "Left" and fits(shape, x - 1, y):
        piece[2] -= 1
    elif k == "Right" and fits(shape, x + 1, y):
        piece[2] += 1
    elif k == "Down" and fits(shape, x, y + 1):
        piece[3] += 1
    elif k == "Up":
        r = rotate(shape)
        if fits(r, x, y):
            piece[0] = r
    elif k == "space":
        hard_drop()
    elif k == "p":
        paused = not paused


def loop():
    if not paused and not over:
        step()
    draw()
    root.after(400, loop)


if "--selftest" in sys.argv:
    root.withdraw()
    new_piece()
    draw()
    root.update_idletasks()
    root.destroy()
    print("tetris.py selftest OK")
else:
    new_piece()
    root.bind("<Key>", on_key)
    draw()
    root.after(400, loop)
    root.mainloop()
