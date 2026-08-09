"""Snake — classic game in pure Python (tkinter). No dependencies.
Controls: arrow keys. Eat red food, don't hit the walls or yourself.
Run:  python snake.py
"""
import random
import sys
import tkinter as tk

W, CELL, SPEED = 400, 20, 110  # pixels, cell size, ms per tick

root = tk.Tk()
root.title("Snake — made by Flipps V0.1")
cv = tk.Canvas(root, width=W, height=W, bg="#111111")
cv.pack()

snake = [(10, 10), (9, 10), (8, 10)]
direction = (1, 0)
food = None
score = 0
over = False


def place_food():
    global food
    while True:
        f = (random.randrange(W // CELL), random.randrange(W // CELL))
        if f not in snake:
            food = f
            return


def draw():
    cv.delete("all")
    for x, y in snake:
        cv.create_rectangle(x * CELL, y * CELL, x * CELL + CELL, y * CELL + CELL,
                            fill="#4ade80", outline="#166534")
    if food:
        fx, fy = food
        cv.create_rectangle(fx * CELL, fy * CELL, fx * CELL + CELL, fy * CELL + CELL,
                            fill="#f87171", outline="")
    cv.create_text(W // 2, 16, text=f"Score: {score}", fill="white", font=("Arial", 14))


def tick():
    global snake, food, score, over
    if over:
        return
    head = (snake[0][0] + direction[0], snake[0][1] + direction[1])
    hit_wall = not (0 <= head[0] < W // CELL and 0 <= head[1] < W // CELL)
    if hit_wall or head in snake:
        over = True
        cv.create_text(W // 2, W // 2, text=f"Game over — score {score}",
                       fill="white", font=("Arial", 22))
        return
    snake.insert(0, head)
    if head == food:
        score += 1
        place_food()
    else:
        snake.pop()
    draw()
    root.after(SPEED, tick)


def on_key(e):
    global direction
    k = e.keysym
    if k == "Up" and direction != (0, 1):
        direction = (0, -1)
    elif k == "Down" and direction != (0, -1):
        direction = (0, 1)
    elif k == "Left" and direction != (1, 0):
        direction = (-1, 0)
    elif k == "Right" and direction != (-1, 0):
        direction = (1, 0)


if "--selftest" in sys.argv:
    root.withdraw()
    place_food()
    draw()
    root.update_idletasks()
    root.destroy()
    print("snake.py selftest OK")
else:
    place_food()
    root.bind("<Key>", on_key)
    draw()
    root.after(SPEED, tick)
    root.mainloop()
