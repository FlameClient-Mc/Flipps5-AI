"""Pong — two-player classic in pure Python (tkinter). No dependencies.
Player 1 (left):  W / S      Player 2 (right):  Up / Down
Run:  python pong.py
"""
import random
import sys
import tkinter as tk

W, H = 600, 400
PAD_W, PAD_H, BALL = 10, 70, 9

root = tk.Tk()
root.title("Pong — made by Flipps V0.1")
cv = tk.Canvas(root, width=W, height=H, bg="#111111")
cv.pack()

p1, p2 = H // 2 - PAD_H // 2, H // 2 - PAD_H // 2
bx, by = W // 2, H // 2
bdx, bdy = random.choice([-5, 5]), random.choice([-3, 3])
s1 = s2 = 0


def draw():
    cv.delete("all")
    cv.create_rectangle(8, p1, 8 + PAD_W, p1 + PAD_H, fill="white")
    cv.create_rectangle(W - 8 - PAD_W, p2, W - 8, p2 + PAD_H, fill="white")
    cv.create_oval(bx - BALL, by - BALL, bx + BALL, by + BALL, fill="white")
    cv.create_line(W // 2, 0, W // 2, H, dash=(8, 8), fill="#444444")
    cv.create_text(W // 2, 20, text=f"{s1}   :   {s2}", fill="white", font=("Arial", 20))


def tick():
    global bx, by, bdx, bdy, s1, s2
    bx += bdx
    by += bdy
    if by < BALL or by > H - BALL:
        bdy = -bdy
    if bx - BALL < 8 + PAD_W and p1 < by < p1 + PAD_H and bdx < 0:
        bdx = -bdx
    elif bx + BALL > W - 8 - PAD_W and p2 < by < p2 + PAD_H and bdx > 0:
        bdx = -bdx
    if bx < -30:
        s2 += 1
        bx, by = W // 2, H // 2
    elif bx > W + 30:
        s1 += 1
        bx, by = W // 2, H // 2
    draw()
    root.after(16, tick)


def on_key(e):
    global p1, p2
    k = e.keysym
    if k == "w" and p1 > 0:
        p1 -= 6
    elif k == "s" and p1 < H - PAD_H:
        p1 += 6
    elif k == "Up" and p2 > 0:
        p2 -= 6
    elif k == "Down" and p2 < H - PAD_H:
        p2 += 6


if "--selftest" in sys.argv:
    root.withdraw()
    draw()
    root.update_idletasks()
    root.destroy()
    print("pong.py selftest OK")
else:
    root.bind("<Key>", on_key)
    draw()
    root.after(16, tick)
    root.mainloop()
