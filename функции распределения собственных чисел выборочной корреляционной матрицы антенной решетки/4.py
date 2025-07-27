# рис. 4
# |F12| <=  N   !!!!
# q > 0 (ОСШ)

import tkinter as tk
import math  # для pi, ln()
import numpy as np
import random
from numpy import sqrt, sin, cos, pi, exp, inf
from scipy.integrate import quad  # для вычисления интеграла I
import scipy.special as sc  # для гипергеометрической функции Mhyper

# размеры окна
WIN_H = 800
WIN_W = 1200

# размеры панели
PANEL1_H = WIN_H
PANEL1_W = 300
PANEL2_H = 100
PANEL2_W = WIN_W - PANEL1_W

# размеры холста
CANVAS_H = WIN_H - PANEL2_H
CANVAS_W = WIN_W - PANEL1_W

win = tk.Tk()
win.title("График 4")
win.config(width=WIN_W, height=WIN_H)
win.resizable(False, False)

# создание панели инструментов (для дальнейшего помещения на нее кнопок и меток)
panel1 = tk.Frame(win, width=PANEL1_W, height=PANEL1_H, bd=4, relief=tk.GROOVE)
panel1.place(x=0, y=0, width=PANEL1_W, height=PANEL1_H)

panel2 = tk.Frame(win, width=PANEL2_W, height=PANEL2_H, bd=4, relief=tk.GROOVE)
panel2.place(x=PANEL1_W, y=0, width=PANEL2_W, height=PANEL2_H)

# создание холста
canvas = tk.Canvas(win, width=CANVAS_W, height=CANVAS_H, bg="white")
canvas.place(x=PANEL1_W, y=PANEL2_H, width=CANVAS_W, height=CANVAS_H)

"""функция для отображения осей координат
параметры - x_left, x_right, y_bottom, y_top - это границы графика функции в системе координат графика
"""


def draw_axis(x_left, x_right, y_bottom, y_top):
    """ вычисление стоимости 1 пикселя холста (кол-во точек графика, которые помещаются в 1 пикселе холста)
    """
    dx = CANVAS_W / (x_right - x_left)
    dy = CANVAS_H / (y_top - y_bottom)

    """координаты центра координат в системе холста"""
    cx = -x_left * dx
    cy = y_top * dy

    # рисуем оси
    canvas.create_line(0, cy, CANVAS_W, cy, fill='#012', width=2, arrow=tk.LAST)
    canvas.create_line(cx, 0, cx, CANVAS_H, fill='#012', width=2, arrow=tk.FIRST)

    """разметка горизонтальной оси"""
    N_x = 11  # количество интервалов (штрихов) - меняем
    x_step = (x_right - x_left) / N_x
    x = x_left
    while x <= x_right:
        x_canvas = (x - x_left) * dx
        canvas.create_line(x_canvas, cy - 3, x_canvas, cy + 3, fill="#012")
        canvas.create_text(x_canvas + 15, cy + 15, text=str(round(x, 1)), font="Verdana 9", fill="#012")
        x += x_step

    """разметка вертикальной оси"""
    N_y = 12  # количество интервалов (штрихов) - меняем
    y_step = (y_top - y_bottom) / N_y
    y = y_top
    while y >= y_bottom:
        y_canvas = (y - y_top) * dy
        canvas.create_line(cx - 3, -y_canvas, cx + 3, -y_canvas, fill="#012")
        canvas.create_text(cx + 20, -y_canvas - 10, text=str(round(y, 1)), font="Verdana 9", fill="#012")
        y -= y_step

    """рисуем сетку (вертикальные линии)"""
    x_step = (x_right - x_left) / N_x
    x = x_left
    while x <= x_right:
        x_canvas = (x - x_left) * dx
        canvas.create_line(x_canvas, cy - y_bottom * dy, x_canvas, 0, fill="#012", width=1, dash=(6, 4))
        x += x_step

    """рисуем сетку (горизонтальные линии)"""
    y_step = (y_top - y_bottom) / N_y
    y = y_top
    while y >= y_bottom:
        y_canvas = (y - y_top) * dy
        canvas.create_line(0, -y_canvas, cx + x_right * dx, -y_canvas, fill="#012", width=1, dash=(6, 4))
        y -= y_step

    """Названия осей"""
    canvas.create_text(cx - 15, 20, text="H", fill="black", font=("Helvetica 12 bold"))
    canvas.create_text((x_right - x_left) * dx - 30, cy + 15, text="q", fill="black", font=("Helvetica 12 bold"))

    return dx, dy


"""Нанесение точек графика на холст 
x_tmp, y_tmp - список координат X,Y
функция graph_dot - преобразование координат в точки на холсте
"""


def graph_dot(x_tmp, y_tmp, color):
    dot_list = []
    i = 0
    for x in x_tmp:
        y = y_tmp[i]
        # пересчет координат в пиксели холста
        x_c = (x - x_left) * dx
        y_c = (y_top - y) * dy
        """функция create_oval рисует точку. Возвращает объект холста"""
        dot = canvas.create_oval(x_c - 1, (y_c - 1), x_c + 1, (y_c + 1), fill=color, outline=color, width=2)
        dot_list.append(dot)
        i += 1

    """Нанесение подписей к графикам"""
    q = float(ent1.get())
    text = "q = " + str(float(q))
    cx = -x_left * dx;
    x_t = cx + x_right * dx - 50;
    y_t = 20 * (len(sign_dot_list) + 1)
    sign = canvas.create_text(x_t, y_t, text=text, fill=color, font=("Helvetica 12 bold"))
    sign_dot_list.append(sign)
    return dot_list


def W(fi, q):
    a = 1 / (2 * pi) * exp(-q ** 2 / 2)
    b = q * cos(fi) / sqrt(2 * pi)
    c = 1 / sqrt(2 * pi)

    func = lambda t: exp(-t ** 2 / 2)
    I = quad(func, -inf, q * cos(fi))[0]

    d = exp(-q ** 2 * (sin(fi)) ** 2 / 2)

    return a + b * c * I * d


def W_gr(fi, q):
    gr = []
    for i in range(len(fi)):
        gr.append(W(fi[i], q))
    return gr


def D(q):
    a = 150
    b = pi ** 2 / 3
    c = 4 * pi
    d = 0
    for n in range(1, a + 1):
        e = 1 / n ** 2
        Gamma = math.gamma(1 + n / 2)
        f = (-1 * q) ** n
        Mhyper = sc.hyp1f1(n / 2, n + 1, -q ** 2 / 2)
        m = pow(2 ** n, 0.5)
        g = pi * m * math.factorial(n)
        d = d + e * Gamma * f * Mhyper / g
    return b + c * d


"""Обработчик кнопки "Отобразить энтропию"""
"""Построить график заново"""


def W_redraw():
    q = float(ent1.get())
    if (q > 0):
        global W_list  # создание глобального списка значений Y
        global W_dots  # объявление глобальным списка значений точек на холсте

        # удаление старого графика
        if (len(W_dots_list) > 0):
            for dot in W_dots:
                canvas.delete(dot)  # у холста есть метод delete
            W_dots_list.pop(len(W_dots_list) - 1)
        if (len(sign_dot_list) > 0):
            canvas.delete(sign_dot_list[len(sign_dot_list) - 1])
            sign_dot_list.pop(len(sign_dot_list) - 1)
        W_list = W_gr(fi_list, q)
        cr = random.randint(0, len(colors) - 1)
        W_dots = graph_dot(fi_list, W_list, colors[cr])
        W_dots_list.append(W_dots)


"""Добавить на график"""


def W_redraw2():
    q = float(ent1.get())
    if (q > 0):
        global W_list  # создание глобального списка значений Y
        global W_dots  # объявление глобальным списка значений точек на холсте
        W_list = W_gr(fi_list, q)
        cr = random.randint(0, len(colors) - 1)
        W_dots = graph_dot(fi_list, W_list, colors[cr])
        W_dots_list.append(W_dots)


"""Стереть последний график"""


def clean_draw():
    index = len(sign_dot_list) - 1
    # удаление старого графика
    if (index >= 0):
        for dot in W_dots_list[index]:
            canvas.delete(dot)  # у холста есть метод delete
        canvas.delete(sign_dot_list[index])
        sign_dot_list.pop(index)

        W_dots_list.pop(index)


"""Стереть все графики"""


def clean_draw_all():
    if (len(sign_dot_list) > 0):
        for W_dots in W_dots_list:
            # удаление старого графика
            for dot in W_dots:
                canvas.delete(dot)  # у холста есть метод delete
            canvas.delete(sign_dot_list[len(sign_dot_list) - 1])
            sign_dot_list.pop(len(sign_dot_list) - 1)
        W_dots_list.clear()


"""ДОБАВЛЕНИЕ ЭЛЕМЕНТОВ НА ПАНЕЛЬ ИНСТРУМЕНТОВ """
""" 1. Добавление меток """
lab1 = tk.Label(panel1, text="ОСШ: ", font='Times 14')
lab1.place(x=35, y=150, width=200, height=50)

lab2 = tk.Label(panel1, text="   q = ", font='Times 14')
lab2.place(x=50, y=230, width=30, height=50)

lab3 = tk.Label(panel2, text="Распределение фазы аддитивной смеси  " + "\n\
детерминированного сигнала  и собственного шума" +
                             "\n при различных значениях ОСШ \n ",
                font='Times 14')
lab3.place(x=0, y=0, width=800, height=100)

""" 2. Добавление полей для ввода  """
ent1 = tk.Entry(panel1, bd=5, font='Times 14')
ent1.place(x=90, y=230, width=90, height=50)
ent1.insert(5, "2")

""" 3. Добавление кнопки и обработчика  """
but1 = tk.Button(panel1, text="Построить \n заново", command=W_redraw, font='Times 14')
but1.place(x=50, y=280, width=200, height=50)

but2 = tk.Button(panel1, text="Добавить на \n  график", command=W_redraw2, font='Times 14')
but2.place(x=50, y=380, width=200, height=50)

but3 = tk.Button(panel1, text="Стереть \n последний график", command=clean_draw, font='Times 14')
but3.place(x=50, y=480, width=200, height=50)

but4 = tk.Button(panel1, text="Стереть \n все графики", command=clean_draw_all, font='Times 14')
but4.place(x=50, y=580, width=200, height=50)


def frange(begin, end, step):
    x = begin
    t = []
    while x <= end:
        t.append(x)
        x += step
    return t


# координаты границ графика - меняем
x_left, x_right = -2, 2
y_bottom, y_top = -1, 5
dx, dy = draw_axis(x_left, x_right, y_bottom, y_top)

"""Вычисление значений X графика параболы"""
fi_step1 = 0.001  # шаг, с которым строим график  - меняем
fi_list = frange(x_left, x_right, fi_step1)
q = float(ent1.get())
W_list = W_gr(fi_list, q)

# СКО:
sigma1 = sqrt(D(q))
print("σ_φ = ", sigma1)

# высоты СКО:
# 1
x1 = sigma1;
y1 = W(x1, q)
print(x1, y1)

colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black', 'orange', 'purple', 'lime', 'brown']
cr = random.randint(0, len(colors) - 1)
"""Нанесение точек графика параболы  на холст """
"""в parabola - записывается список точек"""
W_dots_list = []
global sign_dot_list  # подписи к графикам
sign_dot_list = []
W_dots = graph_dot(fi_list, W_list, colors[cr])  # один график
W_dots_list.append(W_dots)  # все графики

win.mainloop()
