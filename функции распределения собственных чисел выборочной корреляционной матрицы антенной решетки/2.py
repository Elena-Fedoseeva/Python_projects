# рис. 2   
# |F12| <=  N   !!!!
import tkinter as tk
import math  # для pi, ln()
import numpy as np
import random

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
win.title("График 2")
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

    """рисуем линию H = ln(2)"""
    y1 = math.log(2)
    canvas.create_line(cx, (y_top - y1) * dy, cx + x_right * dx, (y_top - y1) * dy, fill="#012", width=4, dash=2)
    cx = -x_left * dx;
    x_t = cx + x_right * dx - 20;
    y_t = (y_top - y1) * dy - 15
    sign = canvas.create_text(x_t - 10, y_t - 5, text="H=ln(2)", fill="black", font=("Helvetica 12 bold"))

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
    F12 = float(ent1.get())
    text = "F12 = " + str(float(F12))
    cx = -x_left * dx;
    x_t = cx + x_right * dx - 50;
    y_t = 20 * (len(sign_dot_list) + 1)
    sign = canvas.create_text(x_t, y_t, text=text, fill=color, font=("Helvetica 12 bold"))
    sign_dot_list.append(sign)
    return dot_list


def H2(q, N, F12):
    H = 0
    lambda0 = [0]
    for i in range(1, N + 1):
        if i == 1:
            lambda0.append((q ** 2 * (N + F12) + 1) / (2 * q ** 2 * N + N))  # |F12| <=  N   !!!!
        elif i == 2:
            lambda0.append((q ** 2 * (N - F12) + 1) / (2 * q ** 2 * N + N))
        else:
            lambda0.append(1 / (2 * q ** 2 * N + N))

        H = H - lambda0[i] * math.log(lambda0[i])
    return (H)


def H2_gr(q, N, F12):
    gr = []
    for i in range(len(q)):
        gr.append(H2(q[i], N, F12))
    return gr


"""Обработчик кнопки "Отобразить энтропию"""
"""Построить график заново"""


def H2_redraw():
    F12 = float(ent1.get())
    if (abs(F12) <= N):
        global H2_list  # создание глобального списка значений Y
        global H2_dots  # объявление глобальным списка значений точек на холсте

        # удаление старого графика 
        if (len(H2_dots_list) > 0):
            for dot in H2_dots:
                canvas.delete(dot)  # у холста есть метод delete
            H2_dots_list.pop(len(H2_dots_list) - 1)
        if (len(sign_dot_list) > 0):
            canvas.delete(sign_dot_list[len(sign_dot_list) - 1])
            sign_dot_list.pop(len(sign_dot_list) - 1)
        H2_list = H2_gr(q_list, N, F12)
        cr = random.randint(0, len(colors) - 1)
        H2_dots = graph_dot(q_list, H2_list, colors[cr])
        H2_dots_list.append(H2_dots)


"""Добавить на график"""


def H2_redraw2():
    F12 = float(ent1.get())
    if (abs(F12) <= N):
        global H2_list  # создание глобального списка значений Y
        global H2_dots  # объявление глобальным списка значений точек на холсте
        H2_list = H2_gr(q_list, N, F12)
        cr = random.randint(0, len(colors) - 1)
        H2_dots = graph_dot(q_list, H2_list, colors[cr])
        H2_dots_list.append(H2_dots)


"""Стереть последний график"""


def clean_draw():
    index = len(sign_dot_list) - 1
    # удаление старого графика 
    if (index >= 0):
        for dot in H2_dots_list[index]:
            canvas.delete(dot)  # у холста есть метод delete
        canvas.delete(sign_dot_list[index])
        sign_dot_list.pop(index)

        H2_dots_list.pop(index)


"""Стереть все графики"""


def clean_draw_all():
    if (len(sign_dot_list) > 0):
        for H2_dots in H2_dots_list:
            # удаление старого графика 
            for dot in H2_dots:
                canvas.delete(dot)  # у холста есть метод delete
            canvas.delete(sign_dot_list[len(sign_dot_list) - 1])
            sign_dot_list.pop(len(sign_dot_list) - 1)
        H2_dots_list.clear()


"""ДОБАВЛЕНИЕ ЭЛЕМЕНТОВ НА ПАНЕЛЬ ИНСТРУМЕНТОВ """
""" 1. Добавление меток """
lab1 = tk.Label(panel1,
                text="Скалярное произведение \n 2-х некоррелированных \n сигналов (-N <= F" + "\u2081\u2082" + " <=  N):",
                font='Times 12')
lab1.place(x=10, y=90, width=250, height=80)

lab2 = tk.Label(panel1, text="   F" + "\u2081\u2082" + " = ", font='Times 14')
lab2.place(x=50, y=230, width=30, height=50)

lab3 = tk.Label(panel2, text="Зависимость значений энтропии от ОСШ для аддитивной смеси" + "\n\
2-х детерминированных сигналов и собственного шума" +
                             "\n для различных значений скалярного произведения векторов F" + "\u2081\u2082" +
                             " \n (длина выборки процесса N = 10)",
                font='Times 14')
lab3.place(x=0, y=0, width=800, height=100)

""" 2. Добавление полей для ввода  """
ent1 = tk.Entry(panel1, bd=5, font='Times 14')
ent1.place(x=90, y=230, width=90, height=50)
ent1.insert(5, "5")

""" 3. Добавление кнопки и обработчика  """
but1 = tk.Button(panel1, text="Построить \n заново", command=H2_redraw, font='Times 14')
but1.place(x=50, y=280, width=200, height=50)

but2 = tk.Button(panel1, text="Добавить на \n  график", command=H2_redraw2, font='Times 14')
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
x_left, x_right = -1, 10
y_bottom, y_top = -1, 5
dx, dy = draw_axis(x_left, x_right, y_bottom, y_top)

"""Вычисление значений X графика параболы"""
q_step1 = 0.01  # шаг, с которым строим график  - меняем
q_list = frange(0, x_right, q_step1)
F12 = float(ent1.get())
N = 10
H2_list = H2_gr(q_list, N, F12)

# q = np.arange(0,10,0.1)


colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black', 'orange', 'purple', 'lime', 'brown']
cr = random.randint(0, len(colors) - 1)
"""Нанесение точек графика параболы  на холст """
"""в parabola - записывается список точек"""
H2_dots_list = []
global sign_dot_list  # подписи к графикам
sign_dot_list = []
H2_dots = graph_dot(q_list, H2_list, colors[cr])  # один график
H2_dots_list.append(H2_dots)  # все графики

win.mainloop()
