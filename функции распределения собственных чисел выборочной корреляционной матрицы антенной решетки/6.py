# рис. 6
# вводим значение СКО случайной фазы  ( > = 0 )
import tkinter as tk
import math  # для pi, ln()
import numpy as np
import random
from numpy import pi, exp
from numpy import linalg  # для собственные значения матрицы
from multiprocessing import Process, Manager, Value, Array
import multiprocessing as mp

M = 50  # кол-во усреднений для оценки КМ-цы
N = 10  # длина выборки (число компонент сигнала)
f = 0  # частота сигнала
Num = 10  # кол-во процессов
q_step1 = 0.01  # шаг, с которым строим график  - меняем

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
win.title("График 6")
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


# Энтропия: детерминированный сигнал со случайной фазой + собственный шум
# fi - это СКО фазы
def H1(A, f, fi):
    R = np.zeros((M * N, N, N), dtype=np.complex128)  # Корреляционная матрица пришедшего сигнала
    for i in range(N * M):
        S0 = np.array([])
        Noise = rnorm(N, 0, 1)
        fi0 = rnorm_1(N, 0, fi)
        for k in range(N):
            S0 = np.append(S0, Noise[k] + A * exp(1j * (2 * math.pi * f * k / N + fi0[k])))
        # транспонированная матрица
        S = np.array([S0]).T
        # комплексно-сопряженная матрица
        S_h = np.conj(S0)
        R[i] = S * S_h
    RRR = np.zeros((N, N), dtype=np.complex128)
    for i in range(N * M):
        RRR = RRR + R[i]
    RRR = RRR / (N * M)
    # собственные значения матрицы RRR
    L = linalg.eig(RRR)[0]
    EV = L
    sum_EV = 0
    for i in range(N): sum_EV += abs(EV[i])
    AN = EV / sum_EV
    E = 0
    for i in range(N): E += - abs(AN[i]) * math.log(abs(AN[i]))
    return E


def H1_gr(A1, f, fi, array, index):
    gr1 = [0] * len(A1)
    for i in range(len(A1)):
        gr1[i] = H1(A1[i], f, fi)
    array[index] = gr1

def H1_gr2(A,f, fi):
    gr=[]
    for i in range(len(A)):
        gr.append(H1(A[i], f, fi))
    return gr

def H1_gr4(A, f, fi):
    Num = 10  # кол-во процессов
    process = []
    manager = Manager()
    arr = manager.list([0] * Num)
    for i in range(Num):
        A1 = A[int(i * len(A) / Num): int((i + 1) * len(A) / Num)]
        # arr - это  адрес массива в памяти
        pr = mp.Process(target=H1_gr, args=(A1, f,fi, arr, i,))
        process.append(pr)
        pr.start()
    for i in process:
        i.join()
    gr = sum(arr, [])  # список списков преобразовать в 1 список
    return gr


"""
def H1_gr(A1,f, fi, array, index):
    gr1=[0]*len (A1)
    for i in range(len(A1)): 
        gr1[i] = H1(A1[i], f, fi)
    array[index] = gr1
  """

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

    """График sigma = 0 """
    """
    H1_list= H1_gr(q_list,f, 0)
    i=0
    for x in q_list:
            #пересчет координат в пиксели холста
            x_c = (x-x_left)*dx 
            y_c = (y_top - H1_list[i])*dy
            dot = canvas.create_oval(x_c-1, (y_c-1), x_c+1, (y_c+1), fill="red", outline="red", width =2)
            i+=1
    cx = -x_left * dx;     x_t = cx+ x_right *dx - 50;   y_t = (y_top - min(H1_list))*dy - 50
    sign = canvas.create_text(x_t, y_t, text="sigma = 0", fill= "red", font=("Helvetica 12 bold"))
    """
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
    sigma = float(ent1.get())
    text = "sigma = pi / " + str(float(sigma))
    cx = -x_left * dx;
    x_t = cx + x_right * dx - 50;
    y_t = 20 * (len(sign_dot_list) + 1)
    sign = canvas.create_text(x_t, y_t, text=text, fill=color, font=("Helvetica 12 bold"))
    sign_dot_list.append(sign)
    return dot_list


# возвращает Массив случайных комплексных чисел <class 'numpy.ndarray'>
def rnorm(N, mu, sigma):
    arr = np.array([])
    for i in range(N):
        a = random.gauss(mu, sigma)
        b = random.gauss(mu, sigma)
        arr = np.append(arr, complex(a, b))
    return arr


# возвращает Массив случайных действительных чисел <class 'numpy.ndarray'>
def rnorm_1(N, mu, sigma):
    arr = np.array([])
    for i in range(N):
        a = random.gauss(mu, sigma)
        arr = np.append(arr, a)
    return arr


"""Обработчик кнопки "Отобразить энтропию"""
"""Построить график заново"""


def H1_redraw():
    sigma = math.pi / float(ent1.get())
    print("sigma= ", sigma)
    if (sigma > 0):
        global H1_list  # создание глобального списка значений Y
        global H1_dots  # объявление глобальным списка значений точек на холсте

        # удаление старого графика
        if (len(H1_dots_list) > 0):
            for dot in H1_dots:
                canvas.delete(dot)  # у холста есть метод delete
            H1_dots_list.pop(len(H1_dots_list) - 1)
        if (len(sign_dot_list) > 0):
            canvas.delete(sign_dot_list[len(sign_dot_list) - 1])
            sign_dot_list.pop(len(sign_dot_list) - 1)
            
        H1_list = H1_gr4(q_list, f, sigma)
        
        cr = random.randint(0, len(colors) - 1)
        H1_dots = graph_dot(q_list, H1_list, colors[cr])
        H1_dots_list.append(H1_dots)


"""Добавить на график"""


def H1_redraw2():
    sigma = math.pi / float(ent1.get())
    if (sigma > 0):
        global H1_list  # создание глобального списка значений Y
        global H1_dots  # объявление глобальным списка значений точек на холсте
        H1_list = H1_gr4(q_list, f, sigma)
        cr = random.randint(0, len(colors) - 1)
        H1_dots = graph_dot(q_list, H1_list, colors[cr])
        H1_dots_list.append(H1_dots)


"""Стереть последний график"""


def clean_draw():
    index = len(sign_dot_list) - 1
    # удаление старого графика
    if (index >= 0):
        for dot in H1_dots_list[index]:
            canvas.delete(dot)  # у холста есть метод delete
        canvas.delete(sign_dot_list[index])
        sign_dot_list.pop(index)

        H1_dots_list.pop(index)


"""Стереть все графики"""


def clean_draw_all():
    if (len(sign_dot_list) > 0):
        for H1_dots in H1_dots_list:
            # удаление старого графика
            for dot in H1_dots:
                canvas.delete(dot)  # у холста есть метод delete
            canvas.delete(sign_dot_list[len(sign_dot_list) - 1])
            sign_dot_list.pop(len(sign_dot_list) - 1)
        H1_dots_list.clear()


"""ДОБАВЛЕНИЕ ЭЛЕМЕНТОВ НА ПАНЕЛЬ ИНСТРУМЕНТОВ """
""" 1. Добавление меток """
lab1 = tk.Label(panel1, text="Значение СКО \n случайной фазы:", font='Times 12')
lab1.place(x=10, y=90, width=250, height=80)

lab2 = tk.Label(panel1, text=" СКО = pi / ", font='Times 14')
lab2.place(x=50, y=230, width=110, height=50)

lab3 = tk.Label(panel2, text="Зависимость значений энтропии от ОСШ для аддитивной смеси\n\
детерминированного синусоидального сигнала \n и собственного шума для различных значений СКО\n случайной фазы: ",
                font='Times 14')
lab3.place(x=0, y=0, width=800, height=100)

""" 2. Добавление полей для ввода  """
ent1 = tk.Entry(panel1, bd=5, font='Times 14')
ent1.place(x=170, y=230, width=90, height=50)
ent1.insert(5, "30")

""" 3. Добавление кнопки и обработчика  """
but1 = tk.Button(panel1, text="Построить \n заново", command=H1_redraw, font='Times 14')
but1.place(x=50, y=280, width=200, height=50)

but2 = tk.Button(panel1, text="Добавить на \n  график", command=H1_redraw2, font='Times 14')
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
y_bottom, y_top = -1, 3
dx, dy = draw_axis(x_left, x_right, y_bottom, y_top)

"""Вычисление значений X графика """

q_list = frange(0, x_right, q_step1)
sigma = 0

if __name__ == '__main__':
    colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black', 'orange', 'purple', 'lime', 'brown']
    """Нанесение точек графика   на холст """

    H1_dots_list = []
    global sign_dot_list;
    sign_dot_list = []  # подписи к графикам

    """
    H1_list= H1_gr(q_list,f, sigma)
    H1_dots = graph_dot(q_list, H1_list, colors[cr])  #  один график 
    H1_dots_list.append(H1_dots)  # все графики
    """

    """График sigma = 0 """

    H1_list = H1_gr4(q_list, f, 0)  # оптимизировать по q_list
    i = 0
    for x in q_list:
        # пересчет координат в пиксели холста
        x_c = (x - x_left) * dx
        y_c = (y_top - H1_list[i]) * dy
        dot = canvas.create_oval(x_c - 1, (y_c - 1), x_c + 1, (y_c + 1), fill="red", outline="red", width=2)
        i += 1
    cx = -x_left * dx;
    x_t = cx + x_right * dx - 50;
    y_t = (y_top - min(H1_list)) * dy + 10
    canvas.create_text(x_t, y_t, text="sigma = 0", fill="red", font=("Helvetica 11 bold"))

    win.mainloop()
