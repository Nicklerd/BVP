import tkinter as tk
from tkinter import ttk
from sympy import parse_expr, Symbol, diff, evalf
import numpy as np
from scipy.integrate import solve_ivp, odeint, trapezoid
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import sys
import os
from PIL import ImageTk, Image

n = -1
dRx = []
dRy = []
dfx = []
x_cur = [[]]
x_cur1 = 0
x_cur2 = 0
num_points = 50
x_res = []
t1 = []
a = 0
b = 0
t0 = 0
method_in = ""
method_out = ""
#dRdxdy: Эта функция вычисляет частные производные матриц dR/dx и dR/dy для заданного вектора p, интервалов a и b и времени t0.

def init():
    global dRx, dRy, dfx, x_cur, x_cur1, x_cur2, x_res, t1,n, a, b, t0, method_in, method_out,res
    dRx = [] #Производная R по xi(a), i=1..n
    dRy = [] #Производная R по xi(b), i=1..n
    dfx = [] #Производная f по xi, i=1..n
    x_cur = [[]]
    x_cur1 = 0
    n = -1
    x_cur2 = 0
    x_res = []
    t1 = []
    a = 0
    b = 0
    t0 = 0
    method_in = ""
    method_out = ""
    res = None

def dRdxdy(p, a, b, t0):
    x_a_p = x_cur1.sol(a)
    x_b_p = x_cur2.sol(b)
    dRdx = np.zeros((n, n))
    dRdy = np.zeros((n, n)) 
    xa = []
    xb = []
    for i in range(n):
        xa.append(Symbol("xa"+str(i + 1)))
        xb.append(Symbol("xb" + str(i + 1)))
    for i in range(n):
        for k in range(n):
            tmp_fun = dRx[k][i]
            for j in range(n):
                tmp_fun = tmp_fun.subs(xa[j], x_a_p[j])
                tmp_fun = tmp_fun.subs(xb[j], x_b_p[j])
            dRdx[k][i] = tmp_fun.evalf()
    for i in range(n):
        for k in range(n):
            tmp_fun = dRy[k][i]
            for j in range(n):
                tmp_fun = tmp_fun.subs(xa[j], x_a_p[j])
                tmp_fun = tmp_fun.subs(xb[j], x_b_p[j])
            dRdy[k][i] = tmp_fun.evalf()
    return [dRdx, dRdy]


# Xrhs: Эта функция представляет правую часть системы дифференциальных уравнений в виде функции для использования в функции solve_ivp. Она 

def Xrhs(tt, y):
    t = Symbol("t")
    x = []
    if tt >= t0:
        x_c = x_cur2.sol(tt)
    else:
        x_c = x_cur1.sol(tt)
    for i in range(1, 7):
        x.append(Symbol("x" + str(i)))
    dfdxp = np.zeros((n, n))
    for i in range(n):
        for k in range(n):
            tmp_fun = dfx[k][i]
            tmp_fun.subs(t, tt)
            for j in range(n):
                tmp_fun = tmp_fun.subs(x[j], x_c[j])
            dfdxp[k][i] = tmp_fun.evalf()
    return np.matmul(dfdxp, np.array(y))

#X: Эта функция решает систему дифференциальных уравнений с использованием функции solve_ivp для интервала от a до b. Возвращает матрицу решений res.

def X(p, a, b):
    res = []
    t = np.linspace(a, b, num=5)
    for i in range(n):
        init = np.zeros(n)
        init[i] = 1
        res.append(solve_ivp(Xrhs, [a, b], init, method=method_in).y[:,-1])
    return np.array(res).transpose()


#dfidmu: Эта функция вычисляет матрицу dF/dmu для заданного вектора p, интервалов a и b и времени t0. Использует функции dRdxdy и X для вычисления необходимых производных и матриц.

def dfidmu(p, a, b, t0):
    global x_cur1, x_cur2
    x_cur1 = solve_ivp(int_task, [t0, a], p, method=method_in, dense_output=True)
    x_cur2 = solve_ivp(int_task, [t0, b], p, method=method_in, dense_output=True)


    R = dRdxdy(p, a, b, t0)
    Xa = X(p, t0, a)
    Xb = X(p, t0, b)
    res = np.matmul(R[0], Xa) + np.matmul(R[1], Xb)
    return res


#dfi_inv: Эта функция представляет правую часть уравнения F(p) = 0 для использования в функции solve_ivp.
#  Вычисляет обратную матрицу f_inv и возвращает -f_inv * f0, где f0 - начальное приближение для F(p) = 0.

def dfi_inv(mu, p, a, b, t0, f0):
    if (mu == 1):
        print(p, " ", mu)
    if pb['value'] < mu * 100:
        pb['value'] = mu * 100
    root.update()
    f_inv = np.linalg.inv(dfidmu(p, a, b, t0))
    return -np.matmul(f_inv, f0)


#define_diff: Эта функция определяет частные производные для системы дифференциальных уравнений и сохраняет их в списках dRx, dRy и dfx.

def define_diff():
    xa = []
    xb = []
    x = []
    for i in range(n):
        xa.append(Symbol("xa"+str(i + 1)))
        xb.append(Symbol("xb" + str(i + 1)))
        x.append(Symbol("x" + str(i + 1)))
    for i in range(n):
        tmp1 = []
        tmp2 = []
        tmp3 = []
        for k in range(n):
            tmp1.append(diff(edge[i], xa[k]))
            tmp2.append(diff(edge[i], xb[k]))
            tmp3.append(diff(func[i],x[k]))
        dRx.append(tmp1)
        dRy.append(tmp2)
        dfx.append(tmp3)

#fi0: Эта функция вычисляет вектор начальных условий F(p0) = 0 для заданного начального приближения p0 и времени t0.

def fi0(p0, a, b, t0):
    res_ode = solve_ivp(int_task, [t0, a], p0, method=method_in).y[:,-1]
    x_a_p0 = res_ode
    res_ode = solve_ivp(int_task, [t0, b], p0, method=method_in).y[:,-1]
    x_b_p0 = res_ode

    xa = []
    xb = []
    for i in range(1, 9):
        xa.append(Symbol("xa" + str(i)))
        xb.append(Symbol("xb" + str(i)))
    res = np.zeros(n)
    for i in range(n):
        tmp_func = edge[i]
        for j in range(n):
            tmp_func = tmp_func.subs(xa[j], x_a_p0[j])
            tmp_func = tmp_func.subs(xb[j], x_b_p0[j])
        res[i] = tmp_func.evalf()
    return res

#int_task: Эта функция представляет систему дифференциальных уравнений в виде функции для использования в функции odeint. Вычисляет значения уравнений для заданного времени t и вектора y.

def int_task(t, y):
    tt = Symbol("t")
    x = []
    for i in range(1, 9):
        x.append(Symbol("x" + str(i)))
    global n
    res = np.zeros(n)
    for i in range(n):
        tmp_func = func[i].subs(tt, t)
        for j in range(n):
            tmp_func = tmp_func.subs(x[j],y[j])
        res[i] = tmp_func.evalf()
    return res


#solve: Эта функция вызывается при нажатии кнопки "Решить". Считывает значения из текстовых полей, вызывает функции для решения задачи и сохраняет результаты в переменных t1 и x_res.

def solve():
    global method_in, method_out
    root.update()
    global a, b, t0
    global x_res
    global t1
    global func, edge
    global res
    init()
    func = []
    edge = []
    button_draw["state"] = tk.DISABLED
    root.update()
    s = init_a.get("1.0", "end-1c")
    s = s.replace('PI','3.1415926535')
    s = s.replace('e', '2.7182818284')
    a = eval(s)
    s = init_b.get("1.0", "end-1c")
    s = s.replace('PI','3.1415926535')
    s = s.replace('e', '2.7182818284')
    b = eval(s)
    for i in range(1, 9):
        input = texts_diff[i - 1].get("1.0", "end-1c")
        input = input.replace('^', '**')
        input = input.replace("PI", '3.1415926535')
        input = input.replace("e", '2.7182818284')
        if input != "":
            func.append(parse_expr(input, evaluate=True))
    for i in range(8):
        input = input = texts_edge[i].get("1.0", "end-1c")
        input = input.replace("PI", '3.1415926535')
        input = input.replace("e", '2.7182818284')
        input = input.replace('x1(a)','xa1')
        input = input.replace('x2(a)','xa2')
        input = input.replace('x3(a)','xa3')
        input = input.replace('x4(a)','xa4')
        input = input.replace('x5(a)','xa5')
        input = input.replace('x6(a)','xa6')
        input = input.replace('x7(a)','xa7')
        input = input.replace('x8(a)','xa8')

        input = input.replace('x1(b)','xb1')
        input = input.replace('x2(b)','xb2')
        input = input.replace('x3(b)','xb3')
        input = input.replace('x4(b)','xb4')
        input = input.replace('x5(b)','xb5')
        input = input.replace('x6(b)','xb6')
        input = input.replace('x7(b)','xb7')
        input = input.replace('x8(b)','xb8')
        
        if input != "":
            edge.append(parse_expr(input, evaluate=True))
    global n
    n = len(func)
    s = init_value.get("1.0", "end-1c")
    s = s.replace('PI','3.1415926535')
    s = s.replace('e', '2.7182818284')
    p0 = np.array(eval(s))
    s = s = init_time.get("1.0", "end-1c")
    s = s.replace('PI','3.1415926535')
    s = s.replace('e', '2.7182818284')
    t0 = eval(s)
    method_in = mth_in.get()
    method_out = mth_out.get()
    
    define_diff()

    f0 = fi0(p0, a, b, t0)
    rhs = lambda tt, y: dfi_inv(tt, y, a, b, t0, f0)
    p_res = solve_ivp(rhs, [0,1], p0, method=method_out).y[:,-1]
    t1 = np.linspace(t0, a, num=50)
    x_res = odeint(int_task, p_res, t1, tfirst=True)
    x_res = np.flip(x_res, 0)
    t = np.linspace(t0, b, num=50)
    x_res = np.append(x_res, odeint(int_task, p_res, t, tfirst=True), axis=0)
    t1 = np.flip(t1)
    t1 = np.append(t1, t)
    print(t1)
    print(x_res)
    button_draw["state"] = tk.NORMAL
    root.update()
    pb.stop()


#solve_ode: Эта функция отображает график решения системы дифференциальных уравнений.


def solve_ode(x1t, x2t):

    plot = tk.Tk()
    plot.title("График")
    fig = Figure(figsize=(5, 5), dpi=100)
    plot1 = fig.add_subplot(111)
    plot1.set_xlabel(x1t)
    plot1.set_ylabel(x2t)

    conv = {
        "t" : t1
    }
    for i in range(n):
        conv["x" + str(i + 1)] = x_res[:, i]

    x1 = conv[x1t]
    x2 = conv[x2t]
    plot1.plot(x1, x2)
    plot1.grid()
    canvas = FigureCanvasTkAgg(fig, master=plot)
    canvas.draw()

    canvas.get_tk_widget().pack()

    toolbar = NavigationToolbar2Tk(canvas, plot)
    toolbar.update()

    canvas.get_tk_widget().pack()

    plot.mainloop()


def draw():
    global t1, x_res
    solve_ode(dr1.get(), dr2.get())


def integral():
    f = f_text.get("1.0", "end-1c")
    f = f.replace("PI", '3.1415926535')
    f = f.replace("e", '2.7182818284')
    f = parse_expr(f)
    x = []
    for i in range(1, 9):
        x.append(Symbol("x" + str(i)))
    resf = []
    for i in range(len(t1)):
        f_tmp = f
        for j in range(n):
            f_tmp = f_tmp.subs(x[j], x_res[i][j])
        resf.append(f_tmp.evalf())
    ans = trapezoid(resf, x=t1)
    ans = round(ans,4)
    ans_label.config(text=str(ans))

#draw: Эта функция вызывается при нажатии кнопки "Построить график". Она вызывает функцию solve_ode и передает ей значения t1 и x_res.


#Остальные строки кода отвечают за создание графического интерфейса пользователя с использованием библиотек tkinter и matplotlib.
texts_diff = []

c = ' '
spaces = c * 80

root = tk.Tk()
root.title('Решение краевой задачи методом продолжения по параметру')

frame = tk.Frame(master=root, relief="groove", borderwidth=0.5)
frame.pack(anchor="nw",side="left")

frame1 = tk.LabelFrame(master=frame,text='Введите систему дифференциальных уравнений', width=300,height=400,relief="solid", borderwidth=0.5)
frame1.pack(anchor="nw",pady=40, padx=40)
for i in range(1, 9):
    fr1 = tk.Frame(frame1)
    fr1.pack(anchor='nw')
    texts_diff.append(tk.Text(fr1, width=30, height=1, font=("Times", 15)))
    tk.Label(fr1, text="dx" + str(i) + "/dt = ").pack(side="left", expand=True)
    texts_diff[i - 1].pack(side="left", expand=True)


frame2 = tk.LabelFrame(master=frame,text="Введите краевые условия", width=300,height=400,relief="solid",borderwidth=0.5)
frame2.pack(anchor="nw",pady=40, padx=40)

texts_edge = []

for i in range(1, 9):
    fr1 = tk.Frame(frame2)
    fr1.pack(anchor='nw',ipadx=60)
    texts_edge.append(tk.Text(fr1, width=30, height=1, font=("Times", 15)))
    tk.Label(fr1, text=str(i)+':', font=("Times", 15)).pack(side="left", expand=True)
    texts_edge[i - 1].pack(side="left", fill="x", expand=True)
    tk.Label(fr1, text=" = 0", font=("Times", 15)).pack(side="left", expand=True)



###

frame_func = tk.Frame(master=frame, relief="groove", width=1000,height=400, borderwidth=0.5)
frame_func.pack(anchor="nw",side="top")

frame5 = tk.LabelFrame(master=frame_func,text='Функционал', relief='solid', borderwidth=0.5)
frame5.pack(side='top',padx=40,pady=66)

ff_1 = tk.Frame(frame5)
ff_1.pack(anchor='nw')
tk.Button(ff_1, text="Посчитать", command=integral, width=15).pack(side='bottom', fill=tk.BOTH, pady=5)

ff_2 = tk.Frame(frame5)
fr1.pack(anchor='nw')

ff_3 = tk.Frame(frame5)
ff_3.pack(anchor='n')
tk.Label(ff_3,text="J = ",font=("Times", 15)).pack(side='left')
f_text = tk.Text(master=ff_3, width=20, height=1,font=("Arial", 15))
f_text.pack(side='left')
tk.Label(ff_3, text=' = ',font=("Times", 15)).pack(side='left')
ans_label = tk.Label(ff_3, text="   ")
ans_label.pack(side='left')

###

framer = tk.Frame(master=root, relief="groove", borderwidth=0.5)
framer.pack(anchor="nw",side="top")




frame3 = tk.LabelFrame(master=framer,text='Настройка метода', relief='solid', borderwidth=0.5)
frame3.pack(anchor='nw',side='top',padx=40,pady=40)


fr=tk.Frame(frame3)
fr.pack(anchor='nw')
tk.Label(fr, text='Левая граница времени'+c*38).pack(side='left', expand=True, pady=5)
init_a = tk.Text(fr, width=8,height=1, font=("Times",15))
tk.Label(fr, text=" = a", font=("Times", 15)).pack(side="right", expand=True)
init_a.pack(side='left', pady=5)

fr=tk.Frame(frame3)
fr.pack(anchor='nw')
tk.Label(fr, text='Правая граница времени'+c*35).pack(side='left', expand=True, pady=5)
init_b = tk.Text(fr, width=8,height=1, font=("Times",15))
tk.Label(fr, text=" = b", font=("Times", 15)).pack(side="right", expand=True)
init_b.pack(side='left', pady=5)

fr = tk.Frame(frame3)
fr.pack(anchor='nw')
tk.Label(fr, text="Вектор начального приближения:"+c*12).pack(side="left", expand=True, pady=5)
fr = tk.Frame(frame3)
fr.pack(anchor='center')
init_value = tk.Text(fr, width=30, height=1, font=("Times", 15))
init_value.pack(side="left", anchor='n',expand=True, pady=5)

fr=tk.Frame(frame3)
fr.pack(anchor='nw')
tk.Label(fr, text='Момент времени t*'+c*53).pack(side='left', expand=True, pady=5)
init_time = tk.Text(fr, width=8,height=1, font=("Times",15))
init_time.pack(side='left', pady=5)

fr=tk.Frame(frame3)
fr.pack(anchor='nw')
tk.Label(fr, text='Метод решения внутренней задачи'+c*17).pack(side='left', expand=True, pady=5)
mth_in = ttk.Combobox(
    fr,
    width = 8,
    state="readonly",
    values=["RK23","RK45" , "DOP853", "Radau", "BDF", "LSODA"]
)
mth_in.current(0)
mth_in.pack(side='left', pady=5)

fr=tk.Frame(frame3)
fr.pack(anchor='nw')
tk.Label(fr, text='Метод решения внешней задачи'+c*22).pack(side='left', expand=True, pady=5)
mth_out = ttk.Combobox(
    fr,
    width = 8,
    state="readonly",
    values=["RK23","RK45" , "DOP853", "Radau", "BDF", "LSODA"]
)
mth_out.current(0)
mth_out.pack(side='left', pady=5)


frame4 = tk.LabelFrame(master=framer,text='Решение', relief='solid', borderwidth=0.5, width=300,height=700)
frame4.pack(side='top',padx=40,pady=66)


fr = tk.Frame(frame4)
fr.pack(anchor='center')
tk.Button(fr, text="Решить", command=solve,width=15).pack(side='left', fill=tk.BOTH, pady=5)

fr1 = tk.Frame(frame4)
fr1.pack(anchor='n')

pb = ttk.Progressbar(fr1, orient="horizontal", length=115, value=0, mode="determinate")
pb.pack(pady=5, side='left')

fr1 = tk.Frame(frame4)
fr1.pack(anchor='nw')
tk.Label(fr1, text="Оси графика:", font=('Times', 15)).pack(side="left", expand=True, pady=5, padx=15)

dr1 = ttk.Combobox(
    fr1,
    width = 5,
    state="readonly",
    values=["t", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8"]
)
dr2 = ttk.Combobox(
    fr1,
    width = 5,
    state="readonly",
    values=["t", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8"]
)
dr1.pack(pady=5, side='left')
dr2.pack(pady=5, side='left')

fr1 = tk.Frame(frame4)
fr1.pack(anchor='n')
button_draw = tk.Button(fr1, text="Построить график", command=draw ,state=tk.DISABLED)
button_draw.pack(side=tk.LEFT, fill=tk.BOTH, pady=5, padx=15)

menubar = tk.Menu(root)
about = tk.Menu(menubar, tearoff=False)
help = tk.Menu(menubar, tearoff=False)
examples = tk.Menu(menubar, tearoff=False)
restart = tk.Menu(menubar,tearoff=False)


s=['При введении задачи необходимо\n',
   'Придерживаться правил:\n\n',
   '1) Переменные задаются в виде "xn".\n',
   '2) Аргументы функций заключаются в\n',
   'скобки: sin(x2).\n',
   '3) Краевые условия вводятся в виде:\n',
   '"xn(a), a - граница."\n',
   '4) Степень необходимо заключать в\n',
   'скобки: (x1**2+x2**2)**(-3/2).\n',
   '5) Знак умножения обязателен к прописыванию:\n',
   'Пример правильного ввода x1*x2\n',
   '6) Функционал  J - интегрального вида\n',
]

def Help():
    newWindow  = tk.Toplevel(root)
    newWindow.title("Помощь")
    newWindow.geometry("400x400")
    text = tk.Text(newWindow,font=('Arial', 15))
    text.pack(side='left', anchor='nw',pady=5)
    text.insert("1.0", s[0]+s[1]+s[2]+s[3]+s[4]+s[5]+s[6]+s[7]+s[8]+s[9]+s[10]+s[11])

def about_prog():
    newWindow  = tk.Toplevel(root)
    newWindow.title("О программе")
    newWindow.geometry("400x600")
    image = Image.open("photo.jpg")
    image = image.resize((150,200))
    img = ImageTk.PhotoImage(image, master=newWindow)
    l = tk.Label(newWindow,image=img, anchor='center')
    l.pack(fill="both",expand=True) 
    text = tk.Text(newWindow, height=12,font=("Arial", 15))
    text.pack(side='top')
    about_me = "Программа для решения краевой задачи\n"+"методом продолжения по параметру.\n"+ \
        "\nПреподаватели: Аввакумов С.Н, \nКиселев Ю.Н.,"+"Дряженков А.А.\n"+\
            "\nАвтор: Иванов Николай Сергеевич,\nстудент кафедры ОУ.\n"\
                "\n""2023г.".center(6)
    text.insert("1.0",about_me)
    newWindow.mainloop()


def clear():
    for i in range(8):
        texts_diff[i].delete("1.0", tk.END)
        texts_edge[i].delete("1.0", tk.END)
    init_a.delete("1.0", tk.END)
    init_b.delete("1.0", tk.END)
    init_time.delete("1.0", tk.END)
    init_value.delete("1.0", tk.END)


def two_body():
    clear()
    texts_diff[0].insert("1.0", "x3")
    texts_diff[1].insert("1.0", "x4")
    texts_diff[2].insert("1.0", "-x1*(x1**2+x2**2)**(-3/2)")
    texts_diff[3].insert("1.0", "-x2*(x1**2+x2**2)**(-3/2)")

    texts_edge[0].insert("1.0", "x1(a)-2")
    texts_edge[1].insert("1.0", "x2(a)")
    texts_edge[2].insert("1.0", "x1(b)-1.0738644361")
    texts_edge[3].insert("1.0", "x2(b)+1.0995343576")

    init_a.insert("1.0", "0")
    init_b.insert("1.0", "7")
    init_time.insert("1.0", "0")

    init_value.insert("1.0", "[2, 0, 0.5, -0.5]")


def cycEq():
    clear()

    texts_diff[0].insert("1.0", "x3*x2")
    texts_diff[1].insert("1.0", "x3*(-x1+sin(x2))")
    texts_diff[2].insert("1.0", "0")
    texts_diff[3].insert("1.0", "0")

    texts_edge[0].insert("1.0", "x1(a)-x4(a)")
    texts_edge[1].insert("1.0", "x1(b)-x4(b)")
    texts_edge[2].insert("1.0", "x2(a)")
    texts_edge[3].insert("1.0", "x2(b)")
    init_a.insert("1.0", "0")
    init_b.insert("1.0", '1')
    init_time.insert("1.0", "0")

    init_value.insert("1.0", "[2, 0, 2*3.1415, 2]")
    

def energy():
    clear()

    texts_diff[0].insert("1.0", "x2")
    texts_diff[1].insert("1.0", "x3")
    texts_diff[2].insert("1.0", "((0.0000000001+(x6+1)**2)**(1/2)-(0.0000000001+(x6-1)**2)**(1/2))/2")
    texts_diff[3].insert("1.0", "0")
    texts_diff[4].insert("1.0", "-x4")
    texts_diff[5].insert("1.0", "-x5")

    texts_edge[0].insert("1.0", "x1(a)-1")
    texts_edge[1].insert("1.0", "x2(a)")
    texts_edge[2].insert("1.0", "x3(a)")
    texts_edge[3].insert("1.0", "x1(b)")
    texts_edge[4].insert("1.0", "x2(b)")
    texts_edge[5].insert("1.0", "x3(b)")

    init_a.insert("1.0", '0')
    init_b.insert("1.0", "3.275")
    init_time.insert("1.0","3.275")
    init_value.insert("1.0","[0, 0, 0, -2.9850435834, 4.8880088678, -2.9083874537]")

def clear_all():
    clear()


#def restart():
#    root.destroy#   os.startfile("C:/Users/User/Desktop/bvp.py")

examples = tk.Menu(menubar, tearoff=0)
examples.add_command(label="Краевая задача двух тел", command=two_body)
examples.add_command(label="Циклы Эквейлера", command=cycEq)
examples.add_command(label="Функционал типа энергия", command=energy)
menubar.add_cascade(label = 'Примеры', menu=examples)

menubar.add_command(label="Сброс", command=clear_all)
menubar.add_cascade(label='Помощь', command=Help)
menubar.add_cascade(label='О программе', command=about_prog)

root.config(menu=menubar)

root.geometry("970x1000")
root.mainloop()