#!/usr/bin/env python
# coding: utf-8

# In[8]:


# import
import numpy as np
import sympy as sym
import random
import matplotlib
import matplotlib.pyplot as plt
import flet as ft
from flet.matplotlib_chart import MatplotlibChart
matplotlib.use("svg")


# In[9]:


def main(page: ft.Page):
    page.window_width = 800
    page.window_height = 700
    global y, c, h, start, end, fig, ax, method

    fig, ax = plt.subplots(figsize = (5, 4))

    def central_diff(y, h, s, e):
        # y : compute function, h : step size, s : start, e : end
        # define grid
        x = np.arange(s, e, h)
        # change the datatype of y
        y = eval(y)
        diff = np.array([]) # create a new np.array for the difference between the value of sin function: y(i-1) and y(i+1)
        for i in range(len(y)-2): # add the difference to diff iterate
            diff = np.append(diff, y[i+2]-y[i]) # add the difference to diff
        diff /= h*2 # devide 2h to get the central difference
        # compute corresponding grid
        x_diff = x[1:-1]
        return diff, x_diff
    
    def forward_diff(y, h, s, e):
        # y : compute function, h : step size, s : start, e : end
        # define grid
        x = np.arange(s, e, h)
        # change the datatype of y
        y = eval(y)
        diff = np.array([]) # create a new np.array for the difference between the value of sin function: y(i-1) and y(i+1)
        for i in range(len(y)-1): # add the difference to diff iterate
            diff = np.append(diff, y[i+1]-y[i]) # add the difference to diff
        diff /= h*2 # devide 2h to get the central difference
        # compute corresponding grid
        x_diff = x[0:-1]
        return diff, x_diff
    
    def backward_diff(y, h, s, e):
        # y : compute function, h : step size, s : start, e : end
        # define grid
        x = np.arange(s, e, h)
        # change the datatype of y
        y = eval(y)
        diff = np.array([]) # create a new np.array for the difference between the value of sin function: y(i-1) and y(i+1)
        for i in range(len(y)-1): # add the difference to diff iterate
            diff = np.append(diff, y[i+1]-y[i]) # add the difference to diff
        diff /= h*2 # devide 2h to get the central difference
        # compute corresponding grid
        x_diff = x[1:]
        return diff, x_diff

    def Riemanns_integral(y, h, s, e):
        # y : compute function, h : step size, s : start, e : end
        # define grid
        x_re = np.arange(s, e, h)
        x = []
        for i in range(len(x)-1):
            x.append((x_re[i]+x_re[i+1])/2)
        # change the datatype of y then get the value on each grid
        integral_grid = eval(y)
        x_diff = x[0:-1]
        integral = h*sum(integral_grid)
        return integral_grid, x_diff, integral

    def Trapezoid_integral(y, h, s, e):
        # y : compute function, h : step size, s : start, e : end
        # define grid
        x = np.arange(s, e, h)
        # change the datatype of y
        y = eval(y)
        # calculate the sum of each small area
        integral_grid = []
        for i in range(len(x)-1):
            integral_grid.append((y[i]+y[i+1])/2)
        x_diff = x[0:-1]
        integral = h*sum(integral_grid)
        return integral_grid, x_diff, integral
    
    def count_button_clicked(e):
        # t_test.value = f"y = {tb1.value}" # for testing
        global y, c, h, start, end, fig, ax
        y = tb1.value
        h = float(tb4.value)
        start = float(tb2.value)
        end = float(tb3.value)
        # print(y, c, h, start, end) # for testing
        if c == "differential":
            if method == "Forward":
                diff, x_diff = forward_diff(y, h, start, end)
            elif method == "Backward":
                diff, x_diff = backward_diff(y, h, start, end)
            else:
                diff, x_diff = central_diff(y, h, start, end)
            i = random.randint(0, len(x_diff)-1)
            result.value = f"The differential at x={x_diff[i]} is {diff[i]}."
            ax.plot(x_diff, diff)
            ax.set_xlim(start, end)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            x = sym.symbols('x')
            if "np" in y:
                y = y.replace("np", "sym")
            y1 = sym.symbols('y1', cls = sym.Function)
            y1 = eval(y)
            d = sym.diff(y1, x)
            ex = d.evalf(subs = {x:x_diff[i]})
            exact.value = f"The Exact differential is {ex}"
            er = 100*abs((ex-diff[i])/diff[i])
            error.value = f"The Error at x={x_diff[i]} is {round(er, 8)}%."
        elif c == "integral":
            if method == "Riemann's":
                inte_g, x_diff, inte = Riemanns_integral(y, h, start, end)
            else:
                inte_g, x_diff, inte = Trapezoid_integral(y, h, start, end)
            result.value = f"Answer: {inte}"
            ax.plot(x_diff, inte_g)
            ax.set_xlim(start, end)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            x = sym.symbols('x')
            if "np" in y:
                y = y.replace("np", "sym")
            y1 = sym.symbols('y1', cls = sym.Function)
            y1 = eval(y)
            ex = sym.integrate(y1, (x, start, end)).evalf()
            exact.value = f"The Exact integral is {ex}"
            er = 100*abs((ex-inte)/inte)
            error.value = f"The Error is {round(er, 8)}%."
        page.update()
    t_function = ft.Text("y = ")
    tb1 = ft.TextField(expand=True)
    b = ft.ElevatedButton(text="Count", on_click=count_button_clicked)
    # t_test = ft.Text() # for testing
    func = ft.Row(
        controls=[
            t_function,
            tb1,
            b,
            # new_task,
            # ft.FloatingActionButton(icon=ft.icons.ADD, on_click=add_clicked),
        ],
    )

    def radiogroup_changed(e):
        # t_test1.value = f"You choice is {e.control.value}." # for testing
        global c
        c = e.control.value
        page.update()
    t_choice = ft.Text("Select your chioce:")
    cg = ft.RadioGroup(
        content=ft.Column([
        ft.Radio(value="differential", label="Differential"),
        ft.Radio(value="integral", label="Integral")], width=300), on_change=radiogroup_changed)
    # t_test1 = ft.Text() # for testing
    ch = ft.Column(
        controls=[
            t_choice,
            cg,
        ],
    )
    
    tb2 = ft.TextField(label="Start", width=150)
    t_to = ft.Text(" ~ ")
    tb3 = ft.TextField(label="End", width=150)
    tb4 = ft.TextField(label="Step Size", width=170)
    def dropdown_changed(e):
        global method
        method = f"Dropdown changed to {dd.value}"
        page.update()
    dd = ft.Dropdown(
        on_change=dropdown_changed,
        label="Method",
        options=[
            ft.dropdown.Option("Forward"),
            ft.dropdown.Option("Center"),
            ft.dropdown.Option("Bacward"),
            ft.dropdown.Option("Riemann's"),
            ft.dropdown.Option("Trapezoid"),
        ],
        width=170,
    )
    att = ft.Column(
        controls=[
            ft.Row(
                controls=[
                    tb2,
                    t_to,
                    tb3,
                ],
            ),
            ft.Row(
                controls=[
                    tb4,
                    dd,
                ],
            ),
        ]
    )

    def clear_button_clicked(e):
        plt.cla()
        result.value=""
        exact.value=""
        error.value=""
        page.update()
    b1 = ft.ElevatedButton(text="Clear", on_click=clear_button_clicked)
    result = ft.Text()
    exact = ft.Text()
    error = ft.Text()
    chart = ft.Row(
        controls=[
            MatplotlibChart(fig, original_size=True),
            ft.Column(
                controls=[
                    result,
                    exact,
                    error,
                    b1,
                ]
            )
        
        ]
    )

    view = ft.Column(
        width=700,
        controls=[
            func,
            # t_test, # for testing
            ft.Row(
                controls=[
                    ch,
                    att,
                ],
            ),
            # t_test1, # for testing
            chart,
        ],
    )
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.add(view)
ft.app(target=main)

