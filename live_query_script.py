import time
import numpy as np
from Config import *
from Modules.BinanceOhlcHandler import BinanceOhlcHandler
import Modules.NeuralNetwork_2_hidden as nn_2_hidden
from Modules.services import MyLogger
from datetime import datetime, timedelta
import Modules.Teacher as Teacher
import tkinter as tk
import threading
import sys
import matplotlib.pyplot as plt
import Modules.services as services
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.backends.backend_tkagg as tkagg
import matplotlib.dates as mdates
from matplotlib import style
matplotlib.use("TkAgg")
style.use('seaborn-pastel')


def get_next_exec_wait():
    now = datetime.now()
    next_ex = now - timedelta(minutes=now.minute % 5 - 5, seconds=now.second, microseconds=now.microsecond)
    diff = next_ex - now
    millis = diff.days * 24 * 60 * 60 * 1000
    millis += diff.seconds * 1000
    millis += diff.microseconds / 1000
    seconds = millis / 1000 - 1

    return seconds, next_ex


def predict_live():

    global start
    global end
    global ans_buy_list
    global ans_sell_list
    global live_dataset_handler_binance

    # 287 is 24 hours - some samples and the end, in term of time to get ready for next execution at midnight
    predict_time = 285

    output_file = open('Outputs/live_log.txt', 'w')

    live_dataset_handler_binance = BinanceOhlcHandler(BINANCE_PAIR)
    live_dataset_handler_binance.get_dataset(24, INTERVAL)
    live_dataset_handler_binance.dataset.drop(live_dataset_handler_binance.dataset.tail(1).index, inplace=True)
    main_normalizer = services.Normalize(live_dataset_handler_binance.dataset)

    live_dataset_handler_binance.print_to_file('Outputs/live_dataset.txt')

    nn = nn_2_hidden.NeuralNetwork(INPUT_SIZE, HIDDEN_LAYER_1, HIDDEN_LAYER_2, OUTPUT_SIZE, LEARNING_RATE)
    nn.load_from_file()

    MyLogger.write("Now going live, lets see! Click Start for start :-)", output_file)
    t = 0
    flag = 1
    profit = 0.0

    for i in range(len(live_dataset_handler_binance.dataset)):
        ans_buy_list.append(np.nan)
        ans_sell_list.append(np.nan)

    while t != predict_time:

        # If start is True, run the loop
        if start:
            # sleep until rounded 5 minute time
            time.sleep(1)
            wait_sec, next_execution_time = get_next_exec_wait()

            MyLogger.write("Next time of execution is {}, waiting {} seconds".format(next_execution_time, np.around(wait_sec, 2)), output_file)
            time.sleep(wait_sec)
            MyLogger.write("\n---Execution start at {} ---".format(datetime.now()), output_file)

            live_dataset_handler_binance.get_recent_OHLC()

            # Delete head of dataset and ans_list
            live_dataset_handler_binance.dataset.drop(live_dataset_handler_binance.dataset.head(1).index, inplace=True)
            ans_buy_list.pop(0)
            ans_sell_list.pop(0)

            live_dataset_handler_binance.print_to_file('Outputs/live_dataset.txt')

            t += 1

            look_back_list = []

            for step in range(LOOK_BACK):
                entry_row = live_dataset_handler_binance.dataset.iloc[-1 - step]
                del entry_row['Date']
                entry_row = entry_row.values
                norm_entry_row = main_normalizer.get_normalized_row(entry_row)
                look_back_list.append(np.array(norm_entry_row))

            inputs = np.concatenate(look_back_list)
            inputs = inputs * 0.99 + 0.01
            ans = nn.query(inputs)

            if ans[0] > NN_OUT_ANS_BUY_THRESHOLD:
                if flag != 0:
                    actual_price = live_dataset_handler_binance.get_actual_price()
                    buying_price = BUY_QUANTITY * actual_price
                    fees = (BUY_QUANTITY * actual_price) * FEE_PERCENTAGE
                    profit = profit - buying_price - fees

                    MyLogger.write("1. Buying at time : {}, for price {}.".format(live_dataset_handler_binance.dataset.iloc[-1]['Date'], np.around(buying_price, 4)), output_file)
                    MyLogger.write("2. Profit is {}\n".format(np.around(profit, 4)), output_file)
                    ans_buy_list.append(actual_price)
                    ans_sell_list.append(np.nan)
                    flag = 0
                    update_message("Buying for price {}.".format(np.around(buying_price, 4)))
                else:
                    ans_buy_list.append(np.nan)
                    ans_sell_list.append(np.nan)
                    MyLogger.write("Holding, profit is {}".format(np.around(profit, 4)), output_file)
                    pass
            elif ans[1] > NN_OUT_ANS_SELL_THRESHOLD:
                if flag != 1:
                    actual_price = live_dataset_handler_binance.get_actual_price()
                    selling_price = BUY_QUANTITY * actual_price
                    fees = (BUY_QUANTITY * actual_price) * FEE_PERCENTAGE
                    profit = profit + selling_price - fees

                    MyLogger.write("1. Selling at time : {}, for price {}.".format(live_dataset_handler_binance.dataset.iloc[-1]['Date'], np.around(selling_price, 4)), output_file)
                    MyLogger.write("2. Profit is {}\n".format(np.around(profit, 4)), output_file)
                    ans_buy_list.append(np.nan)
                    ans_sell_list.append(actual_price)
                    flag = 1
                    update_message("Selling for price {}.".format(np.around(selling_price, 4)))
                else:
                    ans_buy_list.append(np.nan)
                    ans_sell_list.append(np.nan)
                    MyLogger.write("Holding, profit is {}".format(np.around(profit, 4)), output_file)
                    pass
            else:
                ans_buy_list.append(np.nan)
                ans_sell_list.append(np.nan)
                MyLogger.write("Holding, profit is {}".format(np.around(profit, 4)), output_file)
                pass

            update_profit(np.around(profit, 4))
            MyLogger.write("---Execution end at {} ---\n".format(datetime.now()), output_file)

        if end:
            break

    msg = "-----------------------------------------------------------------------------------------------------------------------------------\n" + \
          "Final earning is {} EUR. All fees are included, application was buying for price 10% of actual price of {}.\n".format(profit, live_dataset_handler_binance.pair) + \
          "-----------------------------------------------------------------------------------------------------------------------------------"

    MyLogger.write(msg, output_file)
    output_file.close()


def plot_to_tkinter(ideal_signals: tuple, predicted_signals: tuple):

    global fig
    global ax
    global canvas
    global toolbar
    global live_dataset_handler_binance

    if len(ax) == 0:
        fig.append(plt.figure(figsize=(7, 3)))
        fig.append(plt.figure(figsize=(7, 3)))
        fig.append(plt.figure(figsize=(7, 3)))
        fig.append(plt.figure(figsize=(7, 3)))

        ax.append(fig[0].add_subplot(1, 1, 1))
        ax.append(fig[1].add_subplot(1, 1, 1))
        ax.append(fig[2].add_subplot(1, 1, 1))
        ax.append(fig[3].add_subplot(1, 1, 1))

        canvas.append(FigureCanvasTkAgg(fig[0], frame_left_up))
        canvas.append(FigureCanvasTkAgg(fig[1], frame_left_down))
        canvas.append(FigureCanvasTkAgg(fig[2], frame_right_up))
        canvas.append(FigureCanvasTkAgg(fig[3], frame_right_down))

        toolbar.append(tkagg.NavigationToolbar2Tk(canvas[0], frame_left_up))
        toolbar.append(tkagg.NavigationToolbar2Tk(canvas[1], frame_left_down))
        toolbar.append(tkagg.NavigationToolbar2Tk(canvas[2], frame_right_up))
        toolbar.append(tkagg.NavigationToolbar2Tk(canvas[3], frame_right_down))

        canvas[0].get_tk_widget().pack(side=tk.TOP)
        canvas[1].get_tk_widget().pack(side=tk.BOTTOM)
        canvas[2].get_tk_widget().pack(side=tk.TOP)
        canvas[3].get_tk_widget().pack(side=tk.BOTTOM)

        for t in toolbar:
            t.config(background='white')
            t._message_label.config(background='white')
            for button in t.winfo_children():
                button.config(background='white')
            t.pack()

    for a in ax:
        a.clear()
        a.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        a.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        a.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        a.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    if len(predicted_signals[0]) == len(predicted_signals[1]):
        if len(set(predicted_signals[0])) != 1:
            ax[0].scatter(live_dataset_handler_binance.dataset.index, predicted_signals[0], color='green', label='Predicted buy', marker='X', alpha=1)
        if len(set(predicted_signals[1])) != 1:
            ax[0].scatter(live_dataset_handler_binance.dataset.index, predicted_signals[1], color='red', label='Predicted sell', marker='X', alpha=1)
    else:
        print("\nLength of arrays are not the same, cannot plot!\n")

    if len(set(ideal_signals[0])) != 1:
        ax[0].scatter(live_dataset_handler_binance.dataset.index, ideal_signals[0], color='green', label='Buy Signal', marker='^', alpha=1)
    if len(set(ideal_signals[1])) != 1:
        ax[0].scatter(live_dataset_handler_binance.dataset.index, ideal_signals[1], color='red', label='Sell Signal', marker='v', alpha=1)

    ax[0].plot(live_dataset_handler_binance.dataset['Close'], 'b', label='Close')
    ax[0].grid(True)
    ax[0].legend(loc="lower left")

    ax[1].plot(live_dataset_handler_binance.dataset['Macd'], 'b', label='Macd')  # row=1, col=0
    ax[1].plot(live_dataset_handler_binance.dataset['Signal'], 'm', label='Signal')  # row=1, col=0
    ax[1].plot(live_dataset_handler_binance.dataset['Momentum'], 'go', label='Momentum')  # row=1, col=0
    ax[1].axhline(y=0, color='k')
    ax[1].grid(True)
    ax[1].legend(loc="lower right")

    ind_list = list(live_dataset_handler_binance.dataset)
    matching = [s for s in ind_list if "Ema" in s]
    ax[2].plot(live_dataset_handler_binance.dataset[matching[0]], 'b', label=matching[0])  # row=0, col=0
    ax[2].plot(live_dataset_handler_binance.dataset[matching[1]], 'm', label=matching[1])  # row=0, col=0
    ax[2].grid(True)
    ax[2].legend(loc="lower right")

    ax[3].plot(live_dataset_handler_binance.dataset['Rsi'], 'b', label='Rsi')  # row=0, col=0
    ax[3].axhline(y=30, color='r')
    ax[3].axhline(y=70, color='r')
    ax[3].grid(True)
    ax[3].legend(loc="lower right")

    for c in canvas:
        c.draw()

    for t in toolbar:
        t.update()


def construct_gui(width, height):
    # Tkinter stuff for creating simple GUI
    root = tk.Tk()
    root.title('Automatic CryptoTrader')

    # width for the Tk root
    w = width
    # height for the Tk root
    h = height

    # get screen width and height
    ws = root.winfo_screenwidth()
    hs = root.winfo_screenheight()

    # calculate x and y coordinates for the Tk root window
    x = (ws / 2) - (w / 2)
    y = (hs / 2) - (h / 2)

    # set the dimensions of the screen
    # and where it is placed
    root.geometry('%dx%d+%d+%d' % (w, h, x, y))
    root.configure(bg='white')

    return root


def start_handler():
    global start
    global end
    global start_time
    global live_dataset_handler_binance

    while live_dataset_handler_binance.dataset is None:
        pass

    start = True
    end = False
    update_message("Started predicting!")
    update_graphs()
    start_time = datetime.utcnow()


def end_handler():
    global start
    global end
    global after_id1
    global after_id2
    global window

    start = False
    end = True
    if after_id1 != -1:
        window.after_cancel(after_id1)
    if after_id2 != -1:
        window.after_cancel(after_id2)
    window.destroy()


def update_profit(current_profit):
    Label_Profit_Title.config(text='Current profit is: {}'.format(current_profit))


def update_message(message):
    Label_LastAction_Title.config(text='Last action was at {}: {}'.format(time.strftime('%H:%M:%S', time.localtime()), message))


def update_clock():
    global after_id1
    global start_time
    global Label_Clock
    global Label_Timer
    global Label_Elapsed

    # change the text of the time_label according to the current time
    Label_Clock.config(text=time.strftime('Current time is: %H:%M:%S', time.localtime()), font='Times 25')
    if start:
        seconds, _ = get_next_exec_wait()
        ty_res = time.gmtime(seconds + 1.0)
        Label_Timer.config(text=time.strftime('Next exec is in: %H:%M:%S', ty_res), font='Times 25')

        now = datetime.utcnow()
        elapsed_time = now - start_time
        elapsed_time = elapsed_time.seconds
        elapsed_time = time.gmtime(elapsed_time + 1.0)
        Label_Elapsed.config(text=time.strftime('Elapsed time: %H:%M:%S', elapsed_time), font='Times 25')
    else:
        Label_Timer.config(text='Next exec is in:  -- : -- : --', font='Times 25')
        Label_Elapsed.config(text='Elapsed time:  -- : -- : --', font='Times 25')

    # Reschedule update_clock function to update time_label every 100 ms
    after_id1 = window.after(100, update_clock)


def update_graphs():
    global ax
    global fig
    global canvas
    global toolbar
    global live_dataset_handler_binance
    global after_id2

    '''
    If we will use plot_to_tkinter method from OhlcHandler, use this section
    '''
    # if len(ax) == 0:
    #     ideal_signals = Teacher.generate_buy_sell_signal(live_dataset_handler_binance.dataset, FILTER_CONSTANT)
    #     fig, ax, canvas, toolbar = live_dataset_handler_binance.plot_to_tkinter(ideal_signals, (ans_buy_list, ans_sell_list), graph_frames, ax, canvas, fig, toolbar)
    # elif ax is not None:
    #     ideal_signals = Teacher.generate_buy_sell_signal(live_dataset_handler_binance.dataset, FILTER_CONSTANT)
    #     live_dataset_handler_binance.plot_to_tkinter(ideal_signals, (ans_buy_list, ans_sell_list), graph_frames, ax, canvas=canvas, fig=fig, toolbar=toolbar)

    '''
    If we will use plot_to_tkinter method from this file, use this section
        '''
    ideal_signals = Teacher.generate_buy_sell_signal(live_dataset_handler_binance.dataset, FILTER_CONSTANT)
    plot_to_tkinter(ideal_signals, (ans_buy_list, ans_sell_list))

    after_id2 = window.after(30000, update_graphs)


# Global variables for controlling application and for storing global information needed outside threaded
# predict_live() function. Matplotlib want to run in main thread, not second
# and because predict_live is running as second, plotting must be done outside this function, so we must store info
# for plotting to global variables
start = False
end = False
live_dataset_handler_binance: BinanceOhlcHandler
fig = []
ax = []
canvas = []
toolbar = []
ans_buy_list = []
ans_sell_list = []
after_id1 = -1
after_id2 = -1
start_time: datetime

# Predict_live function must be running in separate thread because this function is sleeping for 99,9 % of time,
# and so if we dont want to block entire GUI app by this sleep, we must run this function in separate thread
thread = threading.Thread(target=predict_live)
thread.start()

window = construct_gui(1600, 900)

# Define frames for graphs
frame_graph = tk.Frame(window)
frame_graph.pack(side=tk.BOTTOM)

frame_left_column = tk.Frame(frame_graph)
frame_left_column.pack(side=tk.LEFT)

frame_left_up = tk.Frame(frame_left_column)
frame_left_up.pack(side=tk.TOP)

frame_left_down = tk.Frame(frame_left_column)
frame_left_down.pack(side=tk.BOTTOM)

frame_right_column = tk.Frame(frame_graph)
frame_right_column.pack(side=tk.RIGHT)

frame_right_up = tk.Frame(frame_right_column)
frame_right_up.pack(side=tk.TOP)

frame_right_down = tk.Frame(frame_right_column)
frame_right_down.pack(side=tk.BOTTOM)

# Define buttons, clocks, etc..
Label_Clock = tk.Label(window, justify='center', bg='white')  # create the label for timer
Label_Clock.place(relx=0.15, rely=0.02, anchor='nw')
Label_Timer = tk.Label(window, justify='center', bg='white')  # create the label for timer
Label_Timer.place(relx=0.85, rely=0.08, anchor='ne')
Label_Elapsed = tk.Label(window, justify='center', bg='white')  # create the label for timer
Label_Elapsed.place(relx=0.85, rely=0.02, anchor='ne')

button1 = tk.Button(window, text='Start', command=start_handler, height=1, width=5, font='Arial 20', activebackground='green', bg='grey70')
button1.place(relx=0.18, rely=0.08, anchor='nw')
button2 = tk.Button(window, text='End', command=end_handler, height=1, width=5, font='Arial 20', activebackground='red', bg='grey70')
button2.place(relx=0.28, rely=0.08, anchor='nw')

Label_LastAction_Title = tk.Label(window, text='Last action was at: ', font='Arial 20', bg='white')
Label_LastAction_Title.place(relx=0.08, rely=0.20, anchor='nw')

Label_Profit_Title = tk.Label(window, text='Current profit is: ', font='Arial 20', bg='white')
Label_Profit_Title.place(relx=0.92, rely=0.20, anchor='ne')

# First call of handler functions
update_profit(0.0)
update_message("Application started")

# Automatic startup switch if ran from console
if len(sys.argv) == 2:
    if sys.argv[1] == '-a':
        time.sleep(2)
        start_handler()
    else:
        pass
else:
    pass

# Set startup time, start updating clocks ans then run GUI app
after_id1 = window.after(0, update_clock)
start_time = datetime.utcnow()
window.mainloop()

# When tkinter windows will be closed, automatically set end = True for breaking infinite loop
end = True

# When the GUI is closed we join the thread
thread.join()




