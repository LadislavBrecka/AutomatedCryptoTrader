import time
import numpy as np
from Config import *
from Modules.BinanceOhlcHandler import BinanceOhlcHandler
import Modules.services as services
import Modules.NeuralNetwork_2_hidden as nn_2_hidden
from Modules.services import MyLogger
from datetime import datetime, timedelta
import Modules.Teacher as Teacher
import tkinter as tk
import threading
import sys


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

            MyLogger.write("Next time of execution is {}, waiting {} seconds".format(next_execution_time, wait_sec), output_file)
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

            # ans = [1.0,0.01]

            if ans[0] > NN_OUT_ANS_BUY_THRESHOLD:
                if flag != 0:
                    actual_price = live_dataset_handler_binance.get_actual_price()
                    buying_price = BUY_QUANTITY * actual_price
                    fees = (BUY_QUANTITY * actual_price) * FEE_PERCENTAGE
                    profit = profit - buying_price - fees

                    MyLogger.write("1. Buying at time : {}, for price {}.".format(live_dataset_handler_binance.dataset.iloc[-1]['Date'], buying_price), output_file)
                    MyLogger.write("2. Profit is {}\n".format(profit), output_file)
                    ans_buy_list.append(actual_price)
                    ans_sell_list.append(np.nan)
                    flag = 0
                    update_message("Buying for price {}.".format(buying_price))
                else:
                    ans_buy_list.append(np.nan)
                    ans_sell_list.append(np.nan)
                    MyLogger.write("Holding, profit is {}".format(profit), output_file)
                    pass
            elif ans[1] > NN_OUT_ANS_SELL_THRESHOLD:
                if flag != 1:
                    actual_price = live_dataset_handler_binance.get_actual_price()
                    selling_price = BUY_QUANTITY * actual_price
                    fees = (BUY_QUANTITY * actual_price) * FEE_PERCENTAGE
                    profit = profit + selling_price - fees

                    MyLogger.write("1. Selling at time : {}, for price {}.".format(live_dataset_handler_binance.dataset.iloc[-1]['Date'], selling_price), output_file)
                    MyLogger.write("2. Profit is {}\n".format(profit), output_file)
                    ans_buy_list.append(np.nan)
                    ans_sell_list.append(actual_price)
                    flag = 1
                    update_message("Selling for price {}.".format(selling_price))
                else:
                    ans_buy_list.append(np.nan)
                    ans_sell_list.append(np.nan)
                    MyLogger.write("Holding, profit is {}".format(profit), output_file)
                    pass
            else:
                ans_buy_list.append(np.nan)
                ans_sell_list.append(np.nan)
                MyLogger.write("Holding, profit is {}".format(profit), output_file)
                pass

            update_profit(profit)
            MyLogger.write("---Execution end at {} ---\n".format(datetime.now()), output_file)

        if end:
            break

    msg = "-----------------------------------------------------------------------------------------------------------------------------------\n" + \
          "Final earning is {} EUR. All fees are included, application was buying for price 10% of actual price of {}.\n".format(profit, live_dataset_handler_binance.pair) + \
          "-----------------------------------------------------------------------------------------------------------------------------------"

    MyLogger.write(msg, output_file)
    output_file.close()


def start_handler():
    global start
    global end
    global start_time
    start = True
    end = False
    update_message("Started predicting!")
    update_graphs()
    start_time = datetime.utcnow()


def end_handler():
    global start
    global end

    start = False
    end = True
    if after_id1 != -1:
        window.after_cancel(after_id1)
    if after_id2 != -1:
        window.after_cancel(after_id2)
    window.destroy()


def update_profit(current_profit):
    Label_Profit.config(text=str(np.around(current_profit,3)), font='Times 20')


def update_message(message):
    Label_LastAction_Title.config(text='Last action was at {}:'.format(time.strftime('%H:%M:%S', time.localtime())))
    Label_LastAction.config(text=str(message), font='Times 20')


def update_clock():
    global after_id1
    global start_time
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

    if ax is None:
        ideal_signals = Teacher.generate_buy_sell_signal(live_dataset_handler_binance.dataset, FILTER_CONSTANT)
        fig, ax, canvas, toolbar = live_dataset_handler_binance.plot_to_tkinter(ideal_signals, (ans_buy_list, ans_sell_list), frame_bottom)
    elif ax is not None:
        ideal_signals = Teacher.generate_buy_sell_signal(live_dataset_handler_binance.dataset, FILTER_CONSTANT)
        live_dataset_handler_binance.plot_to_tkinter(ideal_signals, (ans_buy_list, ans_sell_list), frame_bottom, canvas=canvas, ax=ax, fig=fig, toolbar=toolbar)

    after_id2 = window.after(10000, update_graphs)


# Global variables for controlling application and for storing global information needed outside threaded
# predict_live() function. Matplotlib want to run in main thread, not second
# and because predict_live is running as second, plotting must be done outside this function, so we must store info
# for plotting to global variables
start = False
end = False
live_dataset_handler_binance: BinanceOhlcHandler
after_id1 = -1
after_id2 = -1
start_time: datetime
ax = None
fig = None
canvas = None
ans_buy_list = []
ans_sell_list = []
toolbar = None

# Predict_live function must be running in separate thread because this function is sleeping for 99,9 % of time,
# and so if we dont want to block entire GUI app by this sleep, we must run this function in separate thread
thread = threading.Thread(target=predict_live)
thread.start()

# Tkinter stuff for creating simple GUI
window = tk.Tk()
window.title('Automatic CryptoTrader')
window.geometry('800x600')
window.configure(bg='white')

Label_Clock = tk.Label(window, justify='center', bg='white')  # create the label for timer
Label_Clock.place(relx=0.03, rely=0.02, anchor='nw')
Label_Timer = tk.Label(window, justify='center', bg='white')  # create the label for timer
Label_Timer.place(relx=0.55, rely=0.12, anchor='nw')
Label_Elapsed = tk.Label(window, justify='center', bg='white')  # create the label for timer
Label_Elapsed.place(relx=0.58, rely=0.02, anchor='nw')

button1 = tk.Button(window, text='Start', command=start_handler, height=1, width=5, font='Arial 20', activebackground='green', bg='grey70')
button1.place(relx=0.1, rely=0.12, anchor='nw')
button2 = tk.Button(window, text='End', command=end_handler, height=1, width=5, font='Arial 20', activebackground='red', bg='grey70')
button2.place(relx=0.25, rely=0.12, anchor='nw')

Label_LastAction_Title = tk.Label(window, text='Last action was at:', font='Arial 20', bg='white')
Label_LastAction_Title.place(relx=0.03, rely=0.25, anchor='nw')
Label_LastAction = tk.Label(window, bg='white')
Label_LastAction.place(relx=0.03, rely=0.32, anchor='nw')

Label_Profit_Title = tk.Label(window, text='Current profit is: ', font='Arial 20', bg='white')
Label_Profit_Title.place(relx=0.65, rely=0.25, anchor='nw')
Label_Profit = tk.Label(window, bg='white')
Label_Profit.place(relx=0.65, rely=0.32, anchor='nw')

window.rowconfigure(0, weight=2)
window.rowconfigure(1, weight=1)
window.columnconfigure(0, weight=1)
window.columnconfigure(1, weight=2)

frame_bottom = tk.Frame(window)
frame_bottom.pack(side=tk.BOTTOM)

update_profit(0.0)
update_message("Application started")

'''
Automatic startup switch
'''
if len(sys.argv) == 2:
    if sys.argv[1] == '-a':
        time.sleep(2)
        start_handler()
    else:
        pass
else:
    pass


after_id1 = window.after(0, update_clock)
start_time = datetime.utcnow()
window.mainloop()

# When tkinter windows will be closed, automatically set end = True for breaking infinite loop
end = True

# When the GUI is closed we join the thread
thread.join()




