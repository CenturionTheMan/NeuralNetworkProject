import math
from tkinter import *
from threading import Thread
from tkinter import filedialog as fd

import numpy as np
import pandas as pd

import neural_network as nn

__pixel_size = 20
__canvas_data = np.zeros(shape=(28, 28), dtype=int)
__canvas_pixels_ids = np.zeros(shape=(28, 28), dtype=int)
__canvas: Canvas = None
__labels_perc = [Label] * 10


def run_gui():
    global __pixel_size, __canvas_data, __canvas, __labels_perc

    nn.initialize(784, 16, 16, 10, activation_function='relu')

    root = Tk()
    draw_width = 560
    draw_height = 560

    root.title("digit recognition")
    root.resizable(width=False, height=False)

    # FRAME
    frame = Frame(root, width=900, height=560, background="gray")
    frame.pack()

    # CANVAS
    __canvas = Canvas(frame, width=draw_width, height=draw_height, bg="white")
    __canvas.bind("<B1-Motion>", on_left_mouse_click)
    __canvas.bind("<B3-Motion>", on_right_mouse_click)
    __canvas.pack(side="left")

    # CLEAR BUTTON
    clear_button = Button(frame, text="Clear canvas", command=clear_canvas, background='green', height=5, width=15)
    clear_button.pack(pady=10, padx=2)

    # LABELS
    for i in range(len(__labels_perc)):
        __labels_perc[i] = Label(frame, text="Probability for " + str(i) + ": None", background="gray")
        __labels_perc[i].pack(pady=1)

    # NN BUTTONS
    save_button = Button(frame, text="Save neural network\nto file", command=show_save_dialog)
    save_button.pack(pady=(0, 10), padx=2, side='bottom')
    open_button = Button(frame, text="Load neural network\nfrom file", command=show_open_dialog)
    open_button.pack(pady=2, padx=2, side='bottom')

    init_learning_button = Button(frame, text="Initialize\nmachine learning", command=init_nn_learning_on_new_thread,
                                  background="yellow")
    init_learning_button.pack(pady=10, padx=2, side='bottom')

    root.mainloop()


def show_open_dialog():
    filename = fd.askopenfilename(defaultextension=".xml", filetypes=[("XML Files", "*.xml")])
    nn.load_from_xml_file(filename)


def show_save_dialog():
    filename = fd\
        .asksaveasfile(initialfile='Untitled.xml', defaultextension=".xml", filetypes=[("XML Files", "*.xml")]).name
    nn.save_to_xml_file(filename)


def init_nn_learning_on_new_thread():
    print("Learning starting...")
    Thread(target=init_nn_learning).start()


def load_digit_data(path):
    raw_data = pd.read_csv(path)
    data = []
    for single_row in raw_data.values:
        expected = [0 for x in range(10)]
        expected[int(single_row[0])] = 1
        inputs = [1 if x > 0 else 0 for x in single_row[1:]]
        data.append((inputs, expected))
    return data


def init_nn_learning():
    training_data1 = load_digit_data(
        "mnist_data/mnist_train_data1.csv")

    training_data2 = load_digit_data(
        "mnist_data/mnist_train_data1.csv")

    training_data = training_data1 + training_data2

    test_data = load_digit_data(
        "mnist_data/mnist_test_data.csv")

    print("Data loaded")

    nn.train_with_mini_batch_gradient_descent(training_data, epoch_amount=30, batch_size=50, expected_max_error=0.01,
                                              learning_rate=0.01)

    correct_prediction = 0
    for single in test_data:
        predictions = nn.predict(single[0])
        predicted_num = predictions.index(max(predictions))

        expected_num = single[1].index(max(single[1]))

        if predicted_num == expected_num:
            correct_prediction += 1

    perc_correct = 100 * correct_prediction / len(test_data)
    print(f"Correctness: {round(perc_correct, 2)}%")


def on_left_mouse_click(event):
    x_pos = event.x
    y_pos = event.y
    index_x = int(np.floor(x_pos / __pixel_size))
    index_y = int(np.floor(y_pos / __pixel_size))

    update_canvas(index_x, index_y, 1)
    Thread(target=on_canvas_data_changed).start()


def on_right_mouse_click(event):
    x_pos = event.x
    y_pos = event.y
    index_x = int(np.floor(x_pos / __pixel_size))
    index_y = int(np.floor(y_pos / __pixel_size))

    update_canvas(index_x, index_y, 0)
    Thread(target=on_canvas_data_changed).start()


def update_canvas(x, y, value):
    global __canvas_data, __canvas_pixels_ids

    if x > 27 or y > 27:
        return

    for i in range(-1, 2):
        for j in range(-1, 2):
            if j + y >= len(__canvas_data) or i + x >= len(__canvas_data[j + y]) or i + x < 0 or j + y < 0:
                continue
            __canvas_data[j + y][i + x] = value

    for i in range(x - 1, x + 2):
        for j in range(y - 1, y + 2):
            if i < 0 or i > 27 or j < 0 or j > 27:
                continue

            if value == 1:
                fill_col = "gray24"
                __canvas.delete(__canvas_pixels_ids[j][i])
                __canvas_pixels_ids[j][i] = __canvas.create_rectangle(
                                    i*__pixel_size, j*__pixel_size,
                                    i*__pixel_size + __pixel_size,
                                    j*__pixel_size + __pixel_size, fill=fill_col, outline=fill_col)
            else:
                fill_col = "white"
                __canvas.delete(__canvas_pixels_ids[j][i])
                __canvas_pixels_ids[j][i] = __canvas.create_rectangle(
                    i * __pixel_size, j * __pixel_size,
                    i * __pixel_size + __pixel_size,
                    j * __pixel_size + __pixel_size, fill=fill_col, outline=fill_col)


def clear_canvas():
    for x in range(0, 28):
        for y in range(0, 28):
            update_canvas(x, y, 0)


def on_canvas_data_changed():
    temp_data = np.reshape(__canvas_data, (__canvas_data.size, 1))
    predictions = nn.predict(temp_data)
    change_label_perc_text(predictions)


def change_label_perc_text(predictions):
    global __labels_perc
    index_max = predictions.index(max(predictions))
    for i in range(len(predictions)):
        bg_col = "gray"
        if index_max == i:
            bg_col = "green"
        __labels_perc[i].config(text="Probability for " + str(i) + ": " + str(np.round(predictions[i]*100, 2)) + "%", background=bg_col)