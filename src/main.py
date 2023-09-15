from tkinter import *
import tkinter as tk
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename
import cv2
from keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
from keras.models import load_model

# Set up GUI
root = tk.Tk()  # Makes main window
root.wm_title("Bone Fracure Detection")
root.wm_state('zoomed')

root.configure(bg='gray')

title = Label(root, text='BONE \n\n FRACTURE \n\nDETECTION',
              bg='black', fg='white', padx=30, pady=320, font=('comicsans', 30, 'bold'))
title.place(x=0, y=0)

title = Label(root, text="Please, Proceed the 'Bone Fracture Detection' of X-Ray Image From Here!",
              bg='teal', fg='azure', padx=280, font=('comicsans', 15, 'bold'))
title.place(x=304, y=0)

title = Label(root, text='',
              bg='teal', padx=620, pady=30, font=('comicsans', 20, 'bold'))
title.place(x=304, y=30)

model = load_model('./Model/bone_detect.h5', compile=False)

lib = {0: 'FRACTURE BONE APPEARED', 1: 'NORMAL BONE APPEARED'}


def clear():
    e1.destroy()
    e2.destroy()
    e3.destroy()
    e4.destroy()
    e5.destroy()
    titled.destroy()
    draw6 = Canvas(root, width=20, height=20, bd=0,
                   highlightthickness=0, background='teal')
    draw6.place(x=1455, y=45)
    draw6.create_line(12, 0, 12, 20, fill='teal', arrow='last', width=7)

    draw6 = Canvas(root, width=20, height=20, bd=0,
                   highlightthickness=0, background='teal')
    draw6.place(x=340, y=45)
    draw6.create_line(12, 0, 12, 20, fill='dark red', arrow='last', width=7)


def detect(img):
    img1 = load_img(img, target_size=(224, 224, 3))
    img1 = img_to_array(img1)
    img1 = img1/255
    img1 = np.expand_dims(img1, [0])
    answer = model.predict(img1)
    print(answer)
    y_class = answer.argmax(axis=-1)
    print(y_class)
    y = " ".join(str(x) for x in y_class)
    y = int(y)
    res = lib[y]
    print(res)

    draw6 = Canvas(root, width=20, height=20, bd=0,
                   highlightthickness=0, background='teal')
    draw6.place(x=1280, y=45)
    draw6.create_line(12, 0, 12, 20, fill='teal', arrow='last', width=7)

    draw6 = Canvas(root, width=20, height=20, bd=0,
                   highlightthickness=0, background='teal')
    draw6.place(x=1455, y=45)
    draw6.create_line(12, 0, 12, 20, fill='dark red', arrow='last', width=7)
    global titled
    titled = Label(root, text=res, fg='white',
                   bg='black', padx=350, pady=40, font=('comicsans', 30, 'bold'))
    titled.place(x=304, y=550)

    btn7 = Button(root, text='CLEAR', command=lambda: clear())
    btn7.place(x=1445, y=70)


def segment(img):
    image = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # reshape the image to a 2D array of pixels and 3 color values (RGB)
    pixel_values = image.reshape((-1, 3))
    # convert to float
    pixel_values = np.float32(pixel_values)
    print(pixel_values.shape)
    # define stopping criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.2)
    # number of clusters (K)
    k = 3
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None,
                                      criteria, 15, cv2.KMEANS_RANDOM_CENTERS)
    # convert back to 8 bit values
    centers = np.uint8(centers)
    # print(labels)
    print(centers)

    # flatten the labels array
    labels = labels.flatten()
    # convert all pixels to the color of the centroids
    segmented_image = centers[labels.flatten()]
    # reshape back to the original image dimension
    img1 = segmented_image.reshape(image.shape)
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    print('img1 shape = '+str(img1.shape))
    img1 = Image.fromarray(img1)
    img1 = ImageTk.PhotoImage(img1)
    # cv2.imwrite("seg.jpg", img)
    global e5
    e5 = Label(root, image=img1)
    e5.place(x=1280, y=160)
    e5.image = img1

    draw6 = Canvas(root, width=20, height=20, bd=0,
                   highlightthickness=0, background='teal')
    draw6.place(x=1080, y=45)
    draw6.create_line(12, 0, 12, 20, fill='teal', arrow='last', width=7)

    draw6 = Canvas(root, width=20, height=20, bd=0,
                   highlightthickness=0, background='teal')
    draw6.place(x=1280, y=45)
    draw6.create_line(12, 0, 12, 20, fill='dark red', arrow='last', width=7)

    btn6 = Button(root, text='DETECT  FRACTURE',
                  command=lambda: detect(filename))
    btn6.place(x=1240, y=70)

    title = Label(root, text='SEGMENTATION',
                  bg='teal', padx=18, pady=10, font=('comicsans', 14, 'bold'))
    title.place(x=1280, y=420)


def edges(img):
    img1 = cv2.Canny(image=img, threshold1=100, threshold2=120)
    img1 = Image.fromarray(img1)
    img1 = ImageTk.PhotoImage(img1)
    # cv2.imwrite("edge.jpg", img1)
    global e4
    e4 = Label(root, image=img1)
    e4.place(x=1050, y=160)
    e4.image = img1

    draw6 = Canvas(root, width=20, height=20, bd=0,
                   highlightthickness=0, background='teal')
    draw6.place(x=890, y=45)
    draw6.create_line(12, 0, 12, 20, fill='teal', arrow='last', width=7)

    draw6 = Canvas(root, width=20, height=20, bd=0,
                   highlightthickness=0, background='teal')
    draw6.place(x=1080, y=45)
    draw6.create_line(12, 0, 12, 20, fill='dark red', arrow='last', width=7)

    btn5 = Button(root, text='SEGMENT  IMAGE', command=lambda: segment(img))
    btn5.place(x=1040, y=70)

    title = Label(root, text='EDGE IMAGE',
                  bg='teal', padx=34, pady=10, font=('comicsans', 14, 'bold'))
    title.place(x=1050, y=420)


def noise(img):
    img = cv2.medianBlur(img, 5)
    img1 = Image.fromarray(img)
    img1 = ImageTk.PhotoImage(img1)
    # cv2.imwrite("noise.jpg", img)
    global e3
    e3 = Label(root, image=img1)
    e3.place(x=820, y=160)
    e3.image = img1

    draw6 = Canvas(root, width=20, height=20, bd=0,
                   highlightthickness=0, background='teal')
    draw6.place(x=700, y=45)
    draw6.create_line(12, 0, 12, 20, fill='teal', arrow='last', width=7)

    draw6 = Canvas(root, width=20, height=20, bd=0,
                   highlightthickness=0, background='teal')
    draw6.place(x=890, y=45)
    draw6.create_line(12, 0, 12, 20, fill='dark red', arrow='last', width=7)

    btn4 = Button(root, text='DETECT  EDGES', command=lambda: edges(img))
    btn4.place(x=855, y=70)

    title = Label(root, text='NOISE-LESS IMAGE',
                  bg='teal', padx=3, pady=10, font=('comicsans', 14, 'bold'))
    title.place(x=820, y=420)


def gray(img):
    img = cv2.imread(img)
    img = cv2.resize(img, (190, 230))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img1 = Image.fromarray(img)
    img1 = ImageTk.PhotoImage(img1)
    # cv2.imwrite("gray.jpg", img)
    global e2
    e2 = Label(root, image=img1)
    e2.place(x=590, y=160)
    e2.image = img1

    draw6 = Canvas(root, width=20, height=20, bd=0,
                   highlightthickness=0, background='teal')
    draw6.place(x=520, y=45)
    draw6.create_line(12, 0, 12, 20, fill='teal', arrow='last', width=7)

    draw6 = Canvas(root, width=20, height=20, bd=0,
                   highlightthickness=0, background='teal')
    draw6.place(x=700, y=45)
    draw6.create_line(12, 0, 12, 20, fill='dark red', arrow='last', width=7)

    btn3 = Button(root, text='REMOVE  NOISE', command=lambda: noise(img))
    btn3.place(x=665, y=70)

    title = Label(root, text='GRAY IMAGE',
                  bg='teal', padx=34, pady=10, font=('comicsans', 14, 'bold'))
    title.place(x=590, y=420)


def file_upload():
    f_types = [('Jpg files', '*.jpg'), ('PNG files', '*.png')]
    global filename
    filename = tk.filedialog.askopenfilename(filetypes=f_types)
    img = Image.open(filename)
    img = img.resize((190, 230))
    img = ImageTk.PhotoImage(img)
    # cv2.imwrite("img.jpg", img)
    print(type(img))
    global e1
    e1 = Label(root, image=img)
    e1.place(x=360, y=160)
    e1.image = img

    draw6 = Canvas(root, width=20, height=20, bd=0,
                   highlightthickness=0, background='teal')
    draw6.place(x=340, y=45)
    draw6.create_line(12, 0, 12, 20, fill='teal', arrow='last', width=7)

    draw6 = Canvas(root, width=20, height=20, bd=0,
                   highlightthickness=0, background='teal')
    draw6.place(x=520, y=45)
    draw6.create_line(12, 0, 12, 20, fill='dark red', arrow='last', width=7)

    title = Label(root, text='X-RAY IMAGE',
                  bg='teal', padx=32, pady=10, font=('comicsans', 14, 'bold'))
    title.place(x=360, y=420)


title = Label(root, text='',
              bg='black', padx=615, pady=40, font=('comicsans', 30, 'bold'))
title.place(x=304, y=550)

draw1 = Canvas(root, width=75, height=20, bd=0,
               highlightthickness=0, background='teal')
draw1.place(x=405, y=70)
draw1.create_line(0, 12, 75, 12, fill='blue', arrow='last', width=7)

draw2 = Canvas(root, width=75, height=20, bd=0,
               highlightthickness=0, background='teal')
draw2.place(x=580, y=70)
draw2.create_line(0, 12, 75, 12, fill='blue', arrow='last', width=7)

draw3 = Canvas(root, width=75, height=20, bd=0,
               highlightthickness=0, background='teal')
draw3.place(x=770, y=70)
draw3.create_line(0, 12, 75, 12, fill='blue', arrow='last', width=7)

draw4 = Canvas(root, width=75, height=20, bd=0,
               highlightthickness=0, background='teal')
draw4.place(x=955, y=70)
draw4.create_line(0, 12, 75, 12, fill='blue', arrow='last', width=7)

draw5 = Canvas(root, width=75, height=20, bd=0,
               highlightthickness=0, background='teal')
draw5.place(x=1155, y=70)
draw5.create_line(0, 12, 75, 12, fill='blue', arrow='last', width=7)

draw7 = Canvas(root, width=75, height=20, bd=0,
               highlightthickness=0, background='teal')
draw7.place(x=1360, y=70)
draw7.create_line(0, 12, 75, 12, fill='blue', arrow='last', width=7)

draw6 = Canvas(root, width=20, height=20, bd=0,
               highlightthickness=0, background='teal')
draw6.place(x=340, y=45)
draw6.create_line(12, 0, 12, 20, fill='dark red', arrow='last', width=7)


btn1 = Button(root, text='X-RAY  IMAGE', command=lambda: file_upload())
btn1.place(x=310, y=70)

btn2 = Button(root, text='GRAY  IMAGE', command=lambda: gray(filename))
btn2.place(x=490, y=70)

btn3 = Button(root, text='REMOVE  NOISE')
btn3.place(x=665, y=70)

btn4 = Button(root, text='DETECT  EDGES')
btn4.place(x=855, y=70)

btn5 = Button(root, text='SEGMENT  IMAGE')
btn5.place(x=1040, y=70)

btn6 = Button(root, text='DETECT  FRACTURE')
btn6.place(x=1240, y=70)

btn7 = Button(root, text='CLEAR')
btn7.place(x=1445, y=70)

root.mainloop()
