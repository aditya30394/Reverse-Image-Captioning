#!/usr/bin/python

import Tkinter

from Tkinter import *
from PIL import Image, ImageTk
import numpy
import random

top = Tk()
top.title = 'Reverse Image Captioning'
top.geometry('500x500')

top_row = Frame(top).grid(row=0)

left = Frame(top_row).grid(row=0, column=0)
L1 = Label(left, text="Enter the image description:").grid(row=0)
E1 = Entry(left).grid(row=1)

right = Frame(top_row).grid(row=0, column=1)

canvas = Canvas(right, width=300,height=300, bd=0,bg='white')
canvas.grid(row=0, column=1)

def GenerateImage():
    img = random.randint(1,5)
    img_name = str(img)+'.png'    
    load = Image.open(img_name)
    w, h = load.size
    load = load.resize((w, h))
    imgfile = ImageTk.PhotoImage(load)
    
    canvas.image = imgfile  # <--- keep reference of your image
    canvas.create_image(2,2,anchor='nw',image=imgfile)


submit_button = Button(top, text ='Generate Image', command = GenerateImage)
submit_button.grid(row=2, column=0)

submit_button = Button(top, text ='Exit', command = top.quit)
submit_button.grid(row=2, column=1)

top.mainloop()
