import cv2
from PIL import Image
from src.model import CNN
import numpy as np
import src.image_extraction as vid
import tkinter as tk
import os
from PIL import Image, ImageTk


class CounterGUI:
    def __init__(self, path=None):
        self.root = tk.Tk()
        self.root.title('Image labeling')

        self.running = 0
        self.after_id = None
        self.video_handler = None
        self.current_frame = None
        self.line_of_interest = [None, None, None, None]
        self.draw_state = 0

        self.menubar = tk.Menu(self.root)
        self.filemenu = tk.Menu(self.menubar, tearoff=0)

        self.filemenu.add_command(label="Open")
        self.filemenu.add_command(label="Close")
        self.filemenu.add_separator()
        self.filemenu.add_command(label="Exit", command=self.root.quit)
        self.menubar.add_cascade(label="File", menu=self.filemenu)

        self.menubar.add_command(label="Start", command=self.start)

        self.canvas = tk.Canvas(self.root, width=1920, height=1080)
        self.canvas.pack()

        self.canvas.bind("<Button-1>", self.draw_line)

        if path:
            self.open_file(path)

        self.root.config(menu=self.menubar)
        self.root.mainloop()

    def start(self):
        if self.running:
            self.running = 0
        else:
            self.running = 1
            self.update()
        self.menubar.entryconfigure(2, label='{}'.format('Start' if self.running else 'Stop'))

    def update(self):
        self.current_frame = ImageTk.PhotoImage(Image.fromarray(self.video_handler.get_frame()))
        self.canvas.create_image(0, 0, image=self.current_frame, anchor=tk.N + tk.W)
        self.canvas.update()
        if self.running:
            self.after_id = self.root.after(30, self.update())
        else:
            self.root.after_cancel(self.after_id)

    def open_file(self, path):
        extension = os.path.splitext(path)[1]
        if extension.lower() in ['.jpg', '.jpeg']:
            self.current_frame = ImageTk.PhotoImage(Image.open(path))
        elif extension.lower() in ['.mp4']:
            self.video_handler = vid.VideoHandler(path)
            self.current_frame = ImageTk.PhotoImage(Image.fromarray(self.video_handler.get_frame()))
        self.canvas.create_image(0, 0, image=self.current_frame, anchor=tk.N + tk.W)

    def draw_line(self, event):
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        if not self.draw_state:
            self.line_of_interest[0], self.line_of_interest[1] = x, y
            self.draw_state = (self.draw_state + 1) % 2
            self.canvas.bind("<Motion>", self.draw_line_move)
        else:
            self.canvas.unbind("<Motion>")
            self.line_of_interest[2], self.line_of_interest[3] = x, y
            self.draw_state = (self.draw_state + 1) % 2

    def draw_line_move(self, event):
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        self.canvas.delete('line')
        self.canvas.create_line(self.line_of_interest[0], self.line_of_interest[1], x, y, fill='red', width=3,
                                tag='line')


if __name__ == '__main__':
    x = CounterGUI(path='../data/video/video-25-4-long.MP4')
    print(x.line_of_interest)
