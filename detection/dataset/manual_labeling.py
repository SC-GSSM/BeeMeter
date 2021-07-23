import os
import tkinter as tk
from shutil import copy

from PIL import Image, ImageTk

from detection.dataset.database_connector import Database


class LabelGUI:
    """
    Simple tkinter based GUI for easy and fast one class, bounding box labeling or label correction. With features:
        - Only need Python and Tkinter module
        - Zoom 1x, 2x, 4x
        - Labels are directly stored in SQLite database
    """

    def __init__(self, img_path, db_path, start=0, training_path=None, cp=False):
        """
        :param img_path: Path to images that should be labeled.
        :param db_path: Path to SQLite database for label storage.
        :param start: Start point in label directory.
        :param training_path: Path to training data.
        :param cp: If true, copy labeled files to training_path after quiting this application.
        """
        self.db = Database(db_path)
        self.root = tk.Tk()

        self.img_path = img_path
        self.img_list = os.listdir(img_path)
        self.img_list.sort()

        self.current_img_counter = start
        self.width, self.height = None, None
        self.img = None
        self.img_zoomed2 = None
        self.img_zoomed4 = None
        self.zoom_mode = None  # 1="no zoom", 2="2x zoom", 4="4x zoom"

        self.img_id = None

        self.first_point = (0, 0)
        self.last_rect = None
        self.current_labels = {}

        self.cp = cp
        self.training_path = training_path
        self.finished_img = []

        self.root.title('Image labeling')
        self.root.geometry("3000x1080")
        # self.root.attributes("-fullscreen", True)
        self.root.update()
        screen_width, screen_height = self.root.winfo_width(), self.root.winfo_height()

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        xscrollbar = tk.Scrollbar(self.root, orient=tk.HORIZONTAL, width=22, bg="gray50")
        xscrollbar.grid(row=1, column=0, sticky=tk.E + tk.W)
        yscrollbar = tk.Scrollbar(self.root, width=22, bg="gray50")
        yscrollbar.grid(row=0, column=1, sticky=tk.N + tk.S)

        # Control
        cntrl_bg = "#ffd7b5"
        cntrl_font = "Helvetica 14 bold italic"
        cntrl_font_light = "Helvetica 10 italic"

        self.control_frame = tk.Frame(self.root, bg=cntrl_bg, width=0.1 * screen_width, height=screen_height)
        self.control_frame.pack_propagate(False)
        self.control_frame.grid(row=0, column=2, rowspan=2)
        self.image_name_output = tk.Label(self.control_frame,
                                          text="Image: {}".format(self.img_list[self.current_img_counter]),
                                          bg=cntrl_bg, font=cntrl_font)
        self.image_name_output.pack(anchor="center", expand=1)
        self.img_counter_output = tk.Label(self.control_frame,
                                           text="Image: {0}/{1}".format(self.current_img_counter + 1,
                                                                        len(self.img_list)),
                                           bg=cntrl_bg, font=cntrl_font)
        self.img_counter_output.pack(anchor="center", expand=1)
        self.label_counter_output = tk.Label(self.control_frame, text="Labels: {}".format(len(self.current_labels)),
                                             bg=cntrl_bg, font=cntrl_font)
        self.label_counter_output.pack(anchor="center", expand=1)
        self.zoom_level_output = tk.Label(self.control_frame, text="Zoom: {}x".format(1),
                                          bg=cntrl_bg, font=cntrl_font)
        self.zoom_level_output.pack(anchor="center", expand=1)
        self.mouse_output = tk.Label(self.control_frame, text="x=0  y=0 \nx=0  y=0", bg=cntrl_bg, font=cntrl_font)
        self.mouse_output.pack(anchor="center", expand=1)
        cntrl_txt = "Save & Quit - Escape\nNext Image - Right Arrow\nPrevious Image - Left Arrow\n" \
                    "Create Box - Left Click and drag\nDelete Box - Right click"
        tk.Label(self.control_frame, text=cntrl_txt, bg=cntrl_bg, font=cntrl_font_light).pack(anchor='center', expand=1)

        self.canvas = tk.Canvas(self.root, width=screen_width * 0.9, height=screen_height, bg="black",
                                xscrollcommand=xscrollbar.set,
                                yscrollcommand=yscrollbar.set, xscrollincrement=10, yscrollincrement=10)
        self.canvas.grid(row=0, column=0, sticky=tk.N + tk.W)

        # Bindings
        self.canvas.bind("<Button-1>", self.add_startpoint)
        self.canvas.bind("<Button-3>", self.delete_closest)
        self.canvas.bind("<B1-Motion>", self.rect_move)
        self.canvas.bind("<ButtonRelease-1>", self.rect_release)
        self.canvas.bind("<Button-2>", self.zoom)

        self.root.bind("<Shift-Button-4>", lambda event, direction=-1: self._scroll_x(event, direction))
        self.root.bind("<Shift-Button-5>", lambda event, direction=1: self._scroll_x(event, direction))
        self.root.bind("<Button-4>", lambda event, direction=-1: self._scroll_y(event, direction))
        self.root.bind("<Button-5>", lambda event, direction=1: self._scroll_y(event, direction))
        self.root.bind("<Escape>", self.quit)
        self.root.bind("<Motion>", self.track_mouse)

        self.root.bind("<Right>", self.next_image)
        self.root.bind("<Left>", self.prev_image)

        self.image_swich()

        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))
        xscrollbar.config(command=self.canvas.xview)
        yscrollbar.config(command=self.canvas.yview)

        self.root.mainloop()

    def _scroll_y(self, event, direction):
        self.canvas.yview_scroll(direction * 10, "units")

    def _scroll_x(self, event, direction):
        self.canvas.xview_scroll(direction * 10, "units")

    def get_current_image(self):
        return Image.open(self.img_path + "/" + self.img_list[self.current_img_counter])

    def zoom(self, event):
        self.zoom_mode = (self.zoom_mode * 2) % 7
        self.refresh(scale=self.zoom_mode, px=event.x, py=event.y)

    def ocoord(self, coord):
        """ Return the correct coordinate in zoom mode: canvas coord -> image coord"""
        return int(coord / self.zoom_mode)

    def ccoord(self, coord):
        return int(coord * self.zoom_mode)

    def load_img(self):
        current_img = Image.open(self.img_path + "/" + self.img_list[self.current_img_counter])
        self.width, self.height = current_img.size
        self.img = ImageTk.PhotoImage(current_img)
        self.img_zoomed2 = ImageTk.PhotoImage(current_img.resize((self.width * 2, self.height * 2)))
        self.img_zoomed4 = ImageTk.PhotoImage(current_img.resize((self.width * 4, self.height * 4)))

    def refresh(self, scale=1, px=0, py=0):
        self.canvas.delete('rect')

        if self.img_id:
            self.canvas.delete(self.img_id)

        img_name = self.img_list[self.current_img_counter]
        valid, img_db_id = self.db.image_id_by_filename(img_name)
        if valid:
            for x1, y1, x2, y2 in self.current_labels.values():
                self.canvas.create_rectangle(x1 * self.zoom_mode, y1 * self.zoom_mode,
                                             x2 * self.zoom_mode, y2 * self.zoom_mode,
                                             outline="red", width=3, activefill="red", tag="rect")

        choose_img = self.img
        if scale == 2:
            choose_img = self.img_zoomed2
        elif scale == 4:
            choose_img = self.img_zoomed4

        self.img_id = self.canvas.create_image(0, 0, image=choose_img, anchor=tk.N + tk.W)
        self.canvas.tag_lower(self.img_id)

        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))
        self.canvas.xview_moveto(px / self.width)
        self.canvas.yview_moveto(py / self.height)

        self.zoom_level_output.config(text="Zoom: {}x".format(scale))

    def image_swich(self):
        self.zoom_mode = 1
        self.load_img()
        self.canvas.delete('rect')

        if self.img_id:
            self.canvas.delete(self.img_id)

        img_name = self.img_list[self.current_img_counter]
        valid, img_db_id = self.db.image_id_by_filename(img_name)
        if valid:
            for x1, y1, x2, y2 in self.db.label_coords_by_image_id(img_db_id):
                rect_id = self.canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=3, activefill="red",
                                                       tag="rect")
                self.current_labels[rect_id] = (x1, y1, x2, y2)

        self.img_id = self.canvas.create_image(0, 0, image=self.img, anchor=tk.N + tk.W)
        self.canvas.tag_lower(self.img_id)

        self.img_counter_output.config(text="Image: {0}/{1}".format(self.current_img_counter + 1, len(self.img_list)))
        self.image_name_output.config(text="Image: {}".format(self.img_list[self.current_img_counter]))
        self.label_counter_output.config(text="Labels: {}".format(len(self.current_labels)))
        self.zoom_level_output.config(text="Zoom: 1x")

        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

        # zoom at begin
        self.zoom_mode = 2
        self.refresh(scale=self.zoom_mode, px=0, py=0)

    def get_curr_screen_geometry(self):
        root = tk.Tk()
        root.update_idletasks()
        root.attributes('-fullscreen', True)
        root.state('iconic')
        geometry = root.winfo_geometry()
        root.destroy()
        return geometry

    def add_startpoint(self, event):
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        self.first_point = (x, y)

    def rect_move(self, event):
        if self.last_rect:
            self.canvas.delete(self.last_rect)
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        self.last_rect = self.canvas.create_rectangle(self.first_point[0], self.first_point[1], x, y, outline="red",
                                                      width=3, activefill="red", tag="rect")

    def rect_release(self, event):
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        x1, y1, x2, y2 = self.ocoord(self.first_point[0]), self.ocoord(self.first_point[1]), self.ocoord(
            x), self.ocoord(y)
        if x1 > x2 and y1 > y2:
            x1, y1, x2, y2 = x2, y2, x1, y1
        elif x1 > x2 and y1 < y2:
            x1, x2 = x2, x1
        elif x1 < x2 and y1 > y2:
            y1, y2 = y2, y1
        print(x1, y1, x2, y2)

        if x1 < 0 or y1 < 0 or x2 > self.width or y2 > self.height:
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(x2, self.width), min(y2, self.height)

            self.canvas.create_rectangle(self.ccoord(x1), self.ccoord(y1), self.ccoord(x2), self.ccoord(y2),
                                         outline="red", width=3, activefill="red", tag="rect")

        self.current_labels[self.last_rect] = (x1, y1, x2, y2)
        self.last_rect = None
        self.first_point = None
        self.label_counter_output.config(text="Labels: {}".format(len(self.current_labels)))

    def delete_closest(self, event):
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        closest_obj = self.canvas.find_overlapping(x - 4, y - 4, x + 4, y + 4)
        if len(closest_obj) > 1:
            c = self.canvas.coords(closest_obj[1])
            self.canvas.delete(closest_obj[1])
            to_del = []
            for idc, coords in self.current_labels.items():
                if self.ocoord(c[0]) == coords[0] and self.ocoord(c[1]) == coords[1] and self.ocoord(c[2]) == coords[2] \
                        and self.ocoord(c[3]) == coords[3]:
                    to_del.append(idc)
            for i in to_del:
                del self.current_labels[i]
            self.label_counter_output.config(text="Labels: {}".format(len(self.current_labels)))

    def next_image(self, event):
        if self.current_img_counter < len(self.img_list) - 1:
            self.save_curr_in_db()
            self.current_img_counter += 1
            self.image_swich()
        else:
            print("No more image_session1!")

    def prev_image(self, event):
        if not self.current_img_counter == 0:
            self.save_curr_in_db()
            self.current_img_counter -= 1
            self.image_swich()
        else:
            print("Thats the first image!")

    def save_curr_in_db(self):
        img_name = self.img_list[self.current_img_counter]
        self.finished_img.append(img_name)
        valid, img_id = self.db.image_id_by_filename(img_name)
        if valid:
            self.db.delete_labels_by_image_id(img_id)
        else:
            img_id = self.db.add_image(img_name)

        for rect in self.current_labels.values():
            self.db.add_label(img_id, rect[0], rect[1], rect[2], rect[3])

        self.current_labels = {}

    def quit(self, event):
        self.save_curr_in_db()
        self.root.destroy()
        if self.cp:
            self.copy_files_to_training()
            print("Copied {} files to the training folder.".format(len(self.finished_img)))

    def copy_files_to_training(self):
        for img in self.finished_img:
            copy(os.path.join(self.img_path, img), os.path.join(self.training_path, img))

    def track_mouse(self, event):
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        self.mouse_output.config(text="x={0}  y={1} \nx={2}  y={3}".format(event.x, event.y, cx, cy))


if __name__ == '__main__':
    image_path = "/media/t/Demo"
    database_path = "newDB.db"
    start_point = 0

    LabelGUI(image_path, database_path, start=start_point)
