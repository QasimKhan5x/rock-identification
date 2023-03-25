import tkinter as tk

import cv2
from PIL import Image, ImageTk

from draw_widths import draw_widths
from ImageUploaderGUI import ImageUploaderGUI


class DrawLinesGUI(tk.Frame):
    def __init__(self, master: tk.Frame, image_uploader: ImageUploaderGUI):
        super().__init__(master)
        self.master = master
        self.pack(side="right", fill="both", expand=True)
        
        self.image_uploader = image_uploader

        # create morp1 and morp2 buttons
        self.morp1 = tk.BooleanVar()
        self.morp2 = tk.BooleanVar()
        self.morp1_button = tk.Checkbutton(self, text="morp1", variable=self.morp1)
        self.morp2_button = tk.Checkbutton(self, text="morp2", variable=self.morp2)
        self.morp1_button.grid(row=0, column=0)
        self.morp2_button.grid(row=0, column=1)

        # create entry forms and labels
        self.min_label = tk.Label(self, text="min")
        self.min_label.grid(row=1, column=0)
        self.min_entry = tk.Entry(self)
        self.min_entry.insert(0, "0")
        self.min_entry.grid(row=2, column=0)

        self.max_label = tk.Label(self, text="max")
        self.max_label.grid(row=3, column=0)
        self.max_entry = tk.Entry(self)
        self.max_entry.insert(0, "100")
        self.max_entry.grid(row=4, column=0)

        self.diff_label = tk.Label(self, text="min difference")
        self.diff_label.grid(row=5, column=0)
        self.diff_entry = tk.Entry(self)
        self.diff_entry.insert(0, "5")
        self.diff_entry.grid(row=6, column=0)

        self.vgd_label = tk.Label(self, text="vertical grouping distance")
        self.vgd_label.grid(row=7, column=0)
        self.vgd_entry = tk.Entry(self)
        self.vgd_entry.insert(0, "2")
        self.vgd_entry.grid(row=8, column=0)

        # create draw lines button
        self.draw_button = tk.Button(self, text="draw lines", command=self.draw_lines)
        self.draw_button.grid(row=9, column=0, pady=10)

        # create sample image
        self.sample_image = tk.Label(self)
        self.sample_image.grid(row=10, column=0, padx=5, pady=5, columnspan=2)
        
        # create list to store generated images
        self.images = []
        # index of currently displayed image
        self.curr_img_index = 0

        # create back and next buttons
        self.back_button = tk.Button(self, text="Back", command=self.show_previous_image)
        self.next_button = tk.Button(self, text="Next", command=self.show_next_image)
        

    def draw_lines(self):
        self.images.clear()
        # perform task using previously uploaded images and calculated ratio
        # create new image and display it using self.sample_image.config(image=new_image)
        image = self.image_uploader.image_cv2
        mask = self.image_uploader.mask_cv2
        ratio = self.image_uploader.ratio
        
        annotated_images = draw_widths(image, mask, ratio=ratio,
                                       morph1=self.morp1.get(), morph2=self.morp2.get(), 
                                       w_min=int(self.min_entry.get()), w_max=int(self.max_entry.get()),
                                       line_sim_thresh=int(self.diff_entry.get()), 
                                       group_y_range=int(self.vgd_entry.get()))
        
        for img in annotated_images:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype('uint8')
            img = Image.fromarray(img)
            photo = ImageTk.PhotoImage(img)
            self.images.append(photo)
        
        if len(self.images) > 0:
            # display first image
            self.sample_image.config(image=self.images[0])
            # show back and next buttons
            self.back_button.grid(row=11, column=0, pady=10)
            self.next_button.grid(row=11, column=1, pady=10)

    def show_previous_image(self):
        self.curr_img_index = (self.curr_img_index - 1) % len(self.images)
        self.sample_image.config(image=self.images[self.curr_img_index])

    def show_next_image(self):
        self.curr_img_index = (self.curr_img_index + 1) % len(self.images)
        self.sample_image.config(image=self.images[self.curr_img_index])
