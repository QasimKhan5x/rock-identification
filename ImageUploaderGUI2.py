import math
import tkinter as tk
from tkinter import Entry, filedialog

import numpy as np
from PIL import Image, ImageTk


class ImageUploaderGUI:
    def __init__(self, master):
        self.master = master

        # Create buttons
        self.button1 = tk.Button(master, text="Upload Image", command=self.upload_image)
        self.button1.grid(row=0, column=0, padx=10, pady=10)
        self.button2 = tk.Button(master, text="Upload Mask", command=self.upload_mask)
        self.button2.grid(row=0, column=1, padx=10, pady=10)

        # Create canvas for image and mask
        self.canvas = tk.Canvas(master, width=256, height=256)
        self.canvas.grid(row=1, column=0, padx=10, pady=10)

        self.image2_label = tk.Label(master)
        self.image2_label.grid(row=1, column=1, padx=10, pady=10)

        # Create button, entry form, and horizontal scale
        self.draw_button = tk.Button(
            master, text="Draw", command=self.enter_drawing_mode, width=15
        )
        self.draw_button.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

        self.scale_actual_width_label = tk.Label(
            master, text="Enter Scale Actual Width:"
        )
        self.scale_actual_width_label.grid(row=3, column=0, padx=10, pady=5, sticky="e")

        self.scale_actual_width_entry = Entry(master)
        self.scale_actual_width_entry.grid(row=3, column=1, padx=10, pady=5, sticky="w")
        self.scale_actual_width_entry.insert(0, "15")

        # Initialize instance variables
        self.image1_tk = None
        self.image2_tk = None
        self.start_pos = None
        self.drawing = False
        self.line_length_label = tk.Label(master)
        self.line_length_label.grid(row=4, column=0, columnspan=2)

        # Calculated ratio of pixels to length
        self.ratio = None

        # Bind the mouse button click event to the on_mouse_click function
        self.canvas.bind("<Button-1>", self.on_mouse_click)

    def upload_image(self):
        # get filename
        filename = filedialog.askopenfilename(title="Select Image")
        # open and resize image
        img = Image.open(filename)
        img = img.resize((256, 256), Image.LANCZOS)
        # update image label
        self.image1_tk = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, image=self.image1_tk, anchor="nw")

    def upload_mask(self):
        # get filename
        filename = filedialog.askopenfilename(title="Select Mask")
        # open and resize image
        img = Image.open(filename)
        img = img.resize((256, 256))
        # convert image to opencv format
        self.mask_cv2 = np.asarray(img)
        # display image with colors
        min_value, max_value = self.mask_cv2.min(), self.mask_cv2.max()
        norm_mask = (self.mask_cv2 - min_value) / (max_value - min_value)
        norm_mask = (norm_mask * 255).astype("uint8")
        # update image label
        self.image2 = Image.fromarray(norm_mask)
        self.image2_tk = ImageTk.PhotoImage(self.image2)
        self.image2_label.config(image=self.image2_tk)

    def on_mouse_click(self, event):
        # If in drawing mode and the start position is None,
        # then this is the first click so save the start position
        if self.drawing and self.start_pos is None:
            self.start_pos = (event.x, event.y)
            # Draw a red dot at the start position with the "dot" tag
            self.canvas.create_oval(
                event.x - 3,
                event.y - 3,
                event.x + 3,
                event.y + 3,
                fill="red",
                tags="dot",
            )
        # If in drawing mode and the start position is already set,
        # then this is the second click, so draw the line and calculate its length
        elif self.drawing and self.start_pos is not None:
            # Add the line with the "line" tag
            self.canvas.create_line(
                self.start_pos[0],
                self.start_pos[1],
                event.x,
                event.y,
                fill="red",
                tags="line",
            )
            # Calculate and display line length
            line_length = math.sqrt(
                (self.start_pos[0] - event.x) ** 2 + (self.start_pos[1] - event.y) ** 2
            )
            scale_actual_width = float(self.scale_actual_width_entry.get())
            self.ratio = (
                line_length / scale_actual_width
                if scale_actual_width != 0
                else "Scale Actual Width can't be zero."
            )
            # self.line_length_label.config(text=f"Line length: {line_length:.2f}, Ratio: {ratio} pixels / cm")
            self.line_length_label.config(text=f"Ratio: {self.ratio:.2f} pixels / cm")
            # Reset the start position and exit drawing mode
            self.start_pos = None
            self.drawing = False

    # Function to enter drawing mode and clear existing lines
    def enter_drawing_mode(self):
        # Clear all existing lines and dots
        self.canvas.delete("line")
        self.canvas.delete("dot")
        self.line_length_label.config(text="")
        self.drawing = True


if __name__ == "__main__":
    root = tk.Tk()
    uploader = ImageUploaderGUI(root)
    root.mainloop()
