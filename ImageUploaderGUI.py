import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np
from PIL import Image, ImageTk

from get_scale_ratio import get_pixel_length_ratio


class ImageUploaderGUI:
    def __init__(self, master):
        self.master = master

        # Create buttons
        self.button1 = tk.Button(master, text="Upload Image", command=self.upload_image)
        self.button1.grid(row=0, column=0, padx=10, pady=10)
        self.button2 = tk.Button(master, text="Upload Mask", command=self.upload_mask)
        self.button2.grid(row=0, column=1, padx=10, pady=10)

        # Create image labels
        self.image1_label = tk.Label(master)
        self.image1_label.grid(row=1, column=0, padx=10, pady=10)
        self.image2_label = tk.Label(master)
        self.image2_label.grid(row=1, column=1, padx=10, pady=10)

        # Create button, entry form, and horizontal scale
        self.calculate_button = tk.Button(
            master, text="Calculate Ratio", command=self.calculate_ratio, width=15
        )
        self.calculate_button.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

        self.scale_actual_width_label = tk.Label(
            master, text="Enter Scale Actual Width:"
        )
        self.scale_actual_width_label.grid(row=3, column=0, padx=10, pady=5, sticky="e")

        self.morph_iterations_label = tk.Label(
            master, text="Number of Morphology Iterations (1-7):"
        )
        self.morph_iterations_label.grid(row=4, column=0, padx=10, pady=5, sticky="e")

        self.num_entry = tk.Entry(master, width=10)
        self.num_entry.grid(row=3, column=1, padx=10, pady=5, sticky="w")
        self.num_entry.insert(0, "15")

        self.scale = tk.Scale(
            master, from_=1, to=7, orient=tk.HORIZONTAL, resolution=1, length=200
        )
        self.scale.grid(row=4, column=1, padx=10, pady=5, sticky="w")

        self.result_image = tk.Label(master)
        self.result_image.grid(row=6, column=0, columnspan=2, padx=10, pady=10)

        # Initialize instance variables
        self.image1 = None
        self.image2 = None
        self.image3 = None
        # Read images using OpenCV
        self.image_cv2 = None
        self.image_gray_cv2 = None
        self.mask_cv2 = None

        # Calculated ratio of pixels to length
        self.ratio = None
        # label to display ratio
        self.ratio_label = None

    def upload_image(self):
        # get filename
        filename = filedialog.askopenfilename(title="Select Image")
        # open and resize image
        img = Image.open(filename)
        img = img.resize((256, 256))
        # convert image to opencv format
        self.image_cv2 = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        self.image_gray_cv2 = cv2.cvtColor(self.image_cv2, cv2.COLOR_BGR2GRAY)
        # update image label
        self.image1 = img
        self.image1_tk = ImageTk.PhotoImage(img)
        self.image1_label.config(image=self.image1_tk)

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

    def calculate_ratio(self):
        # check if both images are uploaded
        if self.image1 is None or self.image2 is None:
            return

        # get the ratio from the entry form and the scale
        actual_width = int(self.num_entry.get())
        num_iterations = self.scale.get()

        ratio, result = get_pixel_length_ratio(
            self.image_gray_cv2,
            self.mask_cv2,
            num_closing_iterations=num_iterations,  # type: ignore
            actual_width=actual_width,
            draw=True,
            return_drawn_img=True,
        )
        if ratio < 0:
            ratio = "failed"
        self.ratio = ratio

        # display ratio of pixels to length and result image
        if ratio == "failed":
            text = "Failed to calculate ratio. Please try another image."
            color = "red"
        else:
            text = f"Ratio: {ratio} pixels / cm"
            color = "black"

        if self.ratio_label is not None:
            self.ratio_label.config(text=text, fg=color)
        else:
            self.ratio_label = tk.Label(self.master, text=text, fg=color)
            self.ratio_label.grid(row=5, column=0, columnspan=2)

        result = Image.fromarray(result)
        self.image3 = ImageTk.PhotoImage(result)
        self.result_image.config(image=self.image3)


if __name__ == "__main__":
    root = tk.Tk()
    uploader = ImageUploaderGUI(root)
    root.mainloop()
