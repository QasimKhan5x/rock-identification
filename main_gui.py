import os
import tkinter as tk
from tkinter import ttk

from DrawLinesGUI import DrawLinesGUI
from ImageUploaderGUI import ImageUploaderGUI
from util import resource_path


class MainWindow:
    def __init__(self, master):
        self.master = master
        master.title("Rock Width Identification")
        path = resource_path("assets")
        path = os.path.join(path, "rock.ico")
        master.iconbitmap(path)

        # Create styled frames
        style = ttk.Style()
        style.configure(
            "LeftFrame.TFrame", background="#F0F0F0", borderwidth=2, relief="raised"
        )
        style.configure(
            "RightFrame.TFrame", background="#F0F0F0", borderwidth=2, relief="raised"
        )

        left_frame = ttk.Frame(master, style="LeftFrame.TFrame", width=400, height=400)
        left_frame.grid(row=0, column=0, sticky="nsew")

        right_frame = ttk.Frame(
            master, style="RightFrame.TFrame", width=400, height=400
        )
        right_frame.grid(row=0, column=1, sticky="nsew")

        # place ImageUploaderGUI widget in left frame
        self.image_uploader = ImageUploaderGUI(left_frame)
        self.lines_gui = DrawLinesGUI(right_frame, self.image_uploader)


if __name__ == "__main__":
    root = tk.Tk()
    app = MainWindow(root)
    root.mainloop()
