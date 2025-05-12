import tkinter as tk
import turtle
from PIL import Image, ImageTk

class PsycheGUI:
    def __init__(self, params, on_change_callback):
        """
        params: dict(name -> (min, max, init))
        on_change_callback: callback(name, value)
        """
        # === Setup Turtle screen and canvas ===
        self.screen = turtle.Screen()
        self.screen.title("Psyche Viewer")

        # Use a hidden turtle for raw drawing commands
        self.drawer = turtle.Turtle(visible=False)
        self.drawer.speed(0)

        # Turn off auto screen updates for better FPS
        turtle.tracer(0, 0)

        # Grab the Tkinter canvas from turtle
        self.canvas = self.screen.getcanvas()
        self.root = self.canvas.winfo_toplevel()

        # === Sliders for each parameter ===
        self.sliders = {}    # Track slider vars
        self.slider_widgets = {}  # Track actual slider widgets, might be useful later
        for param_name, (min_val, max_val, init_val) in params.items():
            # Add a label for each slider
            label = tk.Label(self.root, text=param_name)
            label.pack()

            var = tk.DoubleVar(value=init_val)
            slider = tk.Scale(
                self.root,
                from_=min_val,
                to=max_val,
                orient=tk.HORIZONTAL,
                variable=var,
                command=lambda v, n=param_name: on_change_callback(n, float(v))  # capture current param
            )
            slider.pack(fill="x")

            # Store variable and widget
            self.sliders[param_name] = var
            self.slider_widgets[param_name] = slider

        # Just for reference, store current image drawn (avoids garbage collection)
        self.current_photo = None

    def pil_image_from_array(self, array):
        """
        Convert a NumPy array (H×W×3) into a Tkinter PhotoImage.
        """
        pil_img = Image.fromarray(array)
        photo_img = ImageTk.PhotoImage(pil_img)
        return photo_img

    def draw_image(self, photo_img):
        """
        Draw given PhotoImage onto the canvas.
        """
        # Clear everything first (kinda brute-force but works)
        self.canvas.delete("all")

        # Draw at top-left (0,0)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo_img)

        # Keep a reference — Tkinter will otherwise drop the image!
        self.current_photo = photo_img
        # self.canvas.image = photo_img  # old way, but keeping self.current_photo now
