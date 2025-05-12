import cv2
from concurrent.futures import ThreadPoolExecutor

# Try to import VidGear for optional streaming
try:
    from vidgear.gears import CamGear
    _vidgear_enabled = True
except ImportError:
    _vidgear_enabled = False

# Effects modules
from effects import (
    HueSaturation,
    PerlinWarp,
    KaleidoscopeEffect,
    Pixelate,
    OpticalFlowWarp
)
from gui import PsycheGUI

class Application:
    def __init__(self, src=0, use_vidgear=False):
        # --- Setup video source (webcam or file) ---
        if use_vidgear and _vidgear_enabled:
            self.stream = CamGear(source=src).start()
        else:
            self.stream = cv2.VideoCapture(src)

        # --- Parameters for UI sliders ---
        # Each param: (min, max, default)
        self.param_meta = {
            "dosage":       (0, 500, 100),     # Hue/saturation dosage
            "hits":         (1, 50, 5),         # Perlin warp hits
            "tiles":        (1, 20, 4),         # Kaleidoscope tiles
            "pixel_size":   (2, 100, 10),       # Pixelate size
            "flow_strength":(0, 20, 5),         # Optical flow strength
        }

        # Active parameter values (copied from meta defaults)
        self.param_values = {}
        for name, (_, _, init_val) in self.param_meta.items():
            self.param_values[name] = init_val

        # --- GUI setup with sliders ---
        self.gui = PsycheGUI(self.param_meta, self._on_slider_change)

        # Thread pool so we can offload frame processing
        self.executor = ThreadPoolExecutor(max_workers=2)

        # Processing pipeline and control flag
        self.pipeline = []
        self.is_running = False  # Using 'is_running' feels clearer here

    def _on_slider_change(self, param_name, new_val):
        """
        Callback triggered when user moves a slider.
        """
        # Note: all sliders send back string-like values, so cast
        self.param_values[param_name] = float(new_val)

    def build_pipeline(self):
        """
        Construct (or rebuild) the effects pipeline based on current params.
        """
        # Fetch current param values
        dosage = self.param_values["dosage"]
        hits = int(self.param_values["hits"])  # Perlin warp expects int
        tiles = int(self.param_values["tiles"])
        pixel_size = int(self.param_values["pixel_size"])
        flow_strength = self.param_values["flow_strength"]

        # Rebuild pipeline list every time (might optimize later)
        self.pipeline = [
            HueSaturation(dosage),
            PerlinWarp(dosage / 100.0, hits),
            KaleidoscopeEffect(tiles),
            Pixelate(pixel_size),
            OpticalFlowWarp(flow_strength),
        ]

    def fetch_frame(self):
        """
        Grab a frame from the stream. Supports both OpenCV and VidGear.
        """
        # If it's a plain OpenCV stream
        if hasattr(self.stream, "read") and not _vidgear_enabled:
            ret, frame = self.stream.read()
            return frame if ret else None
        else:
            # CamGear read
            return self.stream.read()

    def process_frame(self, frame):
        """
        Run the frame through the pipeline of effects.
        """
        # Note: effects are applied in sequence
        processed = frame
        for fx in self.pipeline:
            processed = fx.apply(processed)
        return processed

    def run(self):
        """
        Start the main app loop: capture, process, display.
        """
        self.is_running = True

        def main_loop():
            if not self.is_running:
                return  # Stop the loop

            frame = self.fetch_frame()
            if frame is None:
                # Stream ended or error occurred
                self.stop()
                return

            # Rebuild pipeline every frame so sliders take effect live
            self.build_pipeline()

            # Offload heavy image processing to thread pool
            # (Note: calling .result() blocks but it's okay here)
            future = self.executor.submit(self.process_frame, frame)
            processed_frame = future.result()

            # Convert from BGR (OpenCV) to RGB (PIL expects RGB)
            img_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

            # Convert numpy array to PIL image
            photo_img = self.gui.pil_image_from_array(img_rgb)

            # Draw updated image in GUI
            self.gui.draw_image(photo_img)

            # Schedule next frame (approx every 33ms ≈ 30 FPS)
            self.gui.screen.ontimer(main_loop, 33)

        # Start looping
        main_loop()
        self.gui.screen.mainloop()  # Start GUI's event loop

    def stop(self):
        """
        Stop the app and clean up resources.
        """
        self.is_running = False

        # Stop video stream safely
        if hasattr(self.stream, "stop"):
            self.stream.stop()
        else:
            self.stream.release()

        # Shutdown thread pool
        self.executor.shutdown(wait=False)
