import cv2
import numpy as np
from utils import perlin_noise, compute_optical_flow

class HueSaturation:
    def __init__(self, dosage):
        self.dosage = dosage  # This controls how strong the hue shift will be

    def apply(self, frame):
        # Convert to HSV where hue/saturation are easier to manipulate
        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        h, s, v = cv2.split(hsv_img)

        # Shift hue and bump saturation slightly
        h = (h + self.dosage) % 180
        s = np.clip(s * (1 + self.dosage / 500.0), 0, 255)

        merged = cv2.merge([h, s, v]).astype(np.uint8)
        return cv2.cvtColor(merged, cv2.COLOR_HSV2BGR)

class PerlinWarp:
    def __init__(self, amplitude, frequency):
        self.amplitude = amplitude  # How much to warp pixels
        self.frequency = frequency  # Controls the noise frequency

    def apply(self, frame):
        h, w = frame.shape[:2]

        # Generate Perlin noise pattern
        noise_map = perlin_noise(h, w, self.frequency)

        # Create mapping coordinates for each pixel
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))

        # Apply noise-based displacement
        displaced_x = (grid_x + self.amplitude * noise_map).astype(np.float32)
        displaced_y = (grid_y + self.amplitude * noise_map).astype(np.float32)

        # Warp the image using remap
        warped = cv2.remap(frame, displaced_x, displaced_y,
                           interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT)
        return warped

class KaleidoscopeEffect:
    def __init__(self, tiles):
        self.tiles = tiles  # Controls how many mirrored sections appear

    def apply(self, frame):
        h, w = frame.shape[:2]

        # Get smallest dimension and compute tile size
        min_side = min(h, w)
        tile_size = max(1, min_side // self.tiles)

        # Crop out the top-left corner quad
        quad = frame[0:tile_size, 0:tile_size]

        # Make mirrored tiles
        top_half = np.concatenate([quad, cv2.flip(quad, 1)], axis=1)
        bottom_half = np.concatenate([cv2.flip(quad, 0), cv2.flip(quad, -1)], axis=1)

        # Stack top and bottom
        pattern = np.concatenate([top_half, bottom_half], axis=0)

        # Resize pattern back to full frame size
        return cv2.resize(pattern, (w, h), interpolation=cv2.INTER_LINEAR)

class Pixelate:
    def __init__(self, tile_size):
        self.tile_size = tile_size  # Each pixel block size

    def apply(self, frame):
        h, w = frame.shape[:2]

        # Calculate reduced size, avoiding zero
        small_w = max(1, w // self.tile_size)
        small_h = max(1, h // self.tile_size)

        # Downscale
        temp = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_LINEAR)

        # Upscale back (nearest neighbor for blocky look)
        pixelated = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
        return pixelated

class OpticalFlowWarp:
    def __init__(self, strength):
        self.strength = strength
        self.prev_gray = None  # We'll store previous grayscale frame here

    def apply(self, frame):
        # Convert to grayscale for optical flow computation
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # If it's the first frame, just store and return as-is
        if self.prev_gray is None:
            self.prev_gray = gray
            return frame

        # Compute dense optical flow
        flow = compute_optical_flow(self.prev_gray, gray)

        self.prev_gray = gray  # Update previous frame

        h, w = gray.shape
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))

        # Apply flow vectors scaled by strength
        map_x = (grid_x + self.strength * flow[..., 0]).astype(np.float32)
        map_y = (grid_y + self.strength * flow[..., 1]).astype(np.float32)

        warped = cv2.remap(frame, map_x, map_y,
                           interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT)

        # Debugging: Uncomment to visualize flow
        # cv2.imshow("flow mag", np.linalg.norm(flow, axis=2) / 10)

        return warped
