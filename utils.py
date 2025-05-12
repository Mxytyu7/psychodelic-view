import numpy as np
import cv2

def perlin_noise(height, width, scale):
    """
    Simple 2D Perlin-ish noise generator.
    Right now this is a placeholder using sin/cos pattern.
    """
    linx = np.linspace(0, scale, width, endpoint=False)
    liny = np.linspace(0, scale, height, endpoint=False)

    x_coords, y_coords = np.meshgrid(linx, liny)

    # This is more of a wavy pattern, not real Perlin noise yet.
    noise_map = np.sin(x_coords) * np.cos(y_coords)

    # TODO: Replace with actual Perlin noise algorithm someday
    return noise_map

def compute_optical_flow(prev, curr):
    """
    Compute dense optical flow between two grayscale frames.
    """
    # Sanity check: frames should be same size
    if prev.shape != curr.shape:
        raise ValueError("Input frames must have the same dimensions.")

    # Parameters for Farneback algorithm
    pyr_scale = 0.5
    levels = 3
    winsize = 15
    iterations = 3
    poly_n = 5
    poly_sigma = 1.2
    flags = 0

    flow = cv2.calcOpticalFlowFarneback(prev, curr,
                                        None,
                                        pyr_scale, levels, winsize,
                                        iterations, poly_n, poly_sigma, flags)

    # Optional alternative params (felt smoother but slower)
    # flow = cv2.calcOpticalFlowFarneback(prev, curr,
    #                                     None, 0.3, 5, 25, 5, 7, 1.5, 0)

    return flow
