TAN_THETA = 0.38559092585596155
FOCAL_LENGTH = -0.5466257668711592
DISPARITY = 21  # in centimeters


def estimate_distance(obj_width_pixels, frame_width_pixels):
    global TAN_THETA, FOCAL_LENGTH

    distance: int = round(((frame_width_pixels * DISPARITY) / (2 * obj_width_pixels * TAN_THETA)) + FOCAL_LENGTH, 2)

    return str(f"dist: {distance} cm")

    
