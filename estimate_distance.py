TAN_THETA = 0.38559092585596155
FOCAL_LENGTH = -0.5466257668711592
BASELINE = 21  # in centimeters


def calc_centroids(obj_bbox, matched_obj_bbox):
    obj_cx = (2*obj_bbox[0]+obj_bbox[2]) / 2
    obj_cy = (obj_bbox[1]+obj_bbox[3]) / 2

    matched_obj_cx = (2*matched_obj_bbox[0] + matched_obj_bbox[2]) / 2
    matched_obj_cy = (matched_obj_bbox[1] + matched_obj_bbox[3]) / 2

    return obj_cx - matched_obj_cx


def estimate_distance(obj_width_pixels, frame_width_pixels):
    global TAN_THETA, FOCAL_LENGTH

    distance: int = round(((frame_width_pixels * BASELINE) / (2 * obj_width_pixels * TAN_THETA)) + FOCAL_LENGTH, 2)

    print(f"distance is: {distance} cm")

    return str(f"dist: {distance} cm")

    
