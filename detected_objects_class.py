from globals import IMG_DIMENSIONS


class DetectedObjectClass:
    _TAN_THETA = 0.38559092585596155
    _FOCAL_LENGTH = -0.5466257668711592
    _BASELINE = 3.0  # in centimeters

    detected_object_name: str
    detected_object_disparity: float
    detected_object_distance: float
    detected_object_conf: float
    detected_object_bbox: tuple

    def __init__(self, detected_obj_properties, matched_obj_bbox):
        """
            Initialize a DetectedObjectClass instance.

            Args:
                detected_obj_properties (tuple): A tuple containing (confidence, name, bounding_box).
                matched_obj_bbox (tuple): A tuple containing the bounding box of the matched object.
        """

        self.detected_object_name = detected_obj_properties[2]
        self.detected_object_conf = detected_obj_properties[0]
        self.detected_object_bbox = detected_obj_properties[1]

        # Compute centroid of detected object and matched object
        obj_bbox = self.detected_object_bbox
        obj_cx = (2 * obj_bbox[0] + obj_bbox[2]) / 2
        matched_obj_cx = (2 * matched_obj_bbox[0] + matched_obj_bbox[2]) / 2

        self.detected_object_disparity = obj_cx - matched_obj_cx

        distance: int = round(((IMG_DIMENSIONS[0] * self._BASELINE) / (2 * self.detected_object_disparity *
                                                                       self._TAN_THETA)) + self._FOCAL_LENGTH, 2)
        # print(f"distance is: {distance} cm")
        self.detected_object_distance = abs(distance)
