import cv2
import time


class MotionDetector:

    def __init__(self, threshold=25):

        self.previous_frame = None
        self.threshold = threshold


    def detect(self, frame):

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.GaussianBlur(
            gray,
            (21, 21),
            0
        )


        if self.previous_frame is None:

            self.previous_frame = gray

            return False


        difference = cv2.absdiff(
            self.previous_frame,
            gray
        )


        _, threshold_frame = cv2.threshold(
            difference,
            self.threshold,
            255,
            cv2.THRESH_BINARY
        )


        threshold_frame = cv2.dilate(
            threshold_frame,
            None,
            iterations=2
        )


        contours, _ = cv2.findContours(
            threshold_frame,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )


        self.previous_frame = gray


        for contour in contours:

            area = cv2.contourArea(contour)

            if area > 1500:

                return True


        return False