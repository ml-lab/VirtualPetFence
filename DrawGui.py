import cv2
import numpy as np


class DrawArea():
    def __init__(self):
        self.drawing = False  # true if mouse is pressed
        self.ix, self.iy = -1, -1
        self.mode = False  # if True, draw rectangle. Press 'm' to toggle to curve

    # mouse callback function
    def draw_circle(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing == True:
                if self.mode == True:
                    cv2.rectangle(self.img, (self.ix, self.iy), (x, y), (0, 255, 0), -1)
                else:
                    cv2.circle(self.img, (x, y), 5, (0, 0, 255), -1)
                cv2.imshow('image', self.img)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if self.mode == True:
                cv2.rectangle(self.img, (self.ix, self.iy), (x, y), (0, 255, 0), -1)
            else:
                cv2.circle(self.img, (x, y), 5, (0, 0, 255), -1)

    def doDraw(self):
        cap = cv2.VideoCapture(0)
        rval, self.img = cap.read()
        cv2.namedWindow('cam')
        cv2.setMouseCallback('cam', self.draw_circle)
        cv2.imshow('cam', self.img)

        while (True):
            # cv2.imshow('image', self.img)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('m'):
                self.mode = not self.mode
            elif k == 27:
                break
        cv2.destroyAllWindows()
        return np.logical_and((self.img[:, :, :2]).sum(2) == 0, self.img[:, :, 2] == 255)


if __name__=="__main__":
    draw = DrawArea()
    drawing = draw.doDraw()
    print drawing
