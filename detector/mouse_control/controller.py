import pyautogui


class Controller:
    def __init__(self, screen_shape):
        self.width, self.height = screen_shape

    def move(self, point):
        x, y = point
        pyautogui.moveTo(self.width * x, self.height * y)
