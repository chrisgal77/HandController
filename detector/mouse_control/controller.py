import pyautogui


class Controller:
    def __init__(self):
        self.width, self.height = pyautogui.size()

    def move(self, point):
        x, y = point
        pyautogui.moveTo(self.width * x, self.height * y)

    @staticmethod
    def click(self):
        pyautogui.click()
