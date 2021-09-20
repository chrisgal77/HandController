from typing import Tuple
import pyautogui


class Controller:
    def __init__(self):
        self.width, self.height = pyautogui.size()

    def move(self, point: Tuple[int, int]) -> None:
        x, y = point
        pyautogui.moveTo(self.width * x / 480, self.height * y / 360)

    def click(self) -> None:
        pyautogui.click()

    def action(self, action: int, point: Tuple[int, int]) -> None:
        if action == 0:
            print('move')
            self.move(point)
        elif action == 1:
            print('click')
            #self.click()