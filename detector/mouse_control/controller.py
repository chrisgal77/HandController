from typing import Tuple
import pyautogui


class Controller:
    def __init__(self):
        self.width, self.height = pyautogui.size()

    def move(self, point: Tuple[int, int]) -> None:
        x, y = point
        pyautogui.moveTo(self.width * x, self.height * y)

    def click(self) -> None:
        pyautogui.click()

    def action(self, action: int, point: Tuple[int, int]) -> None:
        if action == 1:
            pass
            #self.move(point)
        elif action != 1:
            pass
            #self.click()