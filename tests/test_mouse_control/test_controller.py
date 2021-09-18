import pyautogui
from unittest import mock

from detector.mouse_control import Controller


class TestController:
    @mock.patch("detector.mouse_control.controller.pyautogui.moveTo")
    def test_move(self, mock_moveTo):
        x, y = 3, 4
        pyautogui.moveTo(x, y)
        mock_moveTo.assert_called_with(x, y)

    @mock.patch("detector.mouse_control.controller.pyautogui.click")
    def test_click(self, mock_click):
        pyautogui.click()
        mock_click.assert_called()

    @mock.patch("detector.mouse_control.controller.Controller.move")
    def test_action_move(self, mock_moveTo):
        action, point = 0, (3, 4)
        controller = Controller()
        controller.action(action, point)

        mock_moveTo.assert_called_with(point)

    @mock.patch("detector.mouse_control.controller.Controller.click")
    def test_action_click(self, mock_click):
        action, point = 1, (3, 4)
        controller = Controller()
        controller.action(action, point)

        mock_click.assert_called()
