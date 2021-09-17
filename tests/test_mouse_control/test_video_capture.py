from unittest import mock
import pytest

from detector.mouse_control import VideoCapture
from detector.mouse_control.exceptions import FrameReadError


class Test_VideoCapture:
    @mock.patch("detector.mouse_control.VideoCapture.read")
    def test_call(self, mock_read):
        mock_read.return_value = (False, None)
        video_capture = VideoCapture(0)
        with pytest.raises(FrameReadError) as e:
            frame = video_capture()
            assert "Failed to read a frame" in str(e.value)
