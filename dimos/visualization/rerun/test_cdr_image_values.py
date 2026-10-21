# Copyright 2025-2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pytest
import rerun as rr

from dimos.messages.builtin_interfaces.msg import Time
from dimos.messages.std_msgs.msg import Header
from dimos.msgs.image import (
    compressed_image_from_image,
    image_from_array,
    image_from_compressed,
    image_view,
)
from dimos.msgs.time import time_from_seconds
from dimos.visualization.rerun.cdr import image_archetype


@pytest.fixture()
def rgb_image():
    rng = np.random.RandomState(42)
    gradient = np.linspace(0, 255, 640, dtype=np.uint8)
    data = np.broadcast_to(gradient, (480, 640)).copy()
    data = np.stack([data, data // 2, rng.randint(0, 50, (480, 640), dtype=np.uint8)], axis=-1)
    return image_from_array(
        data, encoding="rgb8", header=Header(frame_id="cam", stamp=time_from_seconds(1234.5678))
    )


def test_to_rerun_is_encoded_image(rgb_image) -> None:
    assert isinstance(image_archetype(compressed_image_from_image(rgb_image)), rr.EncodedImage)


@pytest.mark.parametrize("codec", ["png", "jxl"])
def test_compressed_uint16_depth_keeps_units_and_rerun_depth(codec: str) -> None:
    pixels = np.arange(80, dtype=np.uint16).reshape(10, 8)
    source = image_from_array(
        pixels, encoding="16UC1", header=Header(frame_id="depth", stamp=Time(sec=0, nanosec=0))
    )
    compressed = compressed_image_from_image(source, format=codec)
    decoded = image_from_compressed(compressed)
    assert decoded.encoding == "16UC1"
    assert decoded.header == source.header
    np.testing.assert_array_equal(image_view(decoded), pixels)
    assert isinstance(image_archetype(compressed), rr.DepthImage)
