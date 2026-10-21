# Copyright 2026 Dimensional Inc.
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


from pathlib import Path
import re

import numpy as np

from dimos.msgs.image import image_to_rgb


def test_documented_sensor_blocks_run_as_written() -> None:
    guide = Path(__file__).resolve().parents[3] / "docs/usage/visualization.md"
    section = guide.read_text().split("## Arrays and compression for generated sensors\n", 1)[1]
    blocks = re.findall(r"```python\n(.*?)```", section, re.DOTALL)
    assert len(blocks) == 3
    namespace = {}
    for block in blocks:
        exec(compile(block, str(guide), "exec"), namespace)
    np.testing.assert_allclose(
        namespace["xyz"], [[-0.25, -0.25, 1], [0.25, -0.25, 1], [-0.25, 0.25, 1], [0.25, 0.25, 1]]
    )
    np.testing.assert_array_equal(image_to_rgb(namespace["restored"]), namespace["rgb"])
    assert namespace["cloud"].header == namespace["image"].header
    assert not namespace["pixels"].flags.writeable
