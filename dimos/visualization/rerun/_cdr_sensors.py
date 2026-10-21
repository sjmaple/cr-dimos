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

"""Read generated sensor buffers for Rerun display."""

from typing import Any

import imagecodecs
import numpy as np
from numpy.typing import NDArray

from dimos.messages.builtin_interfaces.msg import Time
from dimos.messages.sensor_msgs.msg import CompressedImage, Image, PointCloud2
from dimos.messages.std_msgs.msg import Header

_FORMATS = {
    "rgb8": ("u1", 3),
    "bgr8": ("u1", 3),
    "rgba8": ("u1", 4),
    "bgra8": ("u1", 4),
    "mono8": ("u1", 1),
    "mono16": ("u2", 1),
    "16UC1": ("u2", 1),
    "32FC1": ("f4", 1),
    "64FC1": ("f8", 1),
    "16SC1": ("i2", 1),
    "32FC3": ("f4", 3),
}


def _image_from_array(
    pixels: NDArray[Any], *, encoding: str, header: Header | None = None
) -> Image:
    """Copy an array into a generated image with an explicit pixel encoding."""
    if encoding not in _FORMATS:
        raise ValueError(f"unsupported image encoding {encoding!r}")
    kind, channels = _FORMATS[encoding]
    expected = np.dtype(kind)
    if pixels.dtype.kind != expected.kind or pixels.dtype.itemsize != expected.itemsize:
        raise ValueError(f"array dtype does not match {encoding}")
    if (channels == 1 and pixels.ndim != 2) or (
        channels != 1 and (pixels.ndim != 3 or pixels.shape[2] != channels)
    ):
        raise ValueError(f"array shape does not match {encoding}")
    contiguous = np.array(pixels, copy=True, order="C")
    return Image(
        header=header if header is not None else Header(stamp=Time(sec=0, nanosec=0), frame_id=""),
        width=pixels.shape[1],
        height=pixels.shape[0],
        encoding=encoding,
        is_bigendian=int(not pixels.dtype.isnative if np.little_endian else pixels.dtype.isnative),
        step=pixels.shape[1] * channels * pixels.dtype.itemsize,
        data=np.asarray(contiguous.view(np.uint8).ravel(), dtype=np.uint8),
    )


def image_view(msg: Image) -> NDArray[Any]:
    """Borrow a read-only array, retaining the message and respecting row padding/endian."""
    if msg.encoding not in _FORMATS:
        raise ValueError(f"unsupported image encoding {msg.encoding!r}")
    kind, channels = _FORMATS[msg.encoding]
    dtype = np.dtype((">" if msg.is_bigendian else "<") + kind)
    row_bytes = msg.width * channels * dtype.itemsize
    if msg.step < row_bytes or len(msg.data) != msg.height * msg.step:
        raise ValueError("image data length/step does not match its dimensions")
    shape = (msg.height, msg.width) if channels == 1 else (msg.height, msg.width, channels)
    strides = (
        (msg.step, dtype.itemsize)
        if channels == 1
        else (msg.step, channels * dtype.itemsize, dtype.itemsize)
    )
    view = np.ndarray(shape, dtype=dtype, buffer=msg.data, strides=strides)
    view.setflags(write=False)
    return view


def image_from_compressed(message: CompressedImage) -> Image:
    """Decode JPEG/PNG or the JXL extension into explicitly encoded generated pixels."""
    import cv2

    format_name = message.format.lower()
    if "jxl" in format_name:
        jxl_pixels = imagecodecs.jpegxl_decode(bytes(message.data))
        if jxl_pixels.ndim == 2:
            encoding = {
                np.dtype("float32"): "32FC1",
                np.dtype("uint16"): "mono16",
                np.dtype("uint8"): "mono8",
            }.get(jxl_pixels.dtype)
        elif jxl_pixels.ndim == 3 and jxl_pixels.shape[2] in (3, 4):
            encoding = "rgb8" if jxl_pixels.shape[2] == 3 else "rgba8"
        else:
            encoding = None
        if encoding is None:
            raise ValueError("unsupported decoded JXL layout")
        if encoding == "mono16" and format_name.startswith("16uc1;"):
            encoding = "16UC1"
        return _image_from_array(jxl_pixels, encoding=encoding, header=message.header)
    if not any(name in format_name for name in ("jpeg", "jpg", "png")):
        raise ValueError(f"unsupported compressed image format {message.format!r}")
    pixels = cv2.imdecode(np.frombuffer(bytes(message.data), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if pixels is None:
        raise ValueError("invalid compressed image data")
    if pixels.ndim == 2:
        encoding = "mono16" if pixels.dtype.itemsize == 2 else "mono8"
    elif pixels.ndim == 3 and pixels.shape[2] == 3:
        encoding = "bgr8"
    elif pixels.ndim == 3 and pixels.shape[2] == 4:
        encoding = "bgra8"
    else:
        raise ValueError("unsupported decoded image layout")
    if encoding == "mono16" and format_name.startswith("16uc1;"):
        encoding = "16UC1"
    return _image_from_array(pixels, encoding=encoding, header=message.header)


def image_to_rgb(message: Image) -> NDArray[np.uint8]:
    """Copy a display image as RGB8; 16-bit grayscale uses its high byte.

    Alpha is discarded. Floating-point depth needs an explicit visualization
    scale and is rejected rather than assigning pixel values to meters.
    """
    import cv2

    pixels = image_view(message)
    if message.encoding == "rgb8":
        return pixels.copy()
    if message.encoding in ("mono16", "16UC1"):
        return np.asarray(
            cv2.cvtColor((pixels / 256).astype(np.uint8), cv2.COLOR_GRAY2RGB), dtype=np.uint8
        )
    codes = {
        "bgr8": cv2.COLOR_BGR2RGB,
        "rgba8": cv2.COLOR_RGBA2RGB,
        "bgra8": cv2.COLOR_BGRA2RGB,
        "mono8": cv2.COLOR_GRAY2RGB,
    }
    if message.encoding not in codes:
        raise ValueError(f"Cannot convert {message.encoding!r} to RGB8")
    return np.asarray(cv2.cvtColor(pixels, codes[message.encoding]), dtype=np.uint8)


_FIELD_KINDS = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}


def pointcloud_view(message: PointCloud2) -> NDArray[Any]:
    """Borrow read-only structured (height, width) points, including row/field padding.

    Fields retain their declared counts and endianness. The returned array keeps
    the message's storage alive. Use ``view.copy()`` for independent mutable data.
    """
    if message.width and not message.point_step:
        raise ValueError("nonempty point cloud requires a positive point_step")
    if message.row_step < message.width * message.point_step:
        raise ValueError("point cloud row_step is smaller than its row of points")
    if len(message.data) != message.height * message.row_step:
        raise ValueError("point cloud data length does not match height * row_step")
    names: list[str] = []
    formats: list[Any] = []
    offsets: list[int] = []
    for field in message.fields:
        if not field.name or field.name in names:
            raise ValueError("point cloud field names must be nonempty and unique")
        if field.datatype not in _FIELD_KINDS or not field.count:
            raise ValueError(f"invalid point cloud datatype/count for {field.name!r}")
        dtype = np.dtype((">" if message.is_bigendian else "<") + _FIELD_KINDS[field.datatype])
        if field.offset + dtype.itemsize * field.count > message.point_step:
            raise ValueError(f"point cloud field {field.name!r} exceeds point_step")
        names.append(field.name)
        formats.append(dtype if field.count == 1 else (dtype, (field.count,)))
        offsets.append(field.offset)
    dtype = np.dtype(
        {"names": names, "formats": formats, "offsets": offsets, "itemsize": message.point_step}
    )
    view = np.ndarray(
        (message.height, message.width),
        dtype=dtype,
        buffer=message.data,
        strides=(message.row_step, message.point_step),
    )
    view.setflags(write=False)
    return view


def pointcloud_xyz(message: PointCloud2) -> NDArray[np.float64]:
    """Copy scalar x/y/z fields into an N-by-3 float64 array, preserving point order."""
    points = pointcloud_view(message)
    for name in ("x", "y", "z"):
        if name not in (points.dtype.names or ()) or points[name].ndim != 2:
            raise ValueError(f"point cloud requires a scalar {name!r} field")
    return np.stack([points[name].ravel() for name in ("x", "y", "z")], axis=-1).astype(np.float64)


def pointcloud_rgb(message: PointCloud2) -> NDArray[np.uint8] | None:
    """Copy RGB bytes from the standard packed rgb/rgba field, if present.

    FLOAT32 fields carry color bits, not numeric floating-point color values.
    Both representations preserve the declared byte order and row padding.
    """
    points = pointcloud_view(message)
    names = points.dtype.names or ()
    name = next((name for name in ("rgb", "rgba") if name in names), None)
    if name is None:
        return None
    field = points[name]
    if field.ndim != 2 or field.dtype.kind not in "fu" or field.dtype.itemsize != 4:
        raise ValueError("packed point cloud color must be a scalar FLOAT32 or UINT32 field")
    packed = field.view(np.dtype(">u4" if message.is_bigendian else "<u4")).ravel()
    return np.stack([(packed >> shift) & 255 for shift in (16, 8, 0)], axis=-1).astype(np.uint8)
