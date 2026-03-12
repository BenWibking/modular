# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""HIP host interop helpers for `std.gpu.host`.

This module exposes HIP-native handle types and conversion helpers for the
generic GPU host runtime. It also reserves the public
`DeviceContext.import_stream()` API shape for external HIP stream interop.

The current AsyncRT toolchain does not yet expose a runtime hook that can wrap
an external `hipStream_t` into a `DeviceStream`, so calling
`DeviceContext.import_stream()` currently raises an error.
"""

from ._amdgpu_hip import HIP, HIP_MODULE, hipDevice_t, hipModule_t, hipStream_t
from .device_context import DeviceContext, DeviceStream


__extension DeviceContext:
    @always_inline
    fn import_stream(self, stream: hipStream_t) raises -> DeviceStream:
        """Returns a `DeviceStream` wrapper for an external HIP stream.

        This API reserves the stdlib surface for external HIP stream interop.
        The current AsyncRT toolchain does not yet provide the underlying
        runtime support, so this method raises instead of returning a wrapper.

        Args:
            self: The device context that would own the imported wrapper.
            stream: The HIP stream handle to import.

        Returns:
            A `DeviceStream` wrapper when runtime support is available.

        Raises:
            Error: External HIP stream import is not yet supported by the
                current AsyncRT toolchain.
        """
        _ = self
        _ = stream
        raise Error(
            "External HIP stream import is not supported by this AsyncRT toolchain."
        )
