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
"""CUDA host interop helpers for `std.gpu.host`.

This module exposes CUDA-native handle types and conversion helpers for the
generic GPU host runtime. It also reserves the public
`DeviceContext.import_stream()` API shape for external CUDA stream interop.

The current AsyncRT toolchain does not yet expose a runtime hook that can wrap
an external `CUstream` into a `DeviceStream`, so calling
`DeviceContext.import_stream()` currently raises an error.
"""

from ._nvidia_cuda import (
    CUDA,
    CUDA_MODULE,
    CUDA_get_current_context,
    CUcontext,
    CUevent,
    CUmodule,
    CUstream,
)
from .device_context import DeviceContext, DeviceStream


__extension DeviceContext:
    @always_inline
    fn import_stream(self, stream: CUstream) raises -> DeviceStream:
        """Returns a `DeviceStream` wrapper for an external CUDA stream.

        This API reserves the stdlib surface for external CUDA stream interop.
        The current AsyncRT toolchain does not yet provide the underlying
        runtime support, so this method raises instead of returning a wrapper.

        Args:
            self: The device context that would own the imported wrapper.
            stream: The CUDA stream handle to import.

        Returns:
            A `DeviceStream` wrapper when runtime support is available.

        Raises:
            Error: External CUDA stream import is not yet supported by the
                current AsyncRT toolchain.
        """
        _ = self
        _ = stream
        raise Error(
            "External CUDA stream import is not supported by this AsyncRT toolchain."
        )
