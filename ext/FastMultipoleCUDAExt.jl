module FastMultipoleCUDAExt

# Free device memory for sizing the automatic concat-M2L chunk.

import FastMultipole
import CUDA

FastMultipole._device_free_bytes(::CUDA.CUDABackend) = CUDA.free_memory()  # driver free bytes (CUDA 5 and 6)

end
