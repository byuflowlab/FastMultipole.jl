module FastMultipoleAMDGPUExt

# Free device memory for sizing the automatic concat-M2L chunk.

import FastMultipole
import AMDGPU

FastMultipole._device_free_bytes(::AMDGPU.ROCBackend) = Int(AMDGPU.free())

end
