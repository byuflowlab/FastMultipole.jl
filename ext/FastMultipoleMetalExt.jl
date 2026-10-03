module FastMultipoleMetalExt

# Free device memory for sizing the automatic concat-M2L chunk: the working set
# Metal recommends for this device, less what is already allocated on it.

import FastMultipole
import Metal

function FastMultipole._device_free_bytes(::Metal.MetalBackend)
    dev = Metal.device()
    return Int(dev.recommendedMaxWorkingSetSize) - Int(dev.currentAllocatedSize)
end

end
