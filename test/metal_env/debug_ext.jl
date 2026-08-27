using FastMultipole, KernelAbstractions, Metal

println("=== Extension Loading Diagnostic ===")

# Check if extension is loaded
ext = Base.get_extension(FastMultipole, :FastMultipoleKAExt)
println("Extension loaded: $(ext !== nothing)")
if ext !== nothing
    println("Extension module: $ext")
end

# Check if the function exists
if hasmethod(FastMultipole.ka_m2m_operator_batch!, (Any, Any, Any, Any, Any, Any, Any, Any, Any))
    println("ka_m2m_operator_batch! method found")
else
    println("ka_m2m_operator_batch! method NOT found")
end

# Print all methods
println("\nMethods for ka_m2m_operator_batch!:")
for m in methods(FastMultipole.ka_m2m_operator_batch!)
    println("  $m")
end

# Try to call it with a simple test
println("\n=== Attempting test call ===")
try
    using StaticArrays, Random
    using FastMultipole: FlatCoefficientBuffer, M2MOperatorScratch, OperatorInvariantCache,
                         MaterializedYRotationM2M

    TF = Float32
    P = 4
    nbatch = 2
    lh = Val(true)

    cache = OperatorInvariantCache(TF, P, lh)
    scratch = M2MOperatorScratch(TF, cache.basis_info, nbatch)

    sources = FlatCoefficientBuffer(TF, cache.basis_info, nbatch)
    targets = FlatCoefficientBuffer(TF, cache.basis_info, nbatch)

    phis = rand(TF, nbatch)
    thetas = rand(TF, nbatch)
    rs = rand(TF, nbatch)

    op = MaterializedYRotationM2M()

    println("Calling ka_m2m_operator_batch! with CPU arrays...")
    FastMultipole.ka_m2m_operator_batch!(op, targets, sources, phis, thetas, rs, cache, scratch, lh)
    println("✓ CPU call succeeded")

    println("\nCalling with Metal arrays...")
    targets_metal = FlatCoefficientBuffer(TF, cache.basis_info, nbatch)
    sources_metal = FlatCoefficientBuffer(TF, cache.basis_info, nbatch)
    sources_metal.phi = Metal.MtlArray(sources.phi)
    sources_metal.chi = Metal.MtlArray(sources.chi)
    targets_metal.phi = Metal.MtlArray(targets_metal.phi)
    targets_metal.chi = Metal.MtlArray(targets_metal.chi)

    FastMultipole.ka_m2m_operator_batch!(op, targets_metal, sources_metal, phis, thetas, rs, cache, scratch, lh)
    println("✓ Metal call succeeded")

catch err
    println("ERROR: $err")
    println("\nStacktrace:")
    showerror(stdout, err)
    println()
end
