"Return the first index whose value and every later value are at most `tol`."
function first_suffix_safe(values, tol)
    suffix_max = similar(values)
    running = -Inf
    for i in reverse(eachindex(values))
        running = max(running, values[i])
        suffix_max[i] = running
    end
    return findfirst(<=(tol), suffix_max)
end
