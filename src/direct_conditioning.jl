#------- direct conditioning -------#

"""
    DirectConditioningRule(matcher, before!, after!)

Rule used to temporarily condition a source buffer for selected source-target
system pairs in `direct!` and CPU nearfield interactions in `fmm!`.

`matcher` is queried with `applies(matcher, i_source_system, i_target_system)`.
`before!` and `after!` must be user-provided functions with signature

```julia
f(source_buffer, source_system, i_source_system, target_buffer, i_target_system)
```
"""
struct DirectConditioningRule{TMatcher,TBefore,TAfter}
    matcher::TMatcher
    before!::TBefore
    after!::TAfter
end

"""
    SelfPairs()

Matcher for `DirectConditioningRule` that applies when
`i_source_system == i_target_system`.
"""
struct SelfPairs end

"""
    AllPairs()

Matcher for `DirectConditioningRule` that applies to every source-target system
pair.
"""
struct AllPairs end

"""
    PairSet(((i_source, i_target), ...))

Matcher for `DirectConditioningRule` that applies to explicit source-target
system index pairs.
"""
struct PairSet{TPairs}
    pairs::TPairs
end

PairSet(pairs::Tuple) = PairSet{typeof(pairs)}(pairs)
PairSet(pair::Pair) = PairSet(((pair.first, pair.second),))

"""
    applies(matcher, i_source_system, i_target_system)

Return whether `matcher` applies to a source-target system index pair. Custom
conditioning matchers should overload this function.
"""
@inline applies(::SelfPairs, i_source_system, i_target_system) = i_source_system == i_target_system
@inline applies(::AllPairs, i_source_system, i_target_system) = true

@inline function applies(matcher::PairSet, i_source_system, i_target_system)
    return _contains_pair(matcher.pairs, i_source_system, i_target_system)
end

@inline _contains_pair(::Tuple{}, i_source_system, i_target_system) = false

@inline function _contains_pair(pairs::Tuple, i_source_system, i_target_system)
    i_source, i_target = pairs[1]
    if i_source == i_source_system && i_target == i_target_system
        return true
    else
        return _contains_pair(Base.tail(pairs), i_source_system, i_target_system)
    end
end

@inline normalize_direct_conditioning(::Tuple{}) = ()
@inline normalize_direct_conditioning(rules::Tuple) = rules
@inline normalize_direct_conditioning(rule::DirectConditioningRule) = (rule,)

@inline has_direct_conditioning(::Tuple{}) = false
@inline has_direct_conditioning(rules::Tuple) = true

@inline function apply_direct_conditioning_before!(rules::Tuple, source_buffer, source_system, i_source_system, target_buffer, i_target_system)
    for rule in rules
        if applies(rule.matcher, i_source_system, i_target_system)
            rule.before!(source_buffer, source_system, i_source_system, target_buffer, i_target_system)
        end
    end
end

@inline function apply_direct_conditioning_after!(rules::Tuple, source_buffer, source_system, i_source_system, target_buffer, i_target_system)
    for rule in Iterators.reverse(rules)
        if applies(rule.matcher, i_source_system, i_target_system)
            rule.after!(source_buffer, source_system, i_source_system, target_buffer, i_target_system)
        end
    end
end

@inline function with_direct_conditioning!(f, rules::Tuple, source_buffer, source_system, i_source_system, target_buffer, i_target_system)
    apply_direct_conditioning_before!(rules, source_buffer, source_system, i_source_system, target_buffer, i_target_system)
    try
        f()
    finally
        apply_direct_conditioning_after!(rules, source_buffer, source_system, i_source_system, target_buffer, i_target_system)
    end
end
