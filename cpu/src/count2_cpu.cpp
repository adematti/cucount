// Multi-ISA translation unit. foreach_target.h re-includes THIS FILE once per
// SIMD target with the right per-target flags, and each pass pulls in
// kernel-inl.h again via its toggle guard. HWY_DYNAMIC_DISPATCH then picks a
// target at run time from CPUID.

#include "cucount/cpu/types.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "src/count2_cpu.cpp"
#include "hwy/foreach_target.h"  // must precede highway.h

#include "hwy/highway.h"
// Per-target; the toggle guard lets it re-expand on every pass.
#include "cucount/cpu/kernel-inl.h"

#if HWY_ONCE

#include <cstring>

namespace cucount {
namespace cpu {

// The dispatch tables
HWY_EXPORT(Count2Dispatch);
HWY_EXPORT(KernelTarget);  // for querying the SIMD target

void Count2(const Count2Args& args) {
    HWY_DYNAMIC_DISPATCH(Count2Dispatch)(args);
}

const char* CurrentTarget() { return HWY_DYNAMIC_DISPATCH(KernelTarget)(); }

std::vector<const char*> AvailableTargets() {
    std::vector<const char*> names;
    const int64_t supported = hwy::SupportedTargets();
    for (int i = 0; i < 63; ++i) {
        const int64_t bit = int64_t{1} << i;
        if (supported & bit) names.push_back(hwy::TargetName(bit));
    }
    return names;
}

// Restricting the target set is how we measure SIMD-width scaling and
// check that every ISA agrees; there is no other way to reach a narrower
// target on a machine that supports a wider one.
const char* SetTarget(const char* name) {
    if (name == nullptr || name[0] == '\0') {
        hwy::SetSupportedTargetsForTest(0);
        return CurrentTarget();
    }
    for (int i = 0; i < 63; ++i) {
        const int64_t bit = int64_t{1} << i;
        if (std::strcmp(hwy::TargetName(bit), name) == 0) {
            hwy::SetSupportedTargetsForTest(bit);
            return CurrentTarget();
        }
    }
    return nullptr;
}

}  // namespace cpu
}  // namespace cucount

#endif  // HWY_ONCE
