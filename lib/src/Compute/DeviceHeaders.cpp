// OpenCL Compute includes
#include <CL/Compute/ProgramCache.hpp>

// Device-side OpenCL C headers available to every program built through a
// program_cache: the counterpart of libcu++ (cuda::atomic_ref, hashing) and
// of CUB's block-level collectives (cub::BlockReduce, cub::BlockScan). They
// only need OpenCL C 1.2 and its 32-bit atomics.

namespace {

const char* workgroup_h = R"CLC(
#ifndef CLC_WORKGROUP_H
#define CLC_WORKGROUP_H

/* Work-group collectives over local memory (cub::BlockReduce/BlockScan).
 *
 * CLC_DEFINE_WG_REDUCE(T, NAME, OP) defines
 *     T NAME(T x, __local T* scratch)
 * CLC_DEFINE_WG_SCAN(T, NAME, OP) defines
 *     T NAME##_inclusive(T x, __local T* scratch)
 *     T NAME##_exclusive(T x, T identity, __local T* scratch, T* total)
 *
 * OP is an expression of a and b, where a precedes b in sequence order, so
 * non-commutative associative operators are supported. Requirements: every
 * work-item of the work-group calls the function, get_local_size(0) is a
 * power of two and scratch holds get_local_size(0) elements. scratch may be
 * reused as soon as the function returns.
 */

#define CLC_DEFINE_WG_REDUCE(T, NAME, OP)                                     \
    T NAME(T x, __local T* scratch)                                           \
    {                                                                         \
        const uint lid = get_local_id(0);                                     \
        scratch[lid] = x;                                                     \
        barrier(CLK_LOCAL_MEM_FENCE);                                         \
        for (uint s = get_local_size(0) >> 1; s > 0; s >>= 1)                 \
        {                                                                     \
            if (lid < s)                                                      \
            {                                                                 \
                T a = scratch[lid];                                           \
                T b = scratch[lid + s];                                       \
                scratch[lid] = (OP);                                          \
            }                                                                 \
            barrier(CLK_LOCAL_MEM_FENCE);                                     \
        }                                                                     \
        T result = scratch[0];                                                \
        barrier(CLK_LOCAL_MEM_FENCE);                                         \
        return result;                                                        \
    }

#define CLC_WG_SCAN_BODY_(T, OP)                                              \
    const uint lid = get_local_id(0);                                         \
    const uint n = get_local_size(0);                                         \
    scratch[lid] = x;                                                         \
    barrier(CLK_LOCAL_MEM_FENCE);                                             \
    for (uint off = 1; off < n; off <<= 1)                                    \
    {                                                                         \
        T b = scratch[lid];                                                   \
        T v = b;                                                              \
        if (lid >= off)                                                       \
        {                                                                     \
            T a = scratch[lid - off];                                         \
            v = (OP);                                                         \
        }                                                                     \
        barrier(CLK_LOCAL_MEM_FENCE);                                         \
        scratch[lid] = v;                                                     \
        barrier(CLK_LOCAL_MEM_FENCE);                                         \
    }

#define CLC_DEFINE_WG_SCAN(T, NAME, OP)                                       \
    T NAME##_inclusive(T x, __local T* scratch)                               \
    {                                                                         \
        CLC_WG_SCAN_BODY_(T, OP)                                              \
        T result = scratch[lid];                                              \
        barrier(CLK_LOCAL_MEM_FENCE);                                         \
        return result;                                                        \
    }                                                                         \
    T NAME##_exclusive(T x, T identity, __local T* scratch, T* total)         \
    {                                                                         \
        CLC_WG_SCAN_BODY_(T, OP)                                              \
        T result = lid == 0 ? identity : scratch[lid - 1];                    \
        *total = scratch[n - 1];                                              \
        barrier(CLK_LOCAL_MEM_FENCE);                                         \
        return result;                                                        \
    }

#endif
)CLC";

const char* atomic_h = R"CLC(
#ifndef CLC_ATOMIC_H
#define CLC_ATOMIC_H

/* Atomic helpers missing from OpenCL C 1.2 (cuda::atomic_ref analogue). */

float clc_atomic_add_float(volatile __global float* p, float v)
{
    union { uint u; float f; } expected, desired;
    uint observed = *(volatile __global uint*)p;
    do
    {
        expected.u = observed;
        desired.f = expected.f + v;
        observed = atomic_cmpxchg((volatile __global uint*)p, expected.u,
                                  desired.u);
    } while (observed != expected.u);
    return expected.f;
}

float clc_atomic_add_float_local(volatile __local float* p, float v)
{
    union { uint u; float f; } expected, desired;
    uint observed = *(volatile __local uint*)p;
    do
    {
        expected.u = observed;
        desired.f = expected.f + v;
        observed = atomic_cmpxchg((volatile __local uint*)p, expected.u,
                                  desired.u);
    } while (observed != expected.u);
    return expected.f;
}

uint clc_atomic_max_uint(volatile __global uint* p, uint v)
{
    return atomic_max(p, v);
}

#endif
)CLC";

const char* hash_h = R"CLC(
#ifndef CLC_HASH_H
#define CLC_HASH_H

/* MurmurHash3 finalizer, the default hash of cuco::static_map. */
uint clc_hash_u32(uint k)
{
    k ^= k >> 16;
    k *= 0x85ebca6bu;
    k ^= k >> 13;
    k *= 0xc2b2ae35u;
    k ^= k >> 16;
    return k;
}

#endif
)CLC";

}

namespace cl {
namespace compute {

    const std::vector<std::pair<std::string, std::string>>& builtin_headers()
    {
        static const std::vector<std::pair<std::string, std::string>> headers{
            { "clc/workgroup.h", workgroup_h },
            { "clc/atomic.h", atomic_h },
            { "clc/hash.h", hash_h },
        };
        return headers;
    }

}
}
