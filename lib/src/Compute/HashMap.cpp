// OpenCL Compute includes
#include <CL/Compute/HashMap.hpp>
#include <CL/Compute/Algorithm.hpp>
#include <CL/Compute/ProgramCache.hpp>
#include "Detail.hpp"

// STL includes
#include <stdexcept>

namespace cl {
namespace compute {

    namespace {

        // counters[0]: newly inserted keys, counters[1]: failed inserts.
        const char* static_map_source = R"CLC(
#include <clc/hash.h>

__kernel void clc_map_insert(__global const uint* keys,
                             __global const uint* values, uint n,
                             volatile __global uint* slot_keys,
                             __global uint* slot_values, uint mask,
                             volatile __global uint* counters)
{
    const uint i = get_global_id(0);
    if (i >= n) return;
    const uint key = keys[i];
    if (key == EMPTY_KEY) return;
    uint slot = clc_hash_u32(key) & mask;
    for (uint probe = 0; probe <= mask; ++probe)
    {
        const uint old = atomic_cmpxchg(&slot_keys[slot], EMPTY_KEY, key);
        if (old == EMPTY_KEY)
        {
            slot_values[slot] = values[i];
            atomic_inc(&counters[0]);
            return;
        }
        if (old == key) return;
        slot = (slot + 1) & mask;
    }
    atomic_inc(&counters[1]);
}

uint clc_map_probe(uint key, __global const uint* slot_keys, uint mask)
{
    if (key == EMPTY_KEY) return mask + 1;
    uint slot = clc_hash_u32(key) & mask;
    for (uint probe = 0; probe <= mask; ++probe)
    {
        const uint current = slot_keys[slot];
        if (current == key) return slot;
        if (current == EMPTY_KEY) break;
        slot = (slot + 1) & mask;
    }
    return mask + 1;
}

__kernel void clc_map_find(__global const uint* keys, uint n,
                           __global const uint* slot_keys,
                           __global const uint* slot_values, uint mask,
                           __global uint* out)
{
    const uint i = get_global_id(0);
    if (i >= n) return;
    const uint slot = clc_map_probe(keys[i], slot_keys, mask);
    out[i] = slot <= mask ? slot_values[slot] : EMPTY_VALUE;
}

__kernel void clc_map_contains(__global const uint* keys, uint n,
                               __global const uint* slot_keys, uint mask,
                               __global uchar* out)
{
    const uint i = get_global_id(0);
    if (i >= n) return;
    out[i] = clc_map_probe(keys[i], slot_keys, mask) <= mask ? 1 : 0;
}
)CLC";

        std::size_t next_pow2(std::size_t n)
        {
            std::size_t p = 1;
            while (p < n) p <<= 1;
            return p;
        }

        std::string uint_literal(cl_uint v) { return std::to_string(v) + "u"; }

    }

    static_map::static_map(compute_context& ctx, std::size_t capacity,
                           cl_uint empty_key, cl_uint empty_value)
        : ctx_(ctx), capacity_(next_pow2(std::max<std::size_t>(capacity, 2))),
          empty_key_(empty_key), empty_value_(empty_value),
          keys_(ctx, capacity_), values_(ctx, capacity_), counters_(ctx, 2)
    {
        if (capacity_ > (std::size_t(1) << 31))
            throw std::length_error("static_map: capacity above 2^31");
        clear();
    }

    void static_map::clear()
    {
        fill(ctx_, keys_, empty_key_);
        fill(ctx_, values_, empty_value_);
        size_ = 0;
    }

    std::size_t static_map::insert(const device_buffer<cl_uint>& keys,
                                   const device_buffer<cl_uint>& values,
                                   std::size_t n)
    {
        if (n > keys.size() || n > values.size())
            throw std::invalid_argument("static_map::insert: n out of range");
        if (n == 0) return 0;
        internal::check_size(n);
        const defines defs{ { "EMPTY_KEY", uint_literal(empty_key_) },
                            { "EMPTY_VALUE", uint_literal(empty_value_) } };
        cl::Kernel kernel = ctx_.cache().kernel(
            static_map_source, "clc_map_insert", std::string(), defs);
        fill(ctx_, counters_, cl_uint(0));
        kernel.setArg(0, keys.get());
        kernel.setArg(1, values.get());
        kernel.setArg(2, static_cast<cl_uint>(n));
        kernel.setArg(3, keys_.get());
        kernel.setArg(4, values_.get());
        kernel.setArg(5, static_cast<cl_uint>(capacity_ - 1));
        kernel.setArg(6, counters_.get());
        ctx_.queue().enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(n),
                                          cl::NullRange);
        cl_uint counts[2];
        counters_.copy_to_host(ctx_.queue(), counts, 2);
        size_ += counts[0];
        if (counts[1] != 0)
            throw std::overflow_error("static_map::insert: table is full ("
                                      + std::to_string(counts[1])
                                      + " keys not inserted)");
        return counts[0];
    }

    void static_map::find(const device_buffer<cl_uint>& keys,
                          device_buffer<cl_uint>& values) const
    {
        if (values.size() < keys.size())
            throw std::invalid_argument("static_map::find: output too small");
        if (keys.empty()) return;
        internal::check_size(keys.size());
        const defines defs{ { "EMPTY_KEY", uint_literal(empty_key_) },
                            { "EMPTY_VALUE", uint_literal(empty_value_) } };
        cl::Kernel kernel = ctx_.cache().kernel(
            static_map_source, "clc_map_find", std::string(), defs);
        kernel.setArg(0, keys.get());
        kernel.setArg(1, static_cast<cl_uint>(keys.size()));
        kernel.setArg(2, keys_.get());
        kernel.setArg(3, values_.get());
        kernel.setArg(4, static_cast<cl_uint>(capacity_ - 1));
        kernel.setArg(5, values.get());
        ctx_.queue().enqueueNDRangeKernel(
            kernel, cl::NullRange, cl::NDRange(keys.size()), cl::NullRange);
    }

    void static_map::contains(const device_buffer<cl_uint>& keys,
                              device_buffer<cl_uchar>& found) const
    {
        if (found.size() < keys.size())
            throw std::invalid_argument(
                "static_map::contains: output too small");
        if (keys.empty()) return;
        internal::check_size(keys.size());
        const defines defs{ { "EMPTY_KEY", uint_literal(empty_key_) },
                            { "EMPTY_VALUE", uint_literal(empty_value_) } };
        cl::Kernel kernel = ctx_.cache().kernel(
            static_map_source, "clc_map_contains", std::string(), defs);
        kernel.setArg(0, keys.get());
        kernel.setArg(1, static_cast<cl_uint>(keys.size()));
        kernel.setArg(2, keys_.get());
        kernel.setArg(3, static_cast<cl_uint>(capacity_ - 1));
        kernel.setArg(4, found.get());
        ctx_.queue().enqueueNDRangeKernel(
            kernel, cl::NullRange, cl::NDRange(keys.size()), cl::NullRange);
    }

}
}
