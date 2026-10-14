// OpenCL Compute includes
#include <CL/Compute/Algorithm.hpp>
#include "Detail.hpp"

// STL includes
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace cl {
namespace compute {

    namespace {

        const char* elementwise_source = R"CLC(
__kernel void clc_fill(__global T* out, uint n, T value)
{
    const uint i = get_global_id(0);
    if (i < n) out[i] = value;
}

__kernel void clc_sequence(__global T* out, uint n, T start, T step)
{
    const uint i = get_global_id(0);
    if (i < n) out[i] = start + (T)i * step;
}
)CLC";

        const char* transform_source = R"CLC(
__kernel void clc_transform(__global const T_IN* in, __global T_OUT* out,
                            uint n)
{
    const uint i = get_global_id(0);
    if (i < n)
    {
        const T_IN x = in[i];
        out[i] = (T_OUT)(EXPR);
    }
}
)CLC";

        const char* transform2_source = R"CLC(
__kernel void clc_transform2(__global const T_IN1* in1,
                             __global const T_IN2* in2, __global T_OUT* out,
                             uint n)
{
    const uint i = get_global_id(0);
    if (i < n)
    {
        const T_IN1 x = in1[i];
        const T_IN2 y = in2[i];
        out[i] = (T_OUT)(EXPR);
    }
}
)CLC";

        const char* reduce_source = R"CLC(
#include <clc/workgroup.h>

T clc_op(T a, T b) { return (OP); }
CLC_DEFINE_WG_REDUCE(T, clc_wg_reduce, clc_op(a, b))

__kernel void clc_reduce(__global const T* in, uint n, __global T* out,
                         __local T* scratch, T init, uint apply_init)
{
    T acc = (T)(IDENTITY);
    for (uint i = get_global_id(0); i < n; i += get_global_size(0))
        acc = clc_op(acc, in[i]);
    const T r = clc_wg_reduce(acc, scratch);
    if (get_local_id(0) == 0)
        out[get_group_id(0)] = apply_init ? clc_op(init, r) : r;
}
)CLC";

        // Three-phase scan: each work-group scans a tile of
        // ITEMS * local_size elements and records the tile total; the
        // totals are scanned recursively and added back to every tile.
        const char* scan_source = R"CLC(
#include <clc/workgroup.h>

#define ITEMS 4

T clc_op(T a, T b) { return (OP); }
CLC_DEFINE_WG_SCAN(T, clc_wg_scan, clc_op(a, b))

__kernel void clc_scan_tiles(__global const T* in, __global T* out, uint n,
                             __global T* tile_totals, __local T* scratch,
                             uint exclusive)
{
    const uint lid = get_local_id(0);
    const uint base = get_group_id(0) * get_local_size(0) * ITEMS + lid * ITEMS;
    T items[ITEMS];
    T thread_total = (T)(IDENTITY);
    for (uint k = 0; k < ITEMS; ++k)
    {
        const uint i = base + k;
        items[k] = i < n ? in[i] : (T)(IDENTITY);
        thread_total = clc_op(thread_total, items[k]);
    }
    T tile_total;
    T running = clc_wg_scan_exclusive(thread_total, (T)(IDENTITY), scratch,
                                      &tile_total);
    for (uint k = 0; k < ITEMS; ++k)
    {
        const uint i = base + k;
        const T next = clc_op(running, items[k]);
        if (i < n) out[i] = exclusive ? running : next;
        running = next;
    }
    if (lid == 0) tile_totals[get_group_id(0)] = tile_total;
}

__kernel void clc_scan_add(__global T* out, uint n,
                           __global const T* tile_offsets)
{
    const uint group = get_group_id(0);
    if (group == 0) return;
    const T offset = tile_offsets[group];
    const uint base = group * get_local_size(0) * ITEMS + get_local_id(0) * ITEMS;
    for (uint k = 0; k < ITEMS; ++k)
    {
        const uint i = base + k;
        if (i < n) out[i] = clc_op(offset, out[i]);
    }
}
)CLC";

        // LSD radix sort, 4 bits per pass. Ranks inside a work-group are
        // computed with one scan of uint4 vectors holding sixteen 8-bit
        // per-digit counters, which is why the work-group size is capped at
        // 128 (counts never reach 256, so lanes cannot carry).
        const char* radix_source = R"CLC(
#include <clc/workgroup.h>

#define RADIX 16u
#define ITEMS 8u

CLC_DEFINE_WG_SCAN(uint4, clc_scan4, (a + b))

uint clc_digit(uint k, uint shift)
{
    return ((KEY_TRANSFORM(k)) >> shift) & (RADIX - 1u);
}

uint clc_lane(uint4 v, uint d)
{
    const uint q = d >> 2;
    const uint w = q == 0u ? v.s0 : q == 1u ? v.s1 : q == 2u ? v.s2 : v.s3;
    return (w >> ((d & 3u) * 8u)) & 0xFFu;
}

__kernel void clc_radix_histogram(__global const uint* keys, uint n,
                                  uint shift, __global uint* histogram,
                                  uint num_groups)
{
    __local uint counts[RADIX];
    const uint lid = get_local_id(0);
    const uint wg = get_local_size(0);
    const uint group = get_group_id(0);
    if (lid < RADIX) counts[lid] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    const uint start = group * wg * ITEMS;
    for (uint r = 0; r < ITEMS; ++r)
    {
        const uint i = start + r * wg + lid;
        if (i < n) atomic_inc(&counts[clc_digit(keys[i], shift)]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid < RADIX) histogram[lid * num_groups + group] = counts[lid];
}

__kernel void clc_radix_scatter(__global const uint* keys_in,
                                __global uint* keys_out,
                                __global const uint* values_in,
                                __global uint* values_out, uint n, uint shift,
                                __global const uint* offsets, uint num_groups,
                                __local uint4* scratch)
{
    __local uint base[RADIX];
    const uint lid = get_local_id(0);
    const uint wg = get_local_size(0);
    const uint group = get_group_id(0);
    if (lid < RADIX) base[lid] = offsets[lid * num_groups + group];
    barrier(CLK_LOCAL_MEM_FENCE);
    const uint start = group * wg * ITEMS;
    for (uint r = 0; r < ITEMS; ++r)
    {
        const uint i = start + r * wg + lid;
        const bool valid = i < n;
        const uint key = valid ? keys_in[i] : 0u;
        const uint d = clc_digit(key, shift);
        const uint bit = valid ? 1u << ((d & 3u) * 8u) : 0u;
        const uint q = d >> 2;
        const uint4 flag = (uint4)(q == 0u ? bit : 0u, q == 1u ? bit : 0u,
                                   q == 2u ? bit : 0u, q == 3u ? bit : 0u);
        uint4 total;
        const uint4 prefix = clc_scan4_exclusive(flag, (uint4)(0u), scratch,
                                                 &total);
        if (valid)
        {
            const uint dst = base[d] + clc_lane(prefix, d);
            keys_out[dst] = key;
#if HAS_VALUES
            values_out[dst] = values_in[i];
#endif
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid < RADIX) base[lid] += clc_lane(total, lid);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
)CLC";

        const char* select_source = R"CLC(
__kernel void clc_select_flags(__global const T* in, uint n,
                               __global uint* flags)
{
    const uint i = get_global_id(0);
    if (i < n)
    {
        const T x = in[i];
        flags[i] = (PRED) ? 1u : 0u;
    }
}

__kernel void clc_select_scatter(__global const T* in, uint n,
                                 __global const uint* positions,
                                 __global T* out)
{
    const uint i = get_global_id(0);
    if (i < n)
    {
        const T x = in[i];
        if (PRED) out[positions[i]] = x;
    }
}
)CLC";

        const char* histogram_source = R"CLC(
__kernel void clc_histogram(__global const T* in, uint n,
                            __global uint* histogram, uint bins, ACC lower,
                            ACC upper, __local uint* local_histogram)
{
    const uint lid = get_local_id(0);
    const uint lsize = get_local_size(0);
#if USE_LOCAL
    for (uint b = lid; b < bins; b += lsize) local_histogram[b] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
    const ACC scale = (ACC)bins / (upper - lower);
    for (uint i = get_global_id(0); i < n; i += get_global_size(0))
    {
        const ACC x = (ACC)in[i];
        if (x >= lower && x < upper)
        {
            uint b = (uint)((x - lower) * scale);
            if (b >= bins) b = bins - 1;
#if USE_LOCAL
            atomic_inc(&local_histogram[b]);
#else
            atomic_inc(&histogram[b]);
#endif
        }
    }
#if USE_LOCAL
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint b = lid; b < bins; b += lsize)
        if (local_histogram[b] != 0)
            atomic_add(&histogram[b], local_histogram[b]);
#endif
}
)CLC";

        defines typed(const detail::type_info& t)
        {
            return { { "T", t.name } };
        }

        void scan_impl(compute_context& ctx, const cl::Buffer& in,
                       const cl::Buffer& out, std::size_t n,
                       const detail::type_info& t, const binary_op& op,
                       bool exclusive)
        {
            defines defs = typed(t);
            defs.emplace_back("OP", op.expression);
            defs.emplace_back("IDENTITY", op.identity);
            const std::string source = internal::preamble({ t }) + scan_source;
            cl::Kernel tiles = ctx.cache().kernel(source, "clc_scan_tiles",
                                                  std::string(), defs);
            const std::size_t wg = ctx.work_group_size(tiles);
            const std::size_t tile = wg * 4;
            const std::size_t num_tiles = (n + tile - 1) / tile;

            internal::temp_buffer totals(ctx, num_tiles * t.size);
            tiles.setArg(0, in);
            tiles.setArg(1, out);
            tiles.setArg(2, static_cast<cl_uint>(n));
            tiles.setArg(3, totals.get());
            tiles.setArg(4, cl::Local(wg * t.size));
            tiles.setArg(5, static_cast<cl_uint>(exclusive ? 1 : 0));
            ctx.queue().enqueueNDRangeKernel(tiles, cl::NullRange,
                                             cl::NDRange(num_tiles * wg),
                                             cl::NDRange(wg));
            if (num_tiles == 1) return;

            internal::temp_buffer offsets(ctx, num_tiles * t.size);
            scan_impl(ctx, totals.get(), offsets.get(), num_tiles, t, op, true);

            cl::Kernel add =
                ctx.cache().kernel(source, "clc_scan_add", std::string(), defs);
            add.setArg(0, out);
            add.setArg(1, static_cast<cl_uint>(n));
            add.setArg(2, offsets.get());
            ctx.queue().enqueueNDRangeKernel(add, cl::NullRange,
                                             cl::NDRange(num_tiles * wg),
                                             cl::NDRange(wg));
        }

        void launch_elementwise(compute_context& ctx, cl::Kernel& kernel,
                                std::size_t n)
        {
            ctx.queue().enqueueNDRangeKernel(kernel, cl::NullRange,
                                             cl::NDRange(n), cl::NullRange);
        }

    }

    namespace detail {

        void fill(compute_context& ctx, const cl::Buffer& out, std::size_t n,
                  type_info t, const void* value)
        {
            if (n == 0) return;
            internal::check_size(n);
            cl::Kernel kernel = ctx.cache().kernel(
                internal::preamble({ t }) + elementwise_source, "clc_fill",
                std::string(), typed(t));
            kernel.setArg(0, out);
            kernel.setArg(1, static_cast<cl_uint>(n));
            kernel.setArg(2, t.size, value);
            launch_elementwise(ctx, kernel, n);
        }

        void sequence(compute_context& ctx, const cl::Buffer& out,
                      std::size_t n, type_info t, const void* start,
                      const void* step)
        {
            if (n == 0) return;
            internal::check_size(n);
            cl::Kernel kernel = ctx.cache().kernel(
                internal::preamble({ t }) + elementwise_source, "clc_sequence",
                std::string(), typed(t));
            kernel.setArg(0, out);
            kernel.setArg(1, static_cast<cl_uint>(n));
            kernel.setArg(2, t.size, start);
            kernel.setArg(3, t.size, step);
            launch_elementwise(ctx, kernel, n);
        }

        void transform(compute_context& ctx, const cl::Buffer& in,
                       type_info tin, const cl::Buffer& out, type_info tout,
                       std::size_t n, const std::string& expression)
        {
            if (n == 0) return;
            internal::check_size(n);
            const defines defs{ { "T_IN", tin.name },
                                { "T_OUT", tout.name },
                                { "EXPR", expression } };
            cl::Kernel kernel = ctx.cache().kernel(
                internal::preamble({ tin, tout }) + transform_source,
                "clc_transform", std::string(), defs);
            kernel.setArg(0, in);
            kernel.setArg(1, out);
            kernel.setArg(2, static_cast<cl_uint>(n));
            launch_elementwise(ctx, kernel, n);
        }

        void transform2(compute_context& ctx, const cl::Buffer& in1,
                        type_info t1, const cl::Buffer& in2, type_info t2,
                        const cl::Buffer& out, type_info tout, std::size_t n,
                        const std::string& expression)
        {
            if (n == 0) return;
            internal::check_size(n);
            const defines defs{ { "T_IN1", t1.name },
                                { "T_IN2", t2.name },
                                { "T_OUT", tout.name },
                                { "EXPR", expression } };
            cl::Kernel kernel = ctx.cache().kernel(
                internal::preamble({ t1, t2, tout }) + transform2_source,
                "clc_transform2", std::string(), defs);
            kernel.setArg(0, in1);
            kernel.setArg(1, in2);
            kernel.setArg(2, out);
            kernel.setArg(3, static_cast<cl_uint>(n));
            launch_elementwise(ctx, kernel, n);
        }

        void reduce(compute_context& ctx, const cl::Buffer& in, std::size_t n,
                    type_info t, const binary_op& op, const void* init,
                    void* result)
        {
            if (n == 0)
            {
                std::memcpy(result, init, t.size);
                return;
            }
            internal::check_size(n);
            defines defs = typed(t);
            defs.emplace_back("OP", op.expression);
            defs.emplace_back("IDENTITY", op.identity);
            cl::Kernel kernel =
                ctx.cache().kernel(internal::preamble({ t }) + reduce_source,
                                   "clc_reduce", std::string(), defs);
            const std::size_t wg = ctx.work_group_size(kernel);
            const std::size_t groups =
                std::min<std::size_t>((n + wg - 1) / wg, 256);

            internal::temp_buffer partials(ctx, groups * t.size);
            internal::temp_buffer total(ctx, t.size);
            const cl_uint no = 0, yes = 1;

            kernel.setArg(0, in);
            kernel.setArg(1, static_cast<cl_uint>(n));
            kernel.setArg(2, partials.get());
            kernel.setArg(3, cl::Local(wg * t.size));
            kernel.setArg(4, t.size, init);
            kernel.setArg(5, no);
            ctx.queue().enqueueNDRangeKernel(kernel, cl::NullRange,
                                             cl::NDRange(groups * wg),
                                             cl::NDRange(wg));

            kernel.setArg(0, partials.get());
            kernel.setArg(1, static_cast<cl_uint>(groups));
            kernel.setArg(2, total.get());
            kernel.setArg(5, yes);
            ctx.queue().enqueueNDRangeKernel(kernel, cl::NullRange,
                                             cl::NDRange(wg), cl::NDRange(wg));
            ctx.queue().enqueueReadBuffer(total.get(), CL_TRUE, 0, t.size,
                                          result);
        }

        void scan(compute_context& ctx, const cl::Buffer& in,
                  const cl::Buffer& out, std::size_t n, type_info t,
                  const binary_op& op, bool exclusive)
        {
            if (n == 0) return;
            internal::check_size(n);
            scan_impl(ctx, in, out, n, t, op, exclusive);
        }

        void radix_sort(compute_context& ctx, const cl::Buffer& keys,
                        type_info tkey, const cl::Buffer* values, std::size_t n,
                        bool descending)
        {
            if (n <= 1) return;
            internal::check_size(n);
            std::string transform = "(k)";
            const std::string key_type = tkey.name;
            if (key_type == "int")
                transform = "((k) ^ 0x80000000u)";
            else if (key_type == "float")
                transform =
                    "(((k) & 0x80000000u) ? ~(k) : ((k) | 0x80000000u))";
            else if (key_type != "uint")
                throw std::invalid_argument(
                    "sort: key type must be uint, int or float");
            if (descending) transform = "(~" + transform + ")";

            const defines defs{ { "KEY_TRANSFORM(k)", transform },
                                { "HAS_VALUES", values ? "1" : "0" } };
            const std::string source = radix_source;
            cl::Kernel histogram = ctx.cache().kernel(
                source, "clc_radix_histogram", std::string(), defs);
            cl::Kernel scatter = ctx.cache().kernel(source, "clc_radix_scatter",
                                                    std::string(), defs);
            const std::size_t wg = std::min(ctx.work_group_size(histogram, 128),
                                            ctx.work_group_size(scatter, 128));
            if (wg < 16)
                throw std::runtime_error(
                    "sort: device work-group size below 16");
            const std::size_t tile = wg * 8;
            const std::size_t groups = (n + tile - 1) / tile;
            const std::size_t bytes = n * sizeof(cl_uint);

            internal::temp_buffer keys_alt(ctx, bytes);
            internal::temp_buffer values_alt(ctx, values ? bytes : 0);
            internal::temp_buffer counts(ctx, 16 * groups * sizeof(cl_uint));
            internal::temp_buffer offsets(ctx, 16 * groups * sizeof(cl_uint));

            cl::Buffer key_bufs[2] = { keys, keys_alt.get() };
            cl::Buffer value_bufs[2] = { values ? *values : keys_alt.get(),
                                         values ? values_alt.get()
                                                : keys_alt.get() };
            const detail::type_info uint_info = info_of<cl_uint>();
            for (cl_uint pass = 0; pass < 8; ++pass)
            {
                const cl_uint shift = pass * 4;
                const int src = pass % 2, dst = 1 - src;
                histogram.setArg(0, key_bufs[src]);
                histogram.setArg(1, static_cast<cl_uint>(n));
                histogram.setArg(2, shift);
                histogram.setArg(3, counts.get());
                histogram.setArg(4, static_cast<cl_uint>(groups));
                ctx.queue().enqueueNDRangeKernel(histogram, cl::NullRange,
                                                 cl::NDRange(groups * wg),
                                                 cl::NDRange(wg));

                scan_impl(ctx, counts.get(), offsets.get(), 16 * groups,
                          uint_info, ops::plus(), true);

                scatter.setArg(0, key_bufs[src]);
                scatter.setArg(1, key_bufs[dst]);
                scatter.setArg(2, value_bufs[src]);
                scatter.setArg(3, value_bufs[dst]);
                scatter.setArg(4, static_cast<cl_uint>(n));
                scatter.setArg(5, shift);
                scatter.setArg(6, offsets.get());
                scatter.setArg(7, static_cast<cl_uint>(groups));
                scatter.setArg(8, cl::Local(wg * 4 * sizeof(cl_uint)));
                ctx.queue().enqueueNDRangeKernel(scatter, cl::NullRange,
                                                 cl::NDRange(groups * wg),
                                                 cl::NDRange(wg));
            }
            // Eight passes: the sorted data is back in the caller's buffers.
        }

        std::size_t copy_if(compute_context& ctx, const cl::Buffer& in,
                            const cl::Buffer& out, std::size_t n, type_info t,
                            const std::string& predicate)
        {
            if (n == 0) return 0;
            internal::check_size(n);
            defines defs = typed(t);
            defs.emplace_back("PRED", predicate);
            const std::string source =
                internal::preamble({ t }) + select_source;
            cl::Kernel flags_kernel = ctx.cache().kernel(
                source, "clc_select_flags", std::string(), defs);
            cl::Kernel scatter = ctx.cache().kernel(
                source, "clc_select_scatter", std::string(), defs);

            internal::temp_buffer flags(ctx, n * sizeof(cl_uint));
            internal::temp_buffer positions(ctx, n * sizeof(cl_uint));
            flags_kernel.setArg(0, in);
            flags_kernel.setArg(1, static_cast<cl_uint>(n));
            flags_kernel.setArg(2, flags.get());
            launch_elementwise(ctx, flags_kernel, n);

            scan_impl(ctx, flags.get(), positions.get(), n, info_of<cl_uint>(),
                      ops::plus(), true);

            scatter.setArg(0, in);
            scatter.setArg(1, static_cast<cl_uint>(n));
            scatter.setArg(2, positions.get());
            scatter.setArg(3, out);
            launch_elementwise(ctx, scatter, n);

            cl_uint last_position = 0, last_flag = 0;
            const std::size_t last = (n - 1) * sizeof(cl_uint);
            ctx.queue().enqueueReadBuffer(positions.get(), CL_FALSE, last,
                                          sizeof(cl_uint), &last_position);
            ctx.queue().enqueueReadBuffer(flags.get(), CL_TRUE, last,
                                          sizeof(cl_uint), &last_flag);
            return static_cast<std::size_t>(last_position) + last_flag;
        }

        void histogram_even(compute_context& ctx, const cl::Buffer& in,
                            std::size_t n, type_info t,
                            const cl::Buffer& histogram, std::size_t num_bins,
                            double lower, double upper)
        {
            if (num_bins == 0) return;
            if (!(lower < upper))
                throw std::invalid_argument(
                    "histogram_even: lower must be below upper");
            internal::check_size(n);
            const cl_uint zero = 0;
            fill(ctx, histogram, num_bins, info_of<cl_uint>(), &zero);
            if (n == 0) return;

            const bool wide = std::string(t.name) == "double";
            const std::size_t local_bytes = num_bins * sizeof(cl_uint);
            const bool use_local = local_bytes + 1024
                <= ctx.device().getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
            defines defs = typed(t);
            defs.emplace_back("ACC", wide ? "double" : "float");
            defs.emplace_back("USE_LOCAL", use_local ? "1" : "0");
            cl::Kernel kernel =
                ctx.cache().kernel(internal::preamble({ t }) + histogram_source,
                                   "clc_histogram", std::string(), defs);
            const std::size_t wg = ctx.work_group_size(kernel);
            const std::size_t groups =
                std::min<std::size_t>((n + wg - 1) / wg, 256);

            kernel.setArg(0, in);
            kernel.setArg(1, static_cast<cl_uint>(n));
            kernel.setArg(2, histogram);
            kernel.setArg(3, static_cast<cl_uint>(num_bins));
            if (wide)
            {
                kernel.setArg(4, static_cast<cl_double>(lower));
                kernel.setArg(5, static_cast<cl_double>(upper));
            }
            else
            {
                kernel.setArg(4, static_cast<cl_float>(lower));
                kernel.setArg(5, static_cast<cl_float>(upper));
            }
            kernel.setArg(6, cl::Local(use_local ? local_bytes : 4));
            ctx.queue().enqueueNDRangeKernel(kernel, cl::NullRange,
                                             cl::NDRange(groups * wg),
                                             cl::NDRange(wg));
        }

    }

}
}
