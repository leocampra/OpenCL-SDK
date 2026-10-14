// OpenCL Compute includes
#include <CL/Compute/Compression.hpp>
#include <CL/Compute/Algorithm.hpp>
#include <CL/Compute/ProgramCache.hpp>
#include "Detail.hpp"

// STL includes
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace cl {
namespace compute {

    namespace {

        // Layout of compressed_buffer::data: per chunk {base, width, offset}
        // followed by the packed words; offset is relative to the start of
        // the packed area. Delta mode stores zigzag(x[i] - x[i-1]) using
        // wrapping 32-bit arithmetic, so decompression restores the input
        // exactly with an inclusive prefix sum.
        const char* bitpack_source = R"CLC(
#include <clc/workgroup.h>

CLC_DEFINE_WG_REDUCE(uint, clc_wg_min, min(a, b))
CLC_DEFINE_WG_REDUCE(uint, clc_wg_max, max(a, b))

uint clc_encode(__global const uint* in, uint i)
{
#if DELTA
    const uint d = in[i] - (i > 0 ? in[i - 1] : 0u);
    return (d << 1) ^ (0u - (d >> 31));
#else
    return in[i];
#endif
}

__kernel void clc_bitpack_analyze(__global const uint* in, uint n, uint chunk,
                                  __global uint* headers, __global uint* words,
                                  __local uint* scratch)
{
    const uint c = get_group_id(0);
    const uint lid = get_local_id(0);
    const uint start = c * chunk;
    const uint count = min(chunk, n - start);
    uint lo = UINT_MAX, hi = 0u;
    for (uint j = lid; j < count; j += get_local_size(0))
    {
        const uint v = clc_encode(in, start + j);
        lo = min(lo, v);
        hi = max(hi, v);
    }
    lo = clc_wg_min(lo, scratch);
    hi = clc_wg_max(hi, scratch);
    if (lid == 0)
    {
        const uint range = hi - lo;
        const uint width = range == 0u ? 0u : 32u - clz(range);
        headers[2 * c] = lo;
        headers[2 * c + 1] = width;
        words[c] = (count * width + 31u) / 32u;
    }
}

__kernel void clc_bitpack_pack(__global const uint* in, uint n, uint chunk,
                               __global const uint* headers,
                               __global const uint* offsets,
                               __global uint* out, uint num_chunks)
{
    const uint c = get_group_id(0);
    const uint lid = get_local_id(0);
    const uint start = c * chunk;
    const uint count = min(chunk, n - start);
    const uint base = headers[2 * c];
    const uint width = headers[2 * c + 1];
    const uint offset = offsets[c];
    if (lid == 0)
    {
        out[3 * c] = base;
        out[3 * c + 1] = width;
        out[3 * c + 2] = offset;
    }
    if (width == 0u) return;
    volatile __global uint* packed = out + 3 * num_chunks + offset;
    for (uint j = lid; j < count; j += get_local_size(0))
    {
        const uint v = clc_encode(in, start + j) - base;
        const uint bit = j * width;
        const uint w = bit >> 5;
        const uint b = bit & 31u;
        atomic_or(&packed[w], v << b);
        if (b + width > 32u) atomic_or(&packed[w + 1], v >> (32u - b));
    }
}

__kernel void clc_bitpack_unpack(__global const uint* data, uint n,
                                 uint chunk_shift, uint num_chunks,
                                 __global uint* out)
{
    const uint i = get_global_id(0);
    if (i >= n) return;
    const uint c = i >> chunk_shift;
    const uint j = i & ((1u << chunk_shift) - 1u);
    const uint base = data[3 * c];
    const uint width = data[3 * c + 1];
    uint v = 0u;
    if (width != 0u)
    {
        __global const uint* packed = data + 3 * num_chunks + data[3 * c + 2];
        const uint bit = j * width;
        const uint w = bit >> 5;
        const uint b = bit & 31u;
        uint bits = packed[w] >> b;
        if (b + width > 32u) bits |= packed[w + 1] << (32u - b);
        v = width == 32u ? bits : bits & ((1u << width) - 1u);
    }
    v += base;
#if DELTA
    v = (v >> 1) ^ (0u - (v & 1u));
#endif
    out[i] = v;
}
)CLC";

        const cl_uint magic = 0x50424C43u; // "CLBP"
        const cl_uint version = 1;
        const std::size_t header_words = 8;

        defines bitpack_defines(bool delta)
        {
            return { { "DELTA", delta ? "1" : "0" } };
        }

        unsigned log2_exact(std::size_t v)
        {
            unsigned shift = 0;
            while ((std::size_t(1) << shift) < v) ++shift;
            return shift;
        }
    }

    compressed_buffer compress_bitpack(compute_context& ctx,
                                       const device_buffer<cl_uint>& in,
                                       bitpack_options options)
    {
        const std::size_t chunk = options.chunk_size;
        if (chunk < 256 || chunk > 65536 || (chunk & (chunk - 1)) != 0)
            throw std::invalid_argument(
                "compress_bitpack: chunk_size must be a power of two in "
                "[256, 65536]");
        compressed_buffer result;
        result.num_elements = in.size();
        result.chunk_size = chunk;
        result.delta = options.delta;
        const std::size_t n = in.size();
        if (n == 0) return result;
        internal::check_size(n);

        const defines defs = bitpack_defines(options.delta);
        cl::Kernel analyze = ctx.cache().kernel(
            bitpack_source, "clc_bitpack_analyze", std::string(), defs);
        cl::Kernel pack = ctx.cache().kernel(bitpack_source, "clc_bitpack_pack",
                                             std::string(), defs);
        const std::size_t wg = ctx.work_group_size(analyze);
        const std::size_t chunks = result.num_chunks();

        internal::temp_buffer headers(ctx, 2 * chunks * sizeof(cl_uint));
        device_buffer<cl_uint> words(ctx, chunks), offsets(ctx, chunks);
        analyze.setArg(0, in.get());
        analyze.setArg(1, static_cast<cl_uint>(n));
        analyze.setArg(2, static_cast<cl_uint>(chunk));
        analyze.setArg(3, headers.get());
        analyze.setArg(4, words.get());
        analyze.setArg(5, cl::Local(wg * sizeof(cl_uint)));
        ctx.queue().enqueueNDRangeKernel(
            analyze, cl::NullRange, cl::NDRange(chunks * wg), cl::NDRange(wg));
        exclusive_scan(ctx, words, offsets);

        cl_uint last_offset = 0, last_words = 0;
        offsets.copy_to_host(ctx.queue(), &last_offset, 1, chunks - 1, false);
        words.copy_to_host(ctx.queue(), &last_words, 1, chunks - 1);
        const std::size_t packed_words =
            static_cast<std::size_t>(last_offset) + last_words;

        result.data = device_buffer<cl_uint>(ctx, 3 * chunks + packed_words);
        fill(ctx, result.data, cl_uint(0));
        const std::size_t pack_wg = ctx.work_group_size(pack);
        pack.setArg(0, in.get());
        pack.setArg(1, static_cast<cl_uint>(n));
        pack.setArg(2, static_cast<cl_uint>(chunk));
        pack.setArg(3, headers.get());
        pack.setArg(4, offsets.get());
        pack.setArg(5, result.data.get());
        pack.setArg(6, static_cast<cl_uint>(chunks));
        ctx.queue().enqueueNDRangeKernel(pack, cl::NullRange,
                                         cl::NDRange(chunks * pack_wg),
                                         cl::NDRange(pack_wg));
        return result;
    }

    void decompress_bitpack(compute_context& ctx, const compressed_buffer& in,
                            device_buffer<cl_uint>& out)
    {
        if (out.size() < in.num_elements)
            throw std::invalid_argument("decompress_bitpack: output too small");
        const std::size_t n = in.num_elements;
        if (n == 0) return;
        internal::check_size(n);
        cl::Kernel unpack =
            ctx.cache().kernel(bitpack_source, "clc_bitpack_unpack",
                               std::string(), bitpack_defines(in.delta));
        unpack.setArg(0, in.data.get());
        unpack.setArg(1, static_cast<cl_uint>(n));
        unpack.setArg(2, static_cast<cl_uint>(log2_exact(in.chunk_size)));
        unpack.setArg(3, static_cast<cl_uint>(in.num_chunks()));
        unpack.setArg(4, out.get());
        ctx.queue().enqueueNDRangeKernel(unpack, cl::NullRange, cl::NDRange(n),
                                         cl::NullRange);
        if (in.delta)
            detail::scan(ctx, out.get(), out.get(), n,
                         detail::info_of<cl_uint>(), ops::plus(), false);
    }

    std::vector<unsigned char>
    compressed_buffer::serialize(compute_context& ctx) const
    {
        const std::uint64_t n = num_elements;
        const std::uint64_t words = data.size();
        const cl_uint header[header_words] = {
            magic,
            version,
            delta ? 1u : 0u,
            static_cast<cl_uint>(chunk_size),
            static_cast<cl_uint>(n & 0xFFFFFFFFu),
            static_cast<cl_uint>(n >> 32),
            static_cast<cl_uint>(words & 0xFFFFFFFFu),
            static_cast<cl_uint>(words >> 32),
        };
        std::vector<unsigned char> blob((header_words + data.size())
                                        * sizeof(cl_uint));
        std::memcpy(blob.data(), header, sizeof(header));
        data.copy_to_host(
            ctx.queue(),
            reinterpret_cast<cl_uint*>(blob.data() + sizeof(header)),
            data.size());
        return blob;
    }

    compressed_buffer
    compressed_buffer::deserialize(compute_context& ctx,
                                   const std::vector<unsigned char>& blob)
    {
        cl_uint header[header_words];
        if (blob.size() < sizeof(header))
            throw std::invalid_argument("bitpack: truncated header");
        std::memcpy(header, blob.data(), sizeof(header));
        if (header[0] != magic || header[1] != version)
            throw std::invalid_argument("bitpack: not a bitpack stream");
        const std::uint64_t n = header[4] | (std::uint64_t(header[5]) << 32);
        const std::uint64_t words =
            header[6] | (std::uint64_t(header[7]) << 32);
        if (blob.size() != (header_words + words) * sizeof(cl_uint))
            throw std::invalid_argument("bitpack: size mismatch");

        compressed_buffer result;
        result.delta = header[2] != 0;
        result.chunk_size = header[3];
        result.num_elements = static_cast<std::size_t>(n);
        if (result.chunk_size < 256 || result.chunk_size > 65536
            || (result.chunk_size & (result.chunk_size - 1)) != 0
            || (n != 0 && words < 3 * result.num_chunks()))
            throw std::invalid_argument("bitpack: corrupt header");

        // Reject chunk headers that would make the kernels read out of
        // bounds.
        const auto* body =
            reinterpret_cast<const cl_uint*>(blob.data() + sizeof(header));
        const std::size_t chunks = result.num_chunks();
        const std::uint64_t packed_words = words - 3 * chunks;
        for (std::size_t c = 0; c < chunks; ++c)
        {
            const std::uint64_t count = std::min<std::uint64_t>(
                result.chunk_size, n - c * result.chunk_size);
            const cl_uint width = body[3 * c + 1];
            const std::uint64_t needed = (count * width + 31) / 32;
            if (width > 32 || body[3 * c + 2] + needed > packed_words)
                throw std::invalid_argument("bitpack: corrupt chunk header");
        }

        result.data =
            device_buffer<cl_uint>(ctx, static_cast<std::size_t>(words));
        result.data.copy_from_host(
            ctx.queue(),
            reinterpret_cast<const cl_uint*>(blob.data() + sizeof(header)),
            result.data.size());
        return result;
    }

}
}
