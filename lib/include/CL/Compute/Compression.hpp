#pragma once

// OpenCL Compute includes
#include "OpenCLCompute_Export.h"
#include <CL/Compute/Context.hpp>
#include <CL/Compute/DeviceBuffer.hpp>

// STL includes
#include <cstddef>
#include <vector>

namespace cl {
namespace compute {

    /*! \brief GPU compression of 32-bit integer data (nvcomp "Cascaded" /
     *  Bitcomp analogue).
     *
     *  The input is split into chunks that are compressed independently and
     *  in parallel, one work-group per chunk. Each chunk optionally applies a
     *  delta + zigzag transform, then frame-of-reference bit-packing: values
     *  are stored as (value - chunk minimum) using the smallest bit width that
     *  fits the chunk. Sorted, slowly varying or small-range data compresses
     *  well; random 32-bit data does not grow by more than the chunk headers.
     *
     *  The device layout is self-describing: 3 words per chunk (base, bit
     *  width, word offset of the packed data) followed by the packed words.
     *  The format is specific to this SDK and not compatible with nvcomp.
     */
    struct bitpack_options
    {
        //! Encode differences between consecutive elements first.
        bool delta = false;
        //! Elements per chunk; a power of two between 256 and 65536.
        std::size_t chunk_size = 2048;
    };

    struct COMPUTE_EXPORT compressed_buffer
    {
        device_buffer<cl_uint> data;
        std::size_t num_elements = 0;
        std::size_t chunk_size = 0;
        bool delta = false;

        std::size_t num_chunks() const
        {
            return chunk_size == 0
                ? 0
                : (num_elements + chunk_size - 1) / chunk_size;
        }
        std::size_t compressed_bytes() const { return data.bytes(); }
        std::size_t uncompressed_bytes() const
        {
            return num_elements * sizeof(cl_uint);
        }

        //! Host blob with a small header, suitable for files or the network.
        std::vector<unsigned char> serialize(compute_context& ctx) const;
        static compressed_buffer
        deserialize(compute_context& ctx,
                    const std::vector<unsigned char>& blob);
    };

    COMPUTE_EXPORT compressed_buffer
    compress_bitpack(compute_context& ctx, const device_buffer<cl_uint>& in,
                     bitpack_options options = bitpack_options());

    //! Decompresses into out, which must hold num_elements values.
    COMPUTE_EXPORT void decompress_bitpack(compute_context& ctx,
                                           const compressed_buffer& in,
                                           device_buffer<cl_uint>& out);

}
}
