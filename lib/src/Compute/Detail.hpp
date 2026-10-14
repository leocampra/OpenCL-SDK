#pragma once

// OpenCL Compute includes
#include <CL/Compute/Algorithm.hpp>
#include <CL/Compute/Context.hpp>

// STL includes
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>

namespace cl {
namespace compute {
    namespace internal {

        //! Kernels index with 32-bit unsigned integers.
        inline void check_size(std::size_t n)
        {
            if (n > 0x7FFFFFFFu)
                throw std::length_error(
                    "OpenCL Compute: at most 2^31-1 elements per call");
        }

        //! Extension pragmas needed by the given element types.
        inline std::string
        preamble(std::initializer_list<detail::type_info> types)
        {
            for (const auto& t : types)
                if (std::string(t.name) == "double")
                    return "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
            return std::string();
        }

        //! RAII temporary allocation from the context's memory resource
        //! (CUB's d_temp_storage, managed for the caller).
        class temp_buffer {
        public:
            temp_buffer(const compute_context& ctx, std::size_t bytes)
                : resource_(ctx.resource()), bytes_(bytes)
            {
                if (bytes_ != 0) buffer_ = resource_->allocate(bytes_);
            }
            ~temp_buffer()
            {
                if (bytes_ != 0) resource_->deallocate(buffer_, bytes_);
            }
            temp_buffer(const temp_buffer&) = delete;
            temp_buffer& operator=(const temp_buffer&) = delete;

            const cl::Buffer& get() const { return buffer_; }

        private:
            std::shared_ptr<memory_resource> resource_;
            cl::Buffer buffer_;
            std::size_t bytes_;
        };

    }
}
}
