#pragma once

// OpenCL Compute includes
#include <CL/Compute/Context.hpp>
#include <CL/Compute/MemoryResource.hpp>

// OpenCL includes
#include <CL/opencl.hpp>

// STL includes
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cl {
namespace compute {

    /*! \brief Typed, move-only device allocation drawn from a memory_resource
     *  (rmm::device_uvector / thrust::device_vector / cuda::buffer analogue).
     *
     *  The underlying cl::Buffer may be larger than size() * sizeof(T) when it
     *  comes from a pool; kernels must always be given size() explicitly.
     */
    template <typename T> class device_buffer {
    public:
        device_buffer() = default;

        device_buffer(std::shared_ptr<memory_resource> resource,
                      std::size_t size)
            : resource_(std::move(resource)), size_(size)
        {
            if (size_ != 0) buffer_ = resource_->allocate(bytes());
        }

        device_buffer(const compute_context& ctx, std::size_t size)
            : device_buffer(ctx.resource(), size)
        {}

        device_buffer(const device_buffer&) = delete;
        device_buffer& operator=(const device_buffer&) = delete;

        device_buffer(device_buffer&& other) noexcept
            : resource_(std::move(other.resource_)),
              buffer_(std::move(other.buffer_)), size_(other.size_)
        {
            other.size_ = 0;
            other.buffer_ = cl::Buffer();
        }

        device_buffer& operator=(device_buffer&& other) noexcept
        {
            if (this != &other)
            {
                reset();
                resource_ = std::move(other.resource_);
                buffer_ = std::move(other.buffer_);
                size_ = other.size_;
                other.size_ = 0;
                other.buffer_ = cl::Buffer();
            }
            return *this;
        }

        ~device_buffer() { reset(); }

        //! Returns the allocation to its resource.
        void reset() noexcept
        {
            if (size_ != 0 && resource_)
                resource_->deallocate(buffer_, bytes());
            buffer_ = cl::Buffer();
            size_ = 0;
        }

        std::size_t size() const { return size_; }
        std::size_t bytes() const { return size_ * sizeof(T); }
        bool empty() const { return size_ == 0; }
        const cl::Buffer& get() const { return buffer_; }
        const std::shared_ptr<memory_resource>& resource() const
        {
            return resource_;
        }

        void copy_from_host(cl::CommandQueue& queue, const T* src,
                            std::size_t count, std::size_t offset = 0,
                            bool blocking = true)
        {
            check_range(offset, count);
            if (count != 0)
                queue.enqueueWriteBuffer(buffer_, blocking ? CL_TRUE : CL_FALSE,
                                         offset * sizeof(T), count * sizeof(T),
                                         src);
        }

        void copy_to_host(cl::CommandQueue& queue, T* dst, std::size_t count,
                          std::size_t offset = 0, bool blocking = true) const
        {
            check_range(offset, count);
            if (count != 0)
                queue.enqueueReadBuffer(buffer_, blocking ? CL_TRUE : CL_FALSE,
                                        offset * sizeof(T), count * sizeof(T),
                                        dst);
        }

        std::vector<T> to_host(compute_context& ctx) const
        {
            std::vector<T> result(size_);
            copy_to_host(ctx.queue(), result.data(), size_);
            return result;
        }

        static device_buffer from_host(compute_context& ctx,
                                       const std::vector<T>& values)
        {
            device_buffer result(ctx, values.size());
            result.copy_from_host(ctx.queue(), values.data(), values.size());
            return result;
        }

    private:
        void check_range(std::size_t offset, std::size_t count) const
        {
            if (offset > size_ || count > size_ - offset)
                throw std::out_of_range("device_buffer: range out of bounds");
        }

        std::shared_ptr<memory_resource> resource_;
        cl::Buffer buffer_;
        std::size_t size_ = 0;
    };

}
}
