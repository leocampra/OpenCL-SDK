#pragma once

// OpenCL Compute includes
#include "OpenCLCompute_Export.h"

// OpenCL includes
#include <CL/opencl.hpp>

// STL includes
#include <cstddef>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

namespace cl {
namespace compute {

    /*! \brief Abstract device memory resource (RMM / thrust::mr analogue).
     *
     *  OpenCL has no raw device pointers outside of SVM, so allocations are
     *  handed out as cl::Buffer objects. Callers must pass the same byte count
     *  to deallocate() that they passed to allocate().
     *
     *  Allocations follow queue-ordered semantics, like RMM's stream-ordered
     *  resources: a buffer handed back by deallocate() may be returned by the
     *  next allocate() immediately, which is safe as long as every command
     *  touching it was enqueued on the same in-order command queue. When
     *  several queues share a resource, synchronize before deallocating.
     */
    class COMPUTE_EXPORT memory_resource {
    public:
        virtual ~memory_resource() = default;

        cl::Buffer allocate(std::size_t bytes) { return do_allocate(bytes); }
        void deallocate(const cl::Buffer& buffer, std::size_t bytes) noexcept
        {
            do_deallocate(buffer, bytes);
        }

        virtual const cl::Context& context() const = 0;

    protected:
        virtual cl::Buffer do_allocate(std::size_t bytes) = 0;
        virtual void do_deallocate(const cl::Buffer& buffer,
                                   std::size_t bytes) noexcept = 0;
    };

    /*! \brief Upstream resource calling clCreateBuffer for every allocation
     *  (rmm::mr::cuda_memory_resource analogue).
     */
    class COMPUTE_EXPORT buffer_resource final : public memory_resource {
    public:
        explicit buffer_resource(cl::Context context,
                                 cl_mem_flags flags = CL_MEM_READ_WRITE);

        const cl::Context& context() const override { return context_; }

    protected:
        cl::Buffer do_allocate(std::size_t bytes) override;
        void do_deallocate(const cl::Buffer& buffer,
                           std::size_t bytes) noexcept override;

    private:
        cl::Context context_;
        cl_mem_flags flags_;
    };

    /*! \brief Caching pool on top of an upstream resource
     *  (rmm::mr::pool_memory_resource / thrust::mr::pool_resource analogue).
     *
     *  Requests are rounded up to a size class (powers of two up to
     *  pow2_limit, then multiples of large_granularity) and freed blocks are
     *  kept in per-class free lists until max_cached_bytes is reached. When
     *  the upstream runs out of memory the cache is released and the
     *  allocation retried once. Thread-safe.
     */
    class COMPUTE_EXPORT pool_resource final : public memory_resource {
    public:
        struct options
        {
            std::size_t min_block_bytes = 256;
            std::size_t pow2_limit = std::size_t(16) << 20;
            std::size_t large_granularity = std::size_t(1) << 20;
            std::size_t max_cached_bytes = std::size_t(1) << 30;
        };

        explicit pool_resource(std::shared_ptr<memory_resource> upstream);
        pool_resource(std::shared_ptr<memory_resource> upstream, options opts);
        ~pool_resource();

        const cl::Context& context() const override
        {
            return upstream_->context();
        }

        //! Returns every cached block to the upstream resource.
        void release();

        std::size_t cached_bytes() const;
        std::size_t size_class(std::size_t bytes) const;

    protected:
        cl::Buffer do_allocate(std::size_t bytes) override;
        void do_deallocate(const cl::Buffer& buffer,
                           std::size_t bytes) noexcept override;

    private:
        void release_locked();

        std::shared_ptr<memory_resource> upstream_;
        options opts_;
        mutable std::mutex mutex_;
        std::map<std::size_t, std::vector<cl::Buffer>> free_lists_;
        std::size_t cached_bytes_ = 0;
    };

    /*! \brief Adaptor tracking allocation statistics
     *  (rmm::mr::statistics_resource_adaptor analogue). Thread-safe.
     */
    class COMPUTE_EXPORT statistics_adaptor final : public memory_resource {
    public:
        struct counters
        {
            std::size_t current_bytes = 0;
            std::size_t peak_bytes = 0;
            std::size_t total_bytes = 0;
            std::size_t current_allocations = 0;
            std::size_t total_allocations = 0;
        };

        explicit statistics_adaptor(std::shared_ptr<memory_resource> upstream);

        const cl::Context& context() const override
        {
            return upstream_->context();
        }

        counters statistics() const;

    protected:
        cl::Buffer do_allocate(std::size_t bytes) override;
        void do_deallocate(const cl::Buffer& buffer,
                           std::size_t bytes) noexcept override;

    private:
        std::shared_ptr<memory_resource> upstream_;
        mutable std::mutex mutex_;
        counters counters_;
    };

}
}
