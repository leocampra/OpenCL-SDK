// OpenCL Compute includes
#include <CL/Compute/MemoryResource.hpp>
#include <CL/Compute/Context.hpp>
#include <CL/Compute/ProgramCache.hpp>

// STL includes
#include <algorithm>

namespace cl {
namespace compute {

    buffer_resource::buffer_resource(cl::Context context, cl_mem_flags flags)
        : context_(std::move(context)), flags_(flags)
    {}

    cl::Buffer buffer_resource::do_allocate(std::size_t bytes)
    {
        // clCreateBuffer rejects zero-sized buffers.
        return cl::Buffer(context_, flags_, std::max<std::size_t>(bytes, 1));
    }

    void buffer_resource::do_deallocate(const cl::Buffer&, std::size_t) noexcept
    {
        // The cl::Buffer handle releases the cl_mem when its last copy dies.
    }

    pool_resource::pool_resource(std::shared_ptr<memory_resource> upstream)
        : pool_resource(std::move(upstream), options())
    {}

    pool_resource::pool_resource(std::shared_ptr<memory_resource> upstream,
                                 options opts)
        : upstream_(std::move(upstream)), opts_(opts)
    {
        if (!upstream_)
            throw std::invalid_argument("pool_resource: null upstream");
        if (opts_.min_block_bytes == 0 || opts_.large_granularity == 0)
            throw std::invalid_argument("pool_resource: zero block size");
    }

    pool_resource::~pool_resource()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        release_locked();
    }

    std::size_t pool_resource::size_class(std::size_t bytes) const
    {
        bytes = std::max(bytes, opts_.min_block_bytes);
        if (bytes <= opts_.pow2_limit)
        {
            std::size_t size = 1;
            while (size < bytes) size <<= 1;
            return size;
        }
        const std::size_t g = opts_.large_granularity;
        return (bytes + g - 1) / g * g;
    }

    cl::Buffer pool_resource::do_allocate(std::size_t bytes)
    {
        const std::size_t size = size_class(bytes);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = free_lists_.find(size);
            if (it != free_lists_.end() && !it->second.empty())
            {
                cl::Buffer buffer = std::move(it->second.back());
                it->second.pop_back();
                cached_bytes_ -= size;
                return buffer;
            }
        }
        try
        {
            return upstream_->allocate(size);
        } catch (const cl::Error& e)
        {
            if (e.err() != CL_MEM_OBJECT_ALLOCATION_FAILURE
                && e.err() != CL_OUT_OF_RESOURCES
                && e.err() != CL_INVALID_BUFFER_SIZE)
                throw;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (cached_bytes_ == 0) throw;
                release_locked();
            }
            return upstream_->allocate(size);
        }
    }

    void pool_resource::do_deallocate(const cl::Buffer& buffer,
                                      std::size_t bytes) noexcept
    {
        const std::size_t size = size_class(bytes);
        std::lock_guard<std::mutex> lock(mutex_);
        if (cached_bytes_ + size <= opts_.max_cached_bytes)
        {
            try
            {
                free_lists_[size].push_back(buffer);
                cached_bytes_ += size;
                return;
            } catch (...)
            {
                // Out of host memory: fall through and hand it back.
            }
        }
        upstream_->deallocate(buffer, size);
    }

    void pool_resource::release()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        release_locked();
    }

    void pool_resource::release_locked()
    {
        for (auto& entry : free_lists_)
            for (auto& buffer : entry.second)
                upstream_->deallocate(buffer, entry.first);
        free_lists_.clear();
        cached_bytes_ = 0;
    }

    std::size_t pool_resource::cached_bytes() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return cached_bytes_;
    }

    statistics_adaptor::statistics_adaptor(
        std::shared_ptr<memory_resource> upstream)
        : upstream_(std::move(upstream))
    {
        if (!upstream_)
            throw std::invalid_argument("statistics_adaptor: null upstream");
    }

    cl::Buffer statistics_adaptor::do_allocate(std::size_t bytes)
    {
        cl::Buffer buffer = upstream_->allocate(bytes);
        std::lock_guard<std::mutex> lock(mutex_);
        counters_.current_bytes += bytes;
        counters_.peak_bytes =
            std::max(counters_.peak_bytes, counters_.current_bytes);
        counters_.total_bytes += bytes;
        ++counters_.current_allocations;
        ++counters_.total_allocations;
        return buffer;
    }

    void statistics_adaptor::do_deallocate(const cl::Buffer& buffer,
                                           std::size_t bytes) noexcept
    {
        upstream_->deallocate(buffer, bytes);
        std::lock_guard<std::mutex> lock(mutex_);
        counters_.current_bytes -= bytes;
        --counters_.current_allocations;
    }

    statistics_adaptor::counters statistics_adaptor::statistics() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return counters_;
    }

    compute_context::compute_context(cl::CommandQueue queue,
                                     std::shared_ptr<memory_resource> resource,
                                     std::shared_ptr<program_cache> cache)
        : queue_(std::move(queue)),
          context_(queue_.getInfo<CL_QUEUE_CONTEXT>()),
          device_(queue_.getInfo<CL_QUEUE_DEVICE>()),
          resource_(std::move(resource)), cache_(std::move(cache))
    {
        if (queue_.getInfo<CL_QUEUE_PROPERTIES>()
            & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)
            throw std::invalid_argument(
                "compute_context: the command queue must be in-order");
        if (!resource_)
            resource_ = std::make_shared<pool_resource>(
                std::make_shared<buffer_resource>(context_));
        if (!cache_) cache_ = std::make_shared<program_cache>(context_);
    }

    std::size_t compute_context::work_group_size(const cl::Kernel& kernel,
                                                 std::size_t limit) const
    {
        std::size_t max = std::min<std::size_t>(
            limit, kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device_));
        max = std::min<std::size_t>(
            max, device_.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>().at(0));
        std::size_t size = 1;
        while (size * 2 <= max) size *= 2;
        return size;
    }

}
}
