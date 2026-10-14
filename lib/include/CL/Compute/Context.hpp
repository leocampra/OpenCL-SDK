#pragma once

// OpenCL Compute includes
#include "OpenCLCompute_Export.h"
#include <CL/Compute/MemoryResource.hpp>

// OpenCL includes
#include <CL/opencl.hpp>

// STL includes
#include <memory>

namespace cl {
namespace compute {

    class program_cache;

    /*! \brief Execution environment for Compute algorithms: one device, one
     *  in-order queue, a memory resource and a program cache.
     *
     *  Plays the role of the CUDA stream plus the current device resource in
     *  Thrust/CUB/RMM: every algorithm enqueues its work on queue() and draws
     *  temporary storage from resource().
     */
    class COMPUTE_EXPORT compute_context {
    public:
        /*! \param queue in-order command queue all work is enqueued on.
         *  \param resource memory resource for temporaries; defaults to a
         *         pool_resource over a buffer_resource.
         *  \param cache program cache; defaults to a fresh cache for the
         *         queue's context.
         */
        explicit compute_context(
            cl::CommandQueue queue,
            std::shared_ptr<memory_resource> resource = nullptr,
            std::shared_ptr<program_cache> cache = nullptr);

        const cl::Context& context() const { return context_; }
        const cl::Device& device() const { return device_; }
        cl::CommandQueue& queue() { return queue_; }

        const std::shared_ptr<memory_resource>& resource() const
        {
            return resource_;
        }
        program_cache& cache() const { return *cache_; }

        //! Largest power-of-two work-group size usable with kernel, capped
        //! at limit.
        std::size_t work_group_size(const cl::Kernel& kernel,
                                    std::size_t limit = 256) const;

    private:
        cl::CommandQueue queue_;
        cl::Context context_;
        cl::Device device_;
        std::shared_ptr<memory_resource> resource_;
        std::shared_ptr<program_cache> cache_;
    };

}
}
