#pragma once

// OpenCL Compute includes
#include "OpenCLCompute_Export.h"

// OpenCL includes
#include <CL/opencl.hpp>

// STL includes
#include <cstddef>
#include <future>
#include <memory>
#include <string>

namespace cl {
namespace compute {

    /*! \brief Data path used to move bytes between a file and a cl::Buffer.
     *
     *  mapped maps each chunk of the buffer with clEnqueueMapBuffer and reads
     *  or writes the file straight into the mapping, so on host-visible
     *  memory (CPUs, integrated GPUs, ALLOC_HOST_PTR buffers) the data never
     *  passes through an extra copy. It is the closest OpenCL equivalent of
     *  the cuFile/GDS direct path.
     *
     *  bounce reads into a host staging buffer and copies it with
     *  clEnqueueWriteBuffer (and the reverse for writes), like KvikIO's POSIX
     *  compatibility mode and cuFile's compat mode.
     */
    enum class io_mode
    {
        mapped,
        bounce
    };

    struct io_options
    {
        io_mode mode = io_mode::mapped;
        //! Bytes per parallel task (KVIKIO_TASK_SIZE analogue).
        std::size_t task_size = std::size_t(4) << 20;
    };

    //! Sets the number of I/O threads (KVIKIO_NTHREADS analogue). Defaults to
    //! the CL_COMPUTE_NTHREADS environment variable, or 4. Takes effect when
    //! the pool is created, i.e. before the first I/O call.
    COMPUTE_EXPORT void set_io_threads(unsigned count);
    COMPUTE_EXPORT unsigned io_threads();

    /*! \brief File handle for parallel file <-> device transfers
     *  (kvikio::FileHandle analogue).
     *
     *  Large requests are split into task_size pieces that run on a shared
     *  thread pool; the returned futures resolve to the number of bytes
     *  transferred, which is smaller than requested only at end of file.
     *  Transfers are enqueued on the given queue, so they are ordered after
     *  work already submitted to it.
     */
    class COMPUTE_EXPORT file_handle {
    public:
        /*! \param flags "r" read, "w" create/truncate, "r+" read/write,
         *         "w+" create/truncate read/write.
         */
        file_handle(const std::string& path, const std::string& flags = "r",
                    io_options options = io_options());
        ~file_handle();

        file_handle(const file_handle&) = delete;
        file_handle& operator=(const file_handle&) = delete;
        file_handle(file_handle&& other) noexcept;
        file_handle& operator=(file_handle&& other) noexcept;

        std::size_t nbytes() const;
        const io_options& options() const { return options_; }
        void close();
        bool is_open() const;

        std::future<std::size_t> pread(cl::CommandQueue queue,
                                       const cl::Buffer& buffer,
                                       std::size_t size,
                                       std::size_t file_offset = 0,
                                       std::size_t buffer_offset = 0);
        std::future<std::size_t> pwrite(cl::CommandQueue queue,
                                        const cl::Buffer& buffer,
                                        std::size_t size,
                                        std::size_t file_offset = 0,
                                        std::size_t buffer_offset = 0);

        std::size_t read(cl::CommandQueue queue, const cl::Buffer& buffer,
                         std::size_t size, std::size_t file_offset = 0,
                         std::size_t buffer_offset = 0)
        {
            return pread(queue, buffer, size, file_offset, buffer_offset).get();
        }
        std::size_t write(cl::CommandQueue queue, const cl::Buffer& buffer,
                          std::size_t size, std::size_t file_offset = 0,
                          std::size_t buffer_offset = 0)
        {
            return pwrite(queue, buffer, size, file_offset, buffer_offset)
                .get();
        }

        //! Parallel host-memory transfers sharing the same thread pool.
        std::future<std::size_t> pread_host(void* dst, std::size_t size,
                                            std::size_t file_offset = 0);
        std::future<std::size_t> pwrite_host(const void* src, std::size_t size,
                                             std::size_t file_offset = 0);

    private:
        struct impl;
        std::shared_ptr<impl> impl_;
        io_options options_;
    };

}
}
