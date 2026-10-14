// OpenCL Compute includes
#include <CL/Compute/FileIO.hpp>

// STL includes
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <vector>

#include <cerrno>
#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace cl {
namespace compute {

    namespace {

        class thread_pool {
        public:
            explicit thread_pool(unsigned count)
            {
                for (unsigned i = 0; i < count; ++i)
                    workers_.emplace_back([this] { run(); });
            }
            ~thread_pool()
            {
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    stopping_ = true;
                }
                cv_.notify_all();
                for (auto& worker : workers_) worker.join();
            }

            std::future<std::size_t> submit(std::function<std::size_t()> task)
            {
                auto packaged =
                    std::make_shared<std::packaged_task<std::size_t()>>(
                        std::move(task));
                std::future<std::size_t> result = packaged->get_future();
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    queue_.emplace_back([packaged] { (*packaged)(); });
                }
                cv_.notify_one();
                return result;
            }

        private:
            void run()
            {
                for (;;)
                {
                    std::function<void()> job;
                    {
                        std::unique_lock<std::mutex> lock(mutex_);
                        cv_.wait(lock, [this] {
                            return stopping_ || !queue_.empty();
                        });
                        if (queue_.empty()) return;
                        job = std::move(queue_.front());
                        queue_.pop_front();
                    }
                    job();
                }
            }

            std::vector<std::thread> workers_;
            std::deque<std::function<void()>> queue_;
            std::mutex mutex_;
            std::condition_variable cv_;
            bool stopping_ = false;
        };

        unsigned default_threads()
        {
            const char* env = std::getenv("CL_COMPUTE_NTHREADS");
            if (env != nullptr)
            {
                const long value = std::strtol(env, nullptr, 10);
                if (value > 0 && value <= 1024)
                    return static_cast<unsigned>(value);
            }
            return 4;
        }

        std::mutex pool_mutex;
        unsigned pool_threads = 0;
        std::unique_ptr<thread_pool> pool;

        thread_pool& shared_pool()
        {
            std::lock_guard<std::mutex> lock(pool_mutex);
            if (!pool)
            {
                if (pool_threads == 0) pool_threads = default_threads();
                pool.reset(new thread_pool(pool_threads));
            }
            return *pool;
        }

        // Splits [0, size) into task_size pieces and sums the bytes moved.
        // The aggregate future is deferred: it joins the pieces on get().
        std::future<std::size_t>
        parallel_io(std::size_t size, std::size_t task_size,
                    std::function<std::size_t(std::size_t, std::size_t)> task)
        {
            task_size = std::max<std::size_t>(task_size, 1);
            auto parts =
                std::make_shared<std::vector<std::future<std::size_t>>>();
            thread_pool& workers = shared_pool();
            for (std::size_t offset = 0; offset < size; offset += task_size)
            {
                const std::size_t length = std::min(task_size, size - offset);
                parts->push_back(workers.submit(
                    [task, offset, length] { return task(offset, length); }));
            }
            return std::async(std::launch::deferred, [parts] {
                std::size_t total = 0;
                std::exception_ptr error;
                // Wait for every piece even after a failure, so no task
                // outlives the caller's buffers.
                for (auto& part : *parts)
                {
                    try
                    {
                        total += part.get();
                    } catch (...)
                    {
                        if (!error) error = std::current_exception();
                    }
                }
                if (error) std::rethrow_exception(error);
                return total;
            });
        }
    }

    void set_io_threads(unsigned count)
    {
        if (count == 0)
            throw std::invalid_argument("set_io_threads: count must be > 0");
        std::lock_guard<std::mutex> lock(pool_mutex);
        pool_threads = count;
    }

    unsigned io_threads()
    {
        std::lock_guard<std::mutex> lock(pool_mutex);
        return pool_threads == 0 ? default_threads() : pool_threads;
    }

#if defined(_WIN32)
    // Windows: a FILE* guarded by a mutex. Correct, but transfers on the
    // same handle are serialized.
    struct file_handle::impl
    {
        std::FILE* file = nullptr;
        std::mutex mutex;

        impl(const std::string& path, const std::string& flags)
        {
            std::string mode = flags + "b";
            file = std::fopen(path.c_str(), mode.c_str());
            if (file == nullptr)
                throw std::system_error(errno, std::generic_category(),
                                        "file_handle: cannot open " + path);
        }
        ~impl()
        {
            if (file != nullptr) std::fclose(file);
        }
        std::size_t size()
        {
            std::lock_guard<std::mutex> lock(mutex);
            _fseeki64(file, 0, SEEK_END);
            return static_cast<std::size_t>(_ftelli64(file));
        }
        std::size_t pread(void* dst, std::size_t length, std::size_t offset)
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (_fseeki64(file, static_cast<long long>(offset), SEEK_SET) != 0)
                throw std::system_error(errno, std::generic_category(),
                                        "file_handle: seek failed");
            const std::size_t got = std::fread(dst, 1, length, file);
            if (got < length && std::ferror(file))
                throw std::system_error(errno, std::generic_category(),
                                        "file_handle: read failed");
            return got;
        }
        std::size_t pwrite(const void* src, std::size_t length,
                           std::size_t offset)
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (_fseeki64(file, static_cast<long long>(offset), SEEK_SET) != 0)
                throw std::system_error(errno, std::generic_category(),
                                        "file_handle: seek failed");
            const std::size_t put = std::fwrite(src, 1, length, file);
            if (put < length)
                throw std::system_error(errno, std::generic_category(),
                                        "file_handle: write failed");
            std::fflush(file);
            return put;
        }
    };
#else
    // POSIX: pread/pwrite are positional, so tasks run concurrently.
    struct file_handle::impl
    {
        int fd = -1;

        impl(const std::string& path, const std::string& flags)
        {
            int oflags;
            if (flags == "r")
                oflags = O_RDONLY;
            else if (flags == "w")
                oflags = O_WRONLY | O_CREAT | O_TRUNC;
            else if (flags == "r+")
                oflags = O_RDWR;
            else if (flags == "w+")
                oflags = O_RDWR | O_CREAT | O_TRUNC;
            else
                throw std::invalid_argument("file_handle: unknown flags "
                                            + flags);
            fd = ::open(path.c_str(), oflags | O_CLOEXEC, 0644);
            if (fd < 0)
                throw std::system_error(errno, std::generic_category(),
                                        "file_handle: cannot open " + path);
        }
        ~impl()
        {
            if (fd >= 0) ::close(fd);
        }
        std::size_t size()
        {
            struct stat st;
            if (::fstat(fd, &st) != 0)
                throw std::system_error(errno, std::generic_category(),
                                        "file_handle: fstat failed");
            return static_cast<std::size_t>(st.st_size);
        }
        std::size_t pread(void* dst, std::size_t length, std::size_t offset)
        {
            std::size_t done = 0;
            while (done < length)
            {
                const ssize_t got =
                    ::pread(fd, static_cast<char*>(dst) + done, length - done,
                            static_cast<off_t>(offset + done));
                if (got < 0)
                {
                    if (errno == EINTR) continue;
                    throw std::system_error(errno, std::generic_category(),
                                            "file_handle: pread failed");
                }
                if (got == 0) break; // end of file
                done += static_cast<std::size_t>(got);
            }
            return done;
        }
        std::size_t pwrite(const void* src, std::size_t length,
                           std::size_t offset)
        {
            std::size_t done = 0;
            while (done < length)
            {
                const ssize_t put =
                    ::pwrite(fd, static_cast<const char*>(src) + done,
                             length - done, static_cast<off_t>(offset + done));
                if (put < 0)
                {
                    if (errno == EINTR) continue;
                    throw std::system_error(errno, std::generic_category(),
                                            "file_handle: pwrite failed");
                }
                done += static_cast<std::size_t>(put);
            }
            return done;
        }
    };
#endif

    file_handle::file_handle(const std::string& path, const std::string& flags,
                             io_options options)
        : impl_(std::make_shared<impl>(path, flags)), options_(options)
    {}

    file_handle::~file_handle() = default;
    file_handle::file_handle(file_handle&& other) noexcept = default;
    file_handle& file_handle::operator=(file_handle&& other) noexcept = default;

    // Pending transfers keep their own reference, so the descriptor is
    // closed once the last of them finishes.
    void file_handle::close() { impl_.reset(); }

    bool file_handle::is_open() const { return impl_ != nullptr; }

    std::size_t file_handle::nbytes() const
    {
        if (!impl_) throw std::logic_error("file_handle: closed");
        return impl_->size();
    }

    std::future<std::size_t> file_handle::pread(cl::CommandQueue queue,
                                                const cl::Buffer& buffer,
                                                std::size_t size,
                                                std::size_t file_offset,
                                                std::size_t buffer_offset)
    {
        if (!impl_) throw std::logic_error("file_handle: closed");
        std::shared_ptr<impl> file = impl_;
        const io_mode mode = options_.mode;
        return parallel_io(
            size, options_.task_size,
            [=](std::size_t offset, std::size_t length) -> std::size_t {
                cl::CommandQueue q = queue;
                if (mode == io_mode::mapped)
                {
                    void* ptr = q.enqueueMapBuffer(
                        buffer, CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION,
                        buffer_offset + offset, length);
                    std::size_t got = 0;
                    try
                    {
                        got = file->pread(ptr, length, file_offset + offset);
                    } catch (...)
                    {
                        cl::Event unmapped;
                        q.enqueueUnmapMemObject(buffer, ptr, nullptr,
                                                &unmapped);
                        unmapped.wait();
                        throw;
                    }
                    cl::Event unmapped;
                    q.enqueueUnmapMemObject(buffer, ptr, nullptr, &unmapped);
                    unmapped.wait();
                    return got;
                }
                std::vector<char> bounce(length);
                const std::size_t got =
                    file->pread(bounce.data(), length, file_offset + offset);
                if (got != 0)
                    q.enqueueWriteBuffer(buffer, CL_TRUE,
                                         buffer_offset + offset, got,
                                         bounce.data());
                return got;
            });
    }

    std::future<std::size_t> file_handle::pwrite(cl::CommandQueue queue,
                                                 const cl::Buffer& buffer,
                                                 std::size_t size,
                                                 std::size_t file_offset,
                                                 std::size_t buffer_offset)
    {
        if (!impl_) throw std::logic_error("file_handle: closed");
        std::shared_ptr<impl> file = impl_;
        const io_mode mode = options_.mode;
        return parallel_io(
            size, options_.task_size,
            [=](std::size_t offset, std::size_t length) -> std::size_t {
                cl::CommandQueue q = queue;
                if (mode == io_mode::mapped)
                {
                    void* ptr =
                        q.enqueueMapBuffer(buffer, CL_TRUE, CL_MAP_READ,
                                           buffer_offset + offset, length);
                    std::size_t put = 0;
                    try
                    {
                        put = file->pwrite(ptr, length, file_offset + offset);
                    } catch (...)
                    {
                        cl::Event unmapped;
                        q.enqueueUnmapMemObject(buffer, ptr, nullptr,
                                                &unmapped);
                        unmapped.wait();
                        throw;
                    }
                    cl::Event unmapped;
                    q.enqueueUnmapMemObject(buffer, ptr, nullptr, &unmapped);
                    unmapped.wait();
                    return put;
                }
                std::vector<char> bounce(length);
                q.enqueueReadBuffer(buffer, CL_TRUE, buffer_offset + offset,
                                    length, bounce.data());
                return file->pwrite(bounce.data(), length,
                                    file_offset + offset);
            });
    }

    std::future<std::size_t> file_handle::pread_host(void* dst,
                                                     std::size_t size,
                                                     std::size_t file_offset)
    {
        if (!impl_) throw std::logic_error("file_handle: closed");
        std::shared_ptr<impl> file = impl_;
        char* base = static_cast<char*>(dst);
        return parallel_io(size, options_.task_size,
                           [=](std::size_t offset, std::size_t length) {
                               return file->pread(base + offset, length,
                                                  file_offset + offset);
                           });
    }

    std::future<std::size_t> file_handle::pwrite_host(const void* src,
                                                      std::size_t size,
                                                      std::size_t file_offset)
    {
        if (!impl_) throw std::logic_error("file_handle: closed");
        std::shared_ptr<impl> file = impl_;
        const char* base = static_cast<const char*>(src);
        return parallel_io(size, options_.task_size,
                           [=](std::size_t offset, std::size_t length) {
                               return file->pwrite(base + offset, length,
                                                   file_offset + offset);
                           });
    }

}
}
