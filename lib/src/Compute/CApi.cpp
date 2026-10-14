// OpenCL Compute includes
#include <CL/Compute/Compute.hpp>
#include <CL/Compute/compute_c.h>

// STL includes
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

struct clc_context_s
{
    cl::compute::compute_context ctx;
    bool has_fp64;
};

struct clc_buffer_s
{
    std::shared_ptr<cl::compute::memory_resource> resource;
    cl::Buffer buffer;
    std::size_t bytes;
};

struct clc_kernel_s
{
    cl::Kernel kernel;
};

namespace {

// Errors with a status of their own; everything else is mapped by type.
struct no_device_error : std::runtime_error
{
    using std::runtime_error::runtime_error;
};
struct unsupported_error : std::runtime_error
{
    using std::runtime_error::runtime_error;
};
struct io_error : std::runtime_error
{
    using std::runtime_error::runtime_error;
};

thread_local std::string last_error;
thread_local int last_cl_error = 0;

clc_status fail(clc_status status, const std::string& message, int cl_err = 0)
{
    last_error = message;
    last_cl_error = cl_err;
    return status;
}

// Runs body, translating every exception into a status code.
template <typename F> clc_status guarded(F&& body)
{
    try
    {
        body();
        last_error.clear();
        last_cl_error = 0;
        return CLC_SUCCESS;
    } catch (const no_device_error& e)
    {
        return fail(CLC_ERROR_NO_DEVICE, e.what());
    } catch (const unsupported_error& e)
    {
        return fail(CLC_ERROR_UNSUPPORTED, e.what());
    } catch (const io_error& e)
    {
        return fail(CLC_ERROR_IO, e.what());
    } catch (const std::system_error& e)
    {
        return fail(CLC_ERROR_IO, e.what());
    } catch (const cl::compute::build_error& e)
    {
        return fail(CLC_ERROR_BUILD, e.what(), e.err());
    } catch (const cl::Error& e)
    {
        const int err = e.err();
        const std::string msg =
            std::string(e.what()) + " failed (" + std::to_string(err) + ")";
        if (err == CL_MEM_OBJECT_ALLOCATION_FAILURE
            || err == CL_OUT_OF_RESOURCES || err == CL_OUT_OF_HOST_MEMORY
            || err == CL_INVALID_BUFFER_SIZE)
            return fail(CLC_ERROR_OUT_OF_MEMORY, msg, err);
        return fail(CLC_ERROR_OPENCL, msg, err);
    } catch (const std::bad_alloc&)
    {
        return fail(CLC_ERROR_OUT_OF_MEMORY, "host allocation failed");
    } catch (const std::invalid_argument& e)
    {
        return fail(CLC_ERROR_INVALID_ARGUMENT, e.what());
    } catch (const std::out_of_range& e)
    {
        return fail(CLC_ERROR_INVALID_ARGUMENT, e.what());
    } catch (const std::length_error& e)
    {
        return fail(CLC_ERROR_INVALID_ARGUMENT, e.what());
    } catch (const std::overflow_error& e)
    {
        return fail(CLC_ERROR_OUT_OF_MEMORY, e.what());
    } catch (const std::exception& e)
    {
        return fail(CLC_ERROR_INTERNAL, e.what());
    } catch (...)
    {
        return fail(CLC_ERROR_INTERNAL, "unknown error");
    }
}

// Platform enumeration and context creation are serialized: concurrent first
// calls into some ICD loaders and drivers (ocl-icd 2.2 with PoCL, at least)
// crash while they initialize.
std::mutex& init_mutex()
{
    static std::mutex m;
    return m;
}

std::vector<cl::Device> all_devices()
{
    std::lock_guard<std::mutex> lock(init_mutex());
    std::vector<cl::Device> result;
    std::vector<cl::Platform> platforms;
    try
    {
        cl::Platform::get(&platforms);
    } catch (const cl::Error& e)
    {
        // The ICD loader reports "no platform" as an error.
        if (e.err() == CL_PLATFORM_NOT_FOUND_KHR) return result;
        throw;
    }
    for (const auto& platform : platforms)
    {
        std::vector<cl::Device> devices;
        try
        {
            platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        } catch (const cl::Error& e)
        {
            if (e.err() == CL_DEVICE_NOT_FOUND) continue;
            throw;
        }
        result.insert(result.end(), devices.begin(), devices.end());
    }
    return result;
}

bool device_has_fp64(const cl::Device& device)
{
    return device.getInfo<CL_DEVICE_EXTENSIONS>().find("cl_khr_fp64")
        != std::string::npos
        || device.getInfo<CL_DEVICE_DOUBLE_FP_CONFIG>() != 0;
}

void copy_text(char (&dst)[256], const std::string& src)
{
    std::size_t n = src.size() < 255 ? src.size() : 255;
    std::memcpy(dst, src.data(), n);
    dst[n] = '\0';
}

void fill_info(const cl::Device& device, clc_device_info* info)
{
    cl::Platform platform(device.getInfo<CL_DEVICE_PLATFORM>());
    copy_text(info->name, device.getInfo<CL_DEVICE_NAME>());
    copy_text(info->platform, platform.getInfo<CL_PLATFORM_NAME>());
    info->is_gpu = (device.getInfo<CL_DEVICE_TYPE>() & CL_DEVICE_TYPE_GPU) != 0;
    info->has_fp64 = device_has_fp64(device) ? 1 : 0;
    info->global_memory = device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
    info->max_work_group_size = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
}

cl::compute::detail::type_info type_of(clc_type type)
{
    switch (type)
    {
        case CLC_TYPE_I8: return { "char", 1 };
        case CLC_TYPE_U8: return { "uchar", 1 };
        case CLC_TYPE_I16: return { "short", 2 };
        case CLC_TYPE_U16: return { "ushort", 2 };
        case CLC_TYPE_I32: return { "int", 4 };
        case CLC_TYPE_U32: return { "uint", 4 };
        case CLC_TYPE_I64: return { "long", 8 };
        case CLC_TYPE_U64: return { "ulong", 8 };
        case CLC_TYPE_F32: return { "float", 4 };
        case CLC_TYPE_F64: return { "double", 8 };
    }
    throw std::invalid_argument("clc: unknown element type");
}

// Validates handles and that n elements of each type fit in their buffers.
void require(bool condition, const char* message)
{
    if (!condition) throw std::invalid_argument(message);
}

void require_fp64(clc_context ctx, std::initializer_list<clc_type> types)
{
    for (clc_type t : types)
        if (t == CLC_TYPE_F64 && !ctx->has_fp64)
            throw unsupported_error(
                "clc: device does not support double (cl_khr_fp64)");
}

void require_fits(clc_buffer buffer, std::size_t n, clc_type type)
{
    require(buffer != nullptr, "clc: null buffer");
    require(n <= buffer->bytes / type_of(type).size,
            "clc: buffer too small for the element count");
}

}

extern "C" {

const char* clc_last_error(void) { return last_error.c_str(); }

int clc_last_cl_error(void) { return last_cl_error; }

clc_status clc_device_count(size_t* count)
{
    if (count == nullptr)
        return fail(CLC_ERROR_INVALID_ARGUMENT, "clc: null count");
    return guarded([&] { *count = all_devices().size(); });
}

clc_status clc_device_get_info(size_t index, clc_device_info* info)
{
    if (info == nullptr)
        return fail(CLC_ERROR_INVALID_ARGUMENT, "clc: null info");
    return guarded([&] {
        auto devices = all_devices();
        require(index < devices.size(), "clc: device index out of range");
        fill_info(devices[index], info);
    });
}

clc_status clc_context_create(size_t index, clc_context* out)
{
    return clc_context_create_with_cache(index, nullptr, out);
}

clc_status clc_context_create_with_cache(size_t index, const char* cache_dir,
                                         clc_context* out)
{
    if (out == nullptr)
        return fail(CLC_ERROR_INVALID_ARGUMENT, "clc: null output");
    *out = nullptr;
    return guarded([&] {
        auto devices = all_devices();
        if (devices.empty())
            throw no_device_error("clc: no OpenCL device available");
        if (index == CLC_DEFAULT_DEVICE)
        {
            index = 0;
            for (std::size_t i = 0; i < devices.size(); ++i)
                if (devices[i].getInfo<CL_DEVICE_TYPE>() & CL_DEVICE_TYPE_GPU)
                {
                    index = i;
                    break;
                }
        }
        if (index >= devices.size())
            throw no_device_error("clc: device index out of range");
        const cl::Device& device = devices[index];
        std::lock_guard<std::mutex> lock(init_mutex());
        cl::Context context(device);
        cl::CommandQueue queue(context, device);
        auto cache = std::make_shared<cl::compute::program_cache>(
            context, cache_dir != nullptr ? cache_dir : "");
        *out = new clc_context_s{ cl::compute::compute_context(queue, nullptr,
                                                               cache),
                                  device_has_fp64(device) };
    });
}

void clc_context_release(clc_context ctx) { delete ctx; }

clc_status clc_context_get_info(clc_context ctx, clc_device_info* info)
{
    if (ctx == nullptr || info == nullptr)
        return fail(CLC_ERROR_INVALID_ARGUMENT, "clc: null argument");
    return guarded([&] { fill_info(ctx->ctx.device(), info); });
}

clc_status clc_finish(clc_context ctx)
{
    if (ctx == nullptr)
        return fail(CLC_ERROR_INVALID_ARGUMENT, "clc: null context");
    return guarded([&] { ctx->ctx.queue().finish(); });
}

clc_status clc_buffer_alloc(clc_context ctx, size_t bytes, clc_buffer* out)
{
    if (ctx == nullptr || out == nullptr)
        return fail(CLC_ERROR_INVALID_ARGUMENT, "clc: null argument");
    *out = nullptr;
    return guarded([&] {
        std::unique_ptr<clc_buffer_s> buffer(
            new clc_buffer_s{ ctx->ctx.resource(), cl::Buffer(), bytes });
        if (bytes != 0) buffer->buffer = buffer->resource->allocate(bytes);
        *out = buffer.release();
    });
}

void clc_buffer_release(clc_buffer buffer)
{
    if (buffer == nullptr) return;
    if (buffer->bytes != 0)
        buffer->resource->deallocate(buffer->buffer, buffer->bytes);
    delete buffer;
}

size_t clc_buffer_size(clc_buffer buffer)
{
    return buffer == nullptr ? 0 : buffer->bytes;
}

clc_status clc_buffer_write(clc_context ctx, clc_buffer buffer, size_t offset,
                            size_t bytes, const void* src)
{
    return guarded([&] {
        require(ctx != nullptr && buffer != nullptr, "clc: null argument");
        require(offset <= buffer->bytes && bytes <= buffer->bytes - offset,
                "clc: write out of bounds");
        require(bytes == 0 || src != nullptr, "clc: null source");
        if (bytes != 0)
            ctx->ctx.queue().enqueueWriteBuffer(buffer->buffer, CL_TRUE, offset,
                                                bytes, src);
    });
}

clc_status clc_buffer_read(clc_context ctx, clc_buffer buffer, size_t offset,
                           size_t bytes, void* dst)
{
    return guarded([&] {
        require(ctx != nullptr && buffer != nullptr, "clc: null argument");
        require(offset <= buffer->bytes && bytes <= buffer->bytes - offset,
                "clc: read out of bounds");
        require(bytes == 0 || dst != nullptr, "clc: null destination");
        if (bytes != 0)
            ctx->ctx.queue().enqueueReadBuffer(buffer->buffer, CL_TRUE, offset,
                                               bytes, dst);
    });
}

clc_status clc_buffer_copy(clc_context ctx, clc_buffer src, size_t src_offset,
                           clc_buffer dst, size_t dst_offset, size_t bytes)
{
    return guarded([&] {
        require(ctx != nullptr && src != nullptr && dst != nullptr,
                "clc: null argument");
        require(src_offset <= src->bytes && bytes <= src->bytes - src_offset,
                "clc: copy source out of bounds");
        require(dst_offset <= dst->bytes && bytes <= dst->bytes - dst_offset,
                "clc: copy destination out of bounds");
        if (bytes != 0)
            ctx->ctx.queue().enqueueCopyBuffer(src->buffer, dst->buffer,
                                               src_offset, dst_offset, bytes);
    });
}

namespace {
    cl::compute::file_handle open_file(const char* path, const char* flags)
    {
        try
        {
            return cl::compute::file_handle(path, flags);
        } catch (const std::exception& e)
        {
            throw io_error(std::string("clc: cannot open '") + path
                           + "': " + e.what());
        }
    }
}

clc_status clc_file_size(const char* path, size_t* bytes)
{
    return guarded([&] {
        require(path != nullptr && bytes != nullptr, "clc: null argument");
        *bytes = open_file(path, "r").nbytes();
    });
}

clc_status clc_file_read(clc_context ctx, const char* path, size_t file_offset,
                         clc_buffer buffer, size_t buffer_offset, size_t bytes,
                         size_t* done)
{
    return guarded([&] {
        require(ctx != nullptr && path != nullptr && buffer != nullptr
                    && done != nullptr,
                "clc: null argument");
        require(buffer_offset <= buffer->bytes
                    && bytes <= buffer->bytes - buffer_offset,
                "clc: file read out of buffer bounds");
        *done = 0;
        if (bytes == 0) return;
        auto file = open_file(path, "r");
        *done = file.read(ctx->ctx.queue(), buffer->buffer, bytes, file_offset,
                          buffer_offset);
    });
}

clc_status clc_file_write(clc_context ctx, const char* path, int truncate,
                          size_t file_offset, clc_buffer buffer,
                          size_t buffer_offset, size_t bytes, size_t* done)
{
    return guarded([&] {
        require(ctx != nullptr && path != nullptr && buffer != nullptr
                    && done != nullptr,
                "clc: null argument");
        require(buffer_offset <= buffer->bytes
                    && bytes <= buffer->bytes - buffer_offset,
                "clc: file write out of buffer bounds");
        auto file = open_file(path, truncate != 0 ? "w+" : "r+");
        *done = bytes == 0 ? 0
                           : file.write(ctx->ctx.queue(), buffer->buffer, bytes,
                                        file_offset, buffer_offset);
    });
}

clc_status clc_fill(clc_context ctx, clc_buffer out, size_t n, clc_type type,
                    const void* value)
{
    return guarded([&] {
        require(ctx != nullptr && value != nullptr, "clc: null argument");
        require_fp64(ctx, { type });
        require_fits(out, n, type);
        cl::compute::detail::fill(ctx->ctx, out->buffer, n, type_of(type),
                                  value);
    });
}

clc_status clc_sequence(clc_context ctx, clc_buffer out, size_t n,
                        clc_type type, const void* start, const void* step)
{
    return guarded([&] {
        require(ctx != nullptr && start != nullptr && step != nullptr,
                "clc: null argument");
        require_fp64(ctx, { type });
        require_fits(out, n, type);
        cl::compute::detail::sequence(ctx->ctx, out->buffer, n, type_of(type),
                                      start, step);
    });
}

clc_status clc_transform(clc_context ctx, clc_buffer in, clc_type in_type,
                         clc_buffer out, clc_type out_type, size_t n,
                         const char* expression)
{
    return guarded([&] {
        require(ctx != nullptr && expression != nullptr, "clc: null argument");
        require_fp64(ctx, { in_type, out_type });
        require_fits(in, n, in_type);
        require_fits(out, n, out_type);
        cl::compute::detail::transform(ctx->ctx, in->buffer, type_of(in_type),
                                       out->buffer, type_of(out_type), n,
                                       expression);
    });
}

clc_status clc_transform2(clc_context ctx, clc_buffer in1, clc_type type1,
                          clc_buffer in2, clc_type type2, clc_buffer out,
                          clc_type out_type, size_t n, const char* expression)
{
    return guarded([&] {
        require(ctx != nullptr && expression != nullptr, "clc: null argument");
        require_fp64(ctx, { type1, type2, out_type });
        require_fits(in1, n, type1);
        require_fits(in2, n, type2);
        require_fits(out, n, out_type);
        cl::compute::detail::transform2(
            ctx->ctx, in1->buffer, type_of(type1), in2->buffer, type_of(type2),
            out->buffer, type_of(out_type), n, expression);
    });
}

clc_status clc_reduce(clc_context ctx, clc_buffer in, size_t n, clc_type type,
                      const char* op, const char* identity, const void* init,
                      void* result)
{
    return guarded([&] {
        require(ctx != nullptr && op != nullptr && identity != nullptr
                    && init != nullptr && result != nullptr,
                "clc: null argument");
        require_fp64(ctx, { type });
        require_fits(in, n, type);
        cl::compute::detail::reduce(ctx->ctx, in->buffer, n, type_of(type),
                                    { op, identity }, init, result);
    });
}

clc_status clc_scan(clc_context ctx, clc_buffer in, clc_buffer out, size_t n,
                    clc_type type, const char* op, const char* identity,
                    int exclusive)
{
    return guarded([&] {
        require(ctx != nullptr && op != nullptr && identity != nullptr,
                "clc: null argument");
        require_fp64(ctx, { type });
        require_fits(in, n, type);
        require_fits(out, n, type);
        cl::compute::detail::scan(ctx->ctx, in->buffer, out->buffer, n,
                                  type_of(type), { op, identity },
                                  exclusive != 0);
    });
}

clc_status clc_sort(clc_context ctx, clc_buffer keys, clc_type key_type,
                    clc_buffer values, size_t n, int descending)
{
    return guarded([&] {
        require(ctx != nullptr, "clc: null context");
        require(key_type == CLC_TYPE_I32 || key_type == CLC_TYPE_U32
                    || key_type == CLC_TYPE_F32,
                "clc_sort: keys must be I32, U32 or F32");
        require_fits(keys, n, key_type);
        if (values != nullptr) require_fits(values, n, CLC_TYPE_U32);
        cl::compute::detail::radix_sort(
            ctx->ctx, keys->buffer, type_of(key_type),
            values != nullptr ? &values->buffer : nullptr, n, descending != 0);
    });
}

clc_status clc_copy_if(clc_context ctx, clc_buffer in, clc_buffer out, size_t n,
                       clc_type type, const char* predicate, size_t* count)
{
    return guarded([&] {
        require(ctx != nullptr && predicate != nullptr && count != nullptr,
                "clc: null argument");
        require_fp64(ctx, { type });
        require_fits(in, n, type);
        require_fits(out, n, type);
        *count = cl::compute::detail::copy_if(ctx->ctx, in->buffer, out->buffer,
                                              n, type_of(type), predicate);
    });
}

clc_status clc_add_header(clc_context ctx, const char* name,
                          const char* content)
{
    return guarded([&] {
        require(ctx != nullptr && name != nullptr && content != nullptr,
                "clc: null argument");
        ctx->ctx.cache().add_header(name, content);
    });
}

clc_status clc_kernel_create(clc_context ctx, const char* source,
                             const char* name, const char* options,
                             const char* const* define_names,
                             const char* const* define_values, size_t ndefines,
                             clc_kernel* out)
{
    if (out != nullptr) *out = nullptr;
    return guarded([&] {
        require(ctx != nullptr && source != nullptr && name != nullptr
                    && out != nullptr,
                "clc: null argument");
        require(ndefines == 0
                    || (define_names != nullptr && define_values != nullptr),
                "clc: null defines");
        cl::compute::defines defs;
        for (size_t i = 0; i < ndefines; ++i)
        {
            require(define_names[i] != nullptr && define_values[i] != nullptr,
                    "clc: null define");
            defs.emplace_back(define_names[i], define_values[i]);
        }
        cl::Kernel kernel = ctx->ctx.cache().kernel(
            source, name, options != nullptr ? options : "", defs);
        *out = new clc_kernel_s{ kernel };
    });
}

void clc_kernel_release(clc_kernel kernel) { delete kernel; }

clc_status clc_kernel_set_buffer(clc_kernel kernel, unsigned index,
                                 clc_buffer buffer)
{
    return guarded([&] {
        require(kernel != nullptr && buffer != nullptr, "clc: null argument");
        // A zero-byte buffer has no cl_mem; kernels still need a valid
        // argument, so pass NULL (allowed for __global pointers).
        if (buffer->bytes == 0)
        {
            cl_mem none = nullptr;
            kernel->kernel.setArg(index, sizeof(cl_mem), &none);
        }
        else
            kernel->kernel.setArg(index, buffer->buffer);
    });
}

clc_status clc_kernel_set_value(clc_kernel kernel, unsigned index, size_t size,
                                const void* value)
{
    return guarded([&] {
        require(kernel != nullptr && value != nullptr, "clc: null argument");
        kernel->kernel.setArg(index, size, value);
    });
}

clc_status clc_kernel_set_local(clc_kernel kernel, unsigned index, size_t bytes)
{
    return guarded([&] {
        require(kernel != nullptr && bytes != 0, "clc: invalid local size");
        kernel->kernel.setArg(index, cl::Local(bytes));
    });
}

clc_status clc_kernel_work_group_size(clc_context ctx, clc_kernel kernel,
                                      size_t limit, size_t* out)
{
    return guarded([&] {
        require(ctx != nullptr && kernel != nullptr && out != nullptr,
                "clc: null argument");
        *out = ctx->ctx.work_group_size(kernel->kernel, limit);
    });
}

clc_status clc_kernel_launch(clc_context ctx, clc_kernel kernel, size_t global,
                             size_t local)
{
    return guarded([&] {
        require(ctx != nullptr && kernel != nullptr, "clc: null argument");
        require(local == 0 || global % local == 0,
                "clc: global size must be a multiple of local size");
        if (global == 0) return;
        ctx->ctx.queue().enqueueNDRangeKernel(
            kernel->kernel, cl::NullRange, cl::NDRange(global),
            local == 0 ? cl::NullRange : cl::NDRange(local));
    });
}
}
