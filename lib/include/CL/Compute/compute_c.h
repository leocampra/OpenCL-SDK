/*
 * C interface to the OpenCL Compute library.
 *
 * A stable, exception-free boundary for callers in other languages (Rust,
 * Python, ...). Every function that can fail returns a clc_status; on failure
 * clc_last_error() returns a description, valid until the next call on the
 * same thread. No C++ exception ever crosses this boundary.
 *
 * Handles are opaque. A buffer keeps its context's memory resource alive, so
 * buffers and kernels may outlive the clc_context they were created from,
 * but they must not be used with a different context.
 *
 * Contexts are not thread-safe: use one context per thread, or serialize.
 */
#ifndef CL_COMPUTE_COMPUTE_C_H
#define CL_COMPUTE_COMPUTE_C_H

#include "OpenCLCompute_Export.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct clc_context_s* clc_context;
typedef struct clc_buffer_s* clc_buffer;
typedef struct clc_kernel_s* clc_kernel;

typedef enum clc_status
{
    CLC_SUCCESS = 0,
    CLC_ERROR_OPENCL = 1, /* clc_last_cl_error() has the code */
    CLC_ERROR_INVALID_ARGUMENT = 2,
    CLC_ERROR_BUILD = 3, /* message includes the build log */
    CLC_ERROR_NO_DEVICE = 4,
    CLC_ERROR_OUT_OF_MEMORY = 5,
    CLC_ERROR_UNSUPPORTED = 6, /* e.g. double without cl_khr_fp64 */
    CLC_ERROR_INTERNAL = 7,
    CLC_ERROR_IO = 8 /* file could not be opened/read/written */
} clc_status;

/* Element types of the typed algorithms (OpenCL C names in comments). */
typedef enum clc_type
{
    CLC_TYPE_I8 = 0, /* char   */
    CLC_TYPE_U8 = 1, /* uchar  */
    CLC_TYPE_I16 = 2, /* short  */
    CLC_TYPE_U16 = 3, /* ushort */
    CLC_TYPE_I32 = 4, /* int    */
    CLC_TYPE_U32 = 5, /* uint   */
    CLC_TYPE_I64 = 6, /* long   */
    CLC_TYPE_U64 = 7, /* ulong  */
    CLC_TYPE_F32 = 8, /* float  */
    CLC_TYPE_F64 = 9 /* double */
} clc_type;

/* Pass as device index to pick the first GPU, or the first device if there
 * is no GPU. */
#define CLC_DEFAULT_DEVICE ((size_t)-1)

/* ---- errors ------------------------------------------------------------ */

COMPUTE_EXPORT const char* clc_last_error(void);
/* OpenCL error code of the last CLC_ERROR_OPENCL / CLC_ERROR_BUILD, else 0. */
COMPUTE_EXPORT int clc_last_cl_error(void);

/* ---- devices and contexts ---------------------------------------------- */

typedef struct clc_device_info
{
    char name[256];
    char platform[256];
    int is_gpu;
    int has_fp64;
    unsigned long long global_memory;
    size_t max_work_group_size;
} clc_device_info;

/* Devices of every platform, flattened in platform order. Returns
 * CLC_SUCCESS with *count = 0 when no platform is installed. */
COMPUTE_EXPORT clc_status clc_device_count(size_t* count);
COMPUTE_EXPORT clc_status clc_device_get_info(size_t index,
                                              clc_device_info* info);

/* Creates a context, an in-order queue, a pool memory resource and a program
 * cache for one device. index is a position in the flattened device list or
 * CLC_DEFAULT_DEVICE. */
COMPUTE_EXPORT clc_status clc_context_create(size_t index, clc_context* out);
/* Same, persisting compiled programs in cache_dir (an existing directory);
 * NULL or "" falls back to CL_COMPUTE_CACHE_DIR, as clc_context_create. */
COMPUTE_EXPORT clc_status clc_context_create_with_cache(size_t index,
                                                        const char* cache_dir,
                                                        clc_context* out);
COMPUTE_EXPORT void clc_context_release(clc_context ctx);
COMPUTE_EXPORT clc_status clc_context_get_info(clc_context ctx,
                                               clc_device_info* info);
/* Blocks until all work enqueued on the context has completed. */
COMPUTE_EXPORT clc_status clc_finish(clc_context ctx);

/* ---- buffers ----------------------------------------------------------- */

/* bytes may be 0: the handle is valid and holds no device memory. */
COMPUTE_EXPORT clc_status clc_buffer_alloc(clc_context ctx, size_t bytes,
                                           clc_buffer* out);
COMPUTE_EXPORT void clc_buffer_release(clc_buffer buffer);
COMPUTE_EXPORT size_t clc_buffer_size(clc_buffer buffer);

/* Blocking transfers; offset and bytes must stay within the buffer. */
COMPUTE_EXPORT clc_status clc_buffer_write(clc_context ctx, clc_buffer buffer,
                                           size_t offset, size_t bytes,
                                           const void* src);
COMPUTE_EXPORT clc_status clc_buffer_read(clc_context ctx, clc_buffer buffer,
                                          size_t offset, size_t bytes,
                                          void* dst);
COMPUTE_EXPORT clc_status clc_buffer_copy(clc_context ctx, clc_buffer src,
                                          size_t src_offset, clc_buffer dst,
                                          size_t dst_offset, size_t bytes);

/* ---- file I/O (see FileIO.hpp) ---------------------------------------- */
/* Transfers between a file and a buffer through file_handle (mapped mode,
 * parallel pread/pwrite on POSIX). *done receives the bytes transferred,
 * fewer than requested only at end of file. clc_file_write creates the file
 * when truncate != 0 and opens it read/write otherwise. */

COMPUTE_EXPORT clc_status clc_file_size(const char* path, size_t* bytes);
COMPUTE_EXPORT clc_status clc_file_read(clc_context ctx, const char* path,
                                        size_t file_offset, clc_buffer buffer,
                                        size_t buffer_offset, size_t bytes,
                                        size_t* done);
COMPUTE_EXPORT clc_status clc_file_write(clc_context ctx, const char* path,
                                         int truncate, size_t file_offset,
                                         clc_buffer buffer,
                                         size_t buffer_offset, size_t bytes,
                                         size_t* done);

/* ---- typed algorithms (see Algorithm.hpp) ------------------------------ */
/* Operators and predicates are OpenCL C expressions: binary operators use a
 * and b, transforms and predicates use x (and y). value/init/result point to
 * one host element of the given type. */

COMPUTE_EXPORT clc_status clc_fill(clc_context ctx, clc_buffer out, size_t n,
                                   clc_type type, const void* value);
COMPUTE_EXPORT clc_status clc_sequence(clc_context ctx, clc_buffer out,
                                       size_t n, clc_type type,
                                       const void* start, const void* step);
COMPUTE_EXPORT clc_status clc_transform(clc_context ctx, clc_buffer in,
                                        clc_type in_type, clc_buffer out,
                                        clc_type out_type, size_t n,
                                        const char* expression);
COMPUTE_EXPORT clc_status clc_transform2(clc_context ctx, clc_buffer in1,
                                         clc_type type1, clc_buffer in2,
                                         clc_type type2, clc_buffer out,
                                         clc_type out_type, size_t n,
                                         const char* expression);
COMPUTE_EXPORT clc_status clc_reduce(clc_context ctx, clc_buffer in, size_t n,
                                     clc_type type, const char* op,
                                     const char* identity, const void* init,
                                     void* result);
COMPUTE_EXPORT clc_status clc_scan(clc_context ctx, clc_buffer in,
                                   clc_buffer out, size_t n, clc_type type,
                                   const char* op, const char* identity,
                                   int exclusive);
/* Stable radix sort of 32-bit keys (I32, U32, F32); values may be NULL and,
 * if given, hold 32-bit elements. */
COMPUTE_EXPORT clc_status clc_sort(clc_context ctx, clc_buffer keys,
                                   clc_type key_type, clc_buffer values,
                                   size_t n, int descending);
COMPUTE_EXPORT clc_status clc_copy_if(clc_context ctx, clc_buffer in,
                                      clc_buffer out, size_t n, clc_type type,
                                      const char* predicate, size_t* count);

/* ---- user kernels (see ProgramCache.hpp) ------------------------------- */

COMPUTE_EXPORT clc_status clc_add_header(clc_context ctx, const char* name,
                                         const char* content);
/* Compiles (or fetches from the cache) source specialized with ndefines
 * macro definitions and returns the kernel called name. A kernel handle
 * carries its own argument state; create one per concurrent use. */
COMPUTE_EXPORT clc_status clc_kernel_create(clc_context ctx, const char* source,
                                            const char* name,
                                            const char* options,
                                            const char* const* define_names,
                                            const char* const* define_values,
                                            size_t ndefines, clc_kernel* out);
COMPUTE_EXPORT void clc_kernel_release(clc_kernel kernel);
COMPUTE_EXPORT clc_status clc_kernel_set_buffer(clc_kernel kernel,
                                                unsigned index,
                                                clc_buffer buffer);
COMPUTE_EXPORT clc_status clc_kernel_set_value(clc_kernel kernel,
                                               unsigned index, size_t size,
                                               const void* value);
COMPUTE_EXPORT clc_status clc_kernel_set_local(clc_kernel kernel,
                                               unsigned index, size_t bytes);
/* Largest power-of-two work-group size usable with kernel, capped at limit. */
COMPUTE_EXPORT clc_status clc_kernel_work_group_size(clc_context ctx,
                                                     clc_kernel kernel,
                                                     size_t limit, size_t* out);
/* 1-D launch. global must be a multiple of local; local = 0 lets the
 * implementation choose. */
COMPUTE_EXPORT clc_status clc_kernel_launch(clc_context ctx, clc_kernel kernel,
                                            size_t global, size_t local);

#ifdef __cplusplus
}
#endif

#endif
