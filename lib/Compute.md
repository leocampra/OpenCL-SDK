# OpenCL Compute library

`OpenCL::Compute` is a C++14 library of reusable GPU building blocks for
OpenCL. It covers the same ground as the CUDA ecosystem libraries listed
below, adapted to OpenCL's model: buffers instead of raw device pointers,
in-order command queues instead of streams, and OpenCL C compiled at run time
instead of templates compiled ahead of time.

| CUDA library | OpenCL Compute counterpart | Header |
|:--|:--|:--|
| Thrust, CUB device algorithms | `fill`, `sequence`, `transform`, `reduce`, `inclusive_scan`, `exclusive_scan`, `sort`, `sort_by_key`, `copy_if`, `histogram_even` | `CL/Compute/Algorithm.hpp` |
| CUB block primitives, libcu++ (`cuda::atomic_ref`) | OpenCL C headers `clc/workgroup.h`, `clc/atomic.h`, `clc/hash.h` | built into every `program_cache` |
| RMM, `thrust::mr`, `cuda::memory_pool` | `memory_resource`, `buffer_resource`, `pool_resource`, `statistics_adaptor`, `device_buffer<T>` | `CL/Compute/MemoryResource.hpp`, `CL/Compute/DeviceBuffer.hpp` |
| jitify (on top of NVRTC) | `program_cache` | `CL/Compute/ProgramCache.hpp` |
| cuCollections (`cuco::static_map`) | `static_map` | `CL/Compute/HashMap.hpp` |
| nvcomp (Cascaded / Bitcomp) | `compress_bitpack`, `decompress_bitpack` | `CL/Compute/Compression.hpp` |
| KvikIO (on top of cuFile/GDS) | `file_handle` | `CL/Compute/FileIO.hpp` |
| C bindings (pylibcudf-style boundary for other languages) | `clc_*` functions | `CL/Compute/compute_c.h` |

The library is built when `OPENCL_SDK_BUILD_COMPUTE_LIBRARY` is `ON` (the
default whenever the utility libraries are built). Link `OpenCL::Compute` and
include `<CL/Compute/Compute.hpp>`. The `compute-pipeline` sample uses every
component; `lib/test/Compute` holds the unit tests (`ctest -R compute`).

## Execution context

```c++
cl::CommandQueue queue{ context, device };   // must be in-order
cl::compute::compute_context ctx(queue);    // pool resource + JIT cache
```

`compute_context` plays the role of the CUDA stream plus the current device
resource: every algorithm enqueues its kernels on `ctx.queue()`, takes
temporary storage from `ctx.resource()` and compiles kernels through
`ctx.cache()`. Pass your own resource or cache to share them between contexts.

## Algorithms

Operators and predicates are OpenCL C expressions, specialized and compiled
on first use and cached afterwards. Binary operators use `a` and `b`;
transforms and predicates use `x` (and `y` for the binary transform).

```c++
namespace cc = cl::compute;
auto v = cc::device_buffer<cl_uint>::from_host(ctx, host);
cl_uint total = cc::reduce(ctx, v);                                  // sum
cl_uint top   = cc::reduce(ctx, v, cc::ops::maximum<cl_uint>(), 0u);
cc::inclusive_scan(ctx, v, v);                                       // in place
cc::device_buffer<cl_float> f(ctx, v.size());
cc::transform(ctx, v, f, "x * 0.5f");
size_t kept = cc::copy_if(ctx, v, out, "(x & 1u) == 0u");
cc::sort_by_key(ctx, keys, values);            // stable LSD radix sort
cc::inclusive_scan(ctx, v, v, cc::binary_op{ "max(a, b)", "0" });  // custom op
```

- Scans support non-commutative associative operators. `reduce`, like
  `thrust::reduce`, combines in an unspecified order, so its operator must
  also be commutative.
- `sort`/`sort_by_key` take 32-bit keys (`cl_uint`, `cl_int`, `cl_float`,
  ascending or descending) and 32-bit values.
- Calls handle at most 2^31-1 elements.
- `double` element types need a device with `cl_khr_fp64`.

## Memory resources

```c++
auto upstream = std::make_shared<cc::buffer_resource>(context);
auto stats    = std::make_shared<cc::statistics_adaptor>(upstream);
auto pool     = std::make_shared<cc::pool_resource>(stats);
cc::compute_context ctx(queue, pool);
```

`pool_resource` rounds requests to size classes and keeps freed blocks for
reuse, releasing its cache and retrying once when the device runs out of
memory. As with RMM's stream-ordered resources, reuse is immediate: that is
safe for work on one in-order queue, but when several queues share a
resource, synchronize them before freeing. `device_buffer<T>` is a move-only
typed allocation that returns its block when destroyed; a pooled buffer may
be larger than requested, so kernels always receive the element count.

## JIT program cache

```c++
cc::program_cache& cache = ctx.cache();
cache.add_header("my/ops.h", "#define SQUARE(v) ((v) * (v))\n");
cl::Kernel k = cache.kernel(source, "my_kernel", "-cl-fast-relaxed-math",
                            { { "T", "float" }, { "TILE", "64" } });
```

- `#include "x"` and `#include <x>` are resolved against registered headers
  before the driver sees the source, so programs never depend on the
  driver's include path. Each header is expanded once per program.
- The macro definitions act like jitify's template arguments: each distinct
  set produces a separately cached specialization.
- Programs are memoized by a hash of the expanded source, the options and
  the platform/device/driver identity. Set `CL_COMPUTE_CACHE_DIR` (or pass a
  directory) to persist device binaries across runs; rejected binaries are
  rebuilt from source.
- Build failures throw `cl::compute::build_error` with every device's log.

Device code can include the built-in headers:

```c
#include <clc/workgroup.h>   /* work-group reduce/scan, like cub::BlockScan */
#include <clc/atomic.h>      /* clc_atomic_add_float, ... */
#include <clc/hash.h>        /* clc_hash_u32 (MurmurHash3 finalizer) */

CLC_DEFINE_WG_SCAN(uint, wg_sum, a + b)
__kernel void k(__global uint* data, __local uint* scratch)
{
    uint total;
    uint prefix = wg_sum_exclusive(data[get_global_id(0)], 0u, scratch, &total);
    ...
}
```

## Hash map

```c++
cc::static_map map(ctx, 2 * n);          // capacity rounded to a power of two
map.insert(keys, values);                // returns the number of new keys
map.find(queries, results);              // empty_value() when absent
map.contains(queries, flags);
```

Open addressing with linear probing and `atomic_cmpxchg` on 32-bit keys.
`empty_key()` (default `0xFFFFFFFF`) marks free slots and cannot be inserted.
As with `cuco::static_map::insert`, the first value for a key wins. Inserting
into a full table throws `std::overflow_error`.

## Compression

```c++
cc::bitpack_options opt;
opt.delta = true;                        // good for sorted / time series data
auto packed = cc::compress_bitpack(ctx, input, opt);
cc::decompress_bitpack(ctx, packed, output);
auto blob = packed.serialize(ctx);       // host bytes with a small header
```

Chunks of `chunk_size` elements are compressed independently, one work-group
per chunk: optional delta + zigzag coding, then frame-of-reference
bit-packing with the smallest width that fits the chunk. The format is
specific to this SDK; it is not nvcomp-compatible and does not implement
LZ4, Snappy or Deflate.

## File I/O

```c++
cc::file_handle f("data.bin", "r");
std::future<size_t> pending = f.pread(queue, buffer.get(), bytes);
size_t n = pending.get();                // < bytes only at end of file
```

Requests are split into `task_size` pieces (default 4 MiB) and run on a
shared thread pool sized by `CL_COMPUTE_NTHREADS` (default 4) or
`set_io_threads()`. Two data paths are available:

- `io_mode::mapped` (default) maps each piece of the buffer and reads the
  file straight into the mapping, which avoids a copy on host-visible memory
  (CPUs, integrated GPUs, `CL_MEM_ALLOC_HOST_PTR`). This is the closest
  OpenCL equivalent of the GPUDirect Storage path; OpenCL has no portable
  storage-to-device DMA.
- `io_mode::bounce` stages through host memory, like KvikIO's and cuFile's
  POSIX compatibility mode.

On POSIX systems transfers use `pread`/`pwrite` and run concurrently; on
Windows they are serialized per handle. Pending transfers keep the file open
even if the handle is closed, but the caller's buffers must outlive them.

## C interface

`<CL/Compute/compute_c.h>` exposes the library to other languages through a
plain C ABI: opaque `clc_context`, `clc_buffer` and `clc_kernel` handles,
status codes instead of exceptions (`clc_last_error()` holds the message,
thread-local), and element types as `clc_type` values.

```c
clc_context ctx;
if (clc_context_create(CLC_DEFAULT_DEVICE, &ctx) != CLC_SUCCESS)
    fprintf(stderr, "%s\n", clc_last_error());   /* CLC_ERROR_NO_DEVICE, ... */
clc_buffer buf;
clc_buffer_alloc(ctx, n * sizeof(int), &buf);   /* from the context's pool */
clc_buffer_write(ctx, buf, 0, n * sizeof(int), host);
int init = 0, sum;
clc_reduce(ctx, buf, n, CLC_TYPE_I32, "(a + b)", "0", &init, &sum);

clc_kernel k;                                   /* JIT through program_cache */
const char* names[] = { "T" }, *values[] = { "int" };
clc_kernel_create(ctx, source, "my_kernel", "", names, values, 1, &k);
clc_kernel_set_buffer(k, 0, buf);
clc_kernel_set_value(k, 1, sizeof n, &n);
clc_kernel_launch(ctx, k, global, local);
```

- `CLC_DEFAULT_DEVICE` picks the first GPU, else the first device of any
  platform; `clc_device_count`/`clc_device_get_info` list the flattened
  device list (name, platform, GPU flag, `fp64` support, memory).
- Typed algorithms on `CLC_TYPE_F64` return `CLC_ERROR_UNSUPPORTED` when the
  device lacks `cl_khr_fp64`; build failures return `CLC_ERROR_BUILD` with
  the build log in the message.
- A buffer keeps the memory resource alive, so it may outlive its context.
  A zero-byte buffer is valid and is passed to kernels as a null pointer.
- `clc_file_size`, `clc_file_read` and `clc_file_write` move data between a
  file and a buffer through `file_handle`; failures return `CLC_ERROR_IO`.
- `clc_context_create_with_cache` persists compiled programs in a directory.
- Platform enumeration and context creation are serialized internally: some
  ICD loaders and drivers crash when several threads initialize at once.
- Contexts are not thread-safe; use one per thread. `lib/test/Compute/capi.c`
  exercises the whole interface from C.
