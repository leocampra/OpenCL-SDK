/* Tests for the C interface of the OpenCL Compute library. Written in C so
 * that the header is checked as a C header. Uses the default device; set
 * CL_COMPUTE_TEST_DEVICE to an index of the flattened device list to pick
 * another one. */

#include <CL/Compute/compute_c.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
static int checks = 0;

#define CHECK(cond)                                                            \
    do                                                                         \
    {                                                                          \
        ++checks;                                                              \
        if (!(cond))                                                           \
        {                                                                      \
            ++failures;                                                        \
            fprintf(stderr, "FAILED: %s at %s:%d (%s)\n", #cond, __FILE__,     \
                    __LINE__, clc_last_error());                               \
        }                                                                      \
    } while (0)

#define OK(call) CHECK((call) == CLC_SUCCESS)

static const char* square_source =
    "#include <my/square.h>\n"
    "__kernel void square(__global const T* in, __global T* out, uint n)\n"
    "{\n"
    "    uint i = get_global_id(0);\n"
    "    if (i < n) out[i] = SQUARE(in[i]) + OFFSET;\n"
    "}\n";

static const char* wg_sum_source =
    "#include <clc/workgroup.h>\n"
    "CLC_DEFINE_WG_SCAN(int, wg_sum, a + b)\n"
    "__kernel void block_sum(__global const int* in, __global int* out,\n"
    "                        uint n, __local int* scratch)\n"
    "{\n"
    "    uint i = get_global_id(0);\n"
    "    int total;\n"
    "    wg_sum_exclusive(i < n ? in[i] : 0, 0, scratch, &total);\n"
    "    if (get_local_id(0) == 0) out[get_group_id(0)] = total;\n"
    "}\n";

static void test_devices(void)
{
    size_t count = 0;
    clc_device_info info;
    OK(clc_device_count(&count));
    CHECK(count > 0);
    OK(clc_device_get_info(0, &info));
    CHECK(strlen(info.name) > 0);
    CHECK(clc_device_get_info(count, &info) == CLC_ERROR_INVALID_ARGUMENT);
    CHECK(strlen(clc_last_error()) > 0);
    CHECK(clc_device_count(NULL) == CLC_ERROR_INVALID_ARGUMENT);
}

static void test_buffers(clc_context ctx)
{
    int32_t host[100], back[100];
    clc_buffer a = NULL, b = NULL, empty = NULL;
    int i;
    for (i = 0; i < 100; ++i) host[i] = i * 3 - 50;
    OK(clc_buffer_alloc(ctx, sizeof host, &a));
    OK(clc_buffer_alloc(ctx, sizeof host, &b));
    OK(clc_buffer_alloc(ctx, 0, &empty));
    CHECK(clc_buffer_size(a) == sizeof host);
    CHECK(clc_buffer_size(empty) == 0);
    OK(clc_buffer_write(ctx, a, 0, sizeof host, host));
    OK(clc_buffer_copy(ctx, a, 40, b, 0, 60 * sizeof(int32_t)));
    memset(back, 0, sizeof back);
    OK(clc_buffer_read(ctx, b, 0, 60 * sizeof(int32_t), back));
    CHECK(memcmp(back, host + 10, 60 * sizeof(int32_t)) == 0);
    CHECK(clc_buffer_read(ctx, a, 4, sizeof host, back)
          == CLC_ERROR_INVALID_ARGUMENT);
    CHECK(clc_buffer_write(ctx, empty, 0, 0, NULL) == CLC_SUCCESS);
    clc_buffer_release(a);
    clc_buffer_release(b);
    clc_buffer_release(empty);
    clc_buffer_release(NULL);
}

static void test_algorithms(clc_context ctx, int has_fp64)
{
    enum
    {
        N = 10007
    };
    int32_t* host = malloc(N * sizeof(int32_t));
    int32_t* out = malloc(N * sizeof(int32_t));
    float* fout = malloc(N * sizeof(float));
    clc_buffer in = NULL, dst = NULL, fbuf = NULL;
    int64_t expected = 0;
    int32_t init = 0, sum = 0, zero = 0, start = 5, step = 2, mx = 0;
    size_t kept = 0, expected_kept = 0, i;

    for (i = 0; i < N; ++i)
    {
        host[i] = (int32_t)((i * 7919u) % 1000u) - 500;
        expected += host[i];
        if (host[i] > 0) ++expected_kept;
    }
    OK(clc_buffer_alloc(ctx, N * sizeof(int32_t), &in));
    OK(clc_buffer_alloc(ctx, N * sizeof(int32_t), &dst));
    OK(clc_buffer_alloc(ctx, N * sizeof(float), &fbuf));
    OK(clc_buffer_write(ctx, in, 0, N * sizeof(int32_t), host));

    OK(clc_reduce(ctx, in, N, CLC_TYPE_I32, "(a + b)", "0", &init, &sum));
    CHECK(sum == (int32_t)expected);
    init = INT32_MIN;
    OK(clc_reduce(ctx, in, N, CLC_TYPE_I32, "max(a, b)", "INT_MIN", &init,
                  &mx));
    CHECK(mx == 499);

    OK(clc_scan(ctx, in, dst, N, CLC_TYPE_I32, "(a + b)", "0", 0));
    OK(clc_buffer_read(ctx, dst, 0, N * sizeof(int32_t), out));
    CHECK(out[0] == host[0] && out[N - 1] == (int32_t)expected);

    OK(clc_transform(ctx, in, CLC_TYPE_I32, fbuf, CLC_TYPE_F32, N,
                     "(float)x * 0.5f"));
    OK(clc_buffer_read(ctx, fbuf, 0, N * sizeof(float), fout));
    CHECK(fout[3] == (float)host[3] * 0.5f);

    OK(clc_copy_if(ctx, in, dst, N, CLC_TYPE_I32, "x > 0", &kept));
    CHECK(kept == expected_kept);

    OK(clc_sort(ctx, in, CLC_TYPE_I32, NULL, N, 0));
    OK(clc_buffer_read(ctx, in, 0, N * sizeof(int32_t), out));
    for (i = 1; i < N && out[i - 1] <= out[i]; ++i)
        ;
    CHECK(i == N);

    OK(clc_sequence(ctx, dst, N, CLC_TYPE_I32, &start, &step));
    OK(clc_buffer_read(ctx, dst, 0, N * sizeof(int32_t), out));
    CHECK(out[0] == 5 && out[N - 1] == 5 + 2 * (N - 1));
    OK(clc_fill(ctx, dst, N, CLC_TYPE_I32, &zero));
    OK(clc_buffer_read(ctx, dst, 0, sizeof(int32_t), out));
    CHECK(out[0] == 0);

    /* Too many elements for the buffer. */
    CHECK(clc_fill(ctx, dst, N + 1, CLC_TYPE_I32, &zero)
          == CLC_ERROR_INVALID_ARGUMENT);
    /* 64-bit keys are not sortable. */
    CHECK(clc_sort(ctx, dst, CLC_TYPE_I64, NULL, N / 2, 0)
          == CLC_ERROR_INVALID_ARGUMENT);
    /* An invalid expression is a build error with the log in the message. */
    CHECK(clc_transform(ctx, in, CLC_TYPE_I32, dst, CLC_TYPE_I32, N,
                        "x +* nonsense")
          == CLC_ERROR_BUILD);
    CHECK(strstr(clc_last_error(), "build log") != NULL);

    if (has_fp64)
    {
        double dsum = 0.0, dinit = 0.0, dexp = 0.0;
        double* dh = malloc(N * sizeof(double));
        clc_buffer d = NULL;
        for (i = 0; i < N; ++i)
        {
            dh[i] = (double)host[i] * 0.25;
            dexp += dh[i];
        }
        OK(clc_buffer_alloc(ctx, N * sizeof(double), &d));
        OK(clc_buffer_write(ctx, d, 0, N * sizeof(double), dh));
        OK(clc_reduce(ctx, d, N, CLC_TYPE_F64, "(a + b)", "0.0", &dinit,
                      &dsum));
        CHECK(fabs(dsum - dexp) <= 1e-9 * fabs(dexp) + 1e-9);
        clc_buffer_release(d);
        free(dh);
    }
    else
    {
        double dsum = 0.0, dinit = 0.0;
        CHECK(clc_reduce(ctx, in, N / 2, CLC_TYPE_F64, "(a + b)", "0.0", &dinit,
                         &dsum)
              == CLC_ERROR_UNSUPPORTED);
    }

    clc_buffer_release(in);
    clc_buffer_release(dst);
    clc_buffer_release(fbuf);
    free(host);
    free(out);
    free(fout);
}

static void test_kernels(clc_context ctx)
{
    enum
    {
        N = 1000
    };
    const char* names[] = { "T", "OFFSET" };
    const char* values[] = { "int", "7" };
    int32_t host[N], out[N];
    uint32_t n = N;
    clc_kernel k = NULL;
    clc_buffer in = NULL, dst = NULL, empty = NULL;
    size_t wg = 0, groups, i;
    int ok = 1;
    int64_t total = 0;

    for (i = 0; i < N; ++i) host[i] = (int32_t)i - 300;
    OK(clc_add_header(ctx, "my/square.h", "#define SQUARE(v) ((v) * (v))\n"));
    OK(clc_kernel_create(ctx, square_source, "square", NULL, names, values, 2,
                         &k));
    OK(clc_buffer_alloc(ctx, sizeof host, &in));
    OK(clc_buffer_alloc(ctx, sizeof host, &dst));
    OK(clc_buffer_write(ctx, in, 0, sizeof host, host));
    OK(clc_kernel_set_buffer(k, 0, in));
    OK(clc_kernel_set_buffer(k, 1, dst));
    OK(clc_kernel_set_value(k, 2, sizeof n, &n));
    OK(clc_kernel_work_group_size(ctx, k, 64, &wg));
    CHECK(wg >= 1 && wg <= 64 && (wg & (wg - 1)) == 0);
    OK(clc_kernel_launch(ctx, k, ((N + wg - 1) / wg) * wg, wg));
    OK(clc_buffer_read(ctx, dst, 0, sizeof out, out));
    for (i = 0; i < N; ++i)
        if (out[i] != host[i] * host[i] + 7) ok = 0;
    CHECK(ok);
    CHECK(clc_kernel_launch(ctx, k, 100, 64) == CLC_ERROR_INVALID_ARGUMENT);
    /* Zero-sized launch is a no-op, a zero-byte buffer is a valid argument. */
    OK(clc_buffer_alloc(ctx, 0, &empty));
    OK(clc_kernel_set_buffer(k, 0, empty));
    OK(clc_kernel_launch(ctx, k, 0, 0));
    clc_kernel_release(k);

    /* Work-group primitives from the built-in headers, with local memory. */
    k = NULL;
    OK(clc_kernel_create(ctx, wg_sum_source, "block_sum", "", NULL, NULL, 0,
                         &k));
    OK(clc_kernel_work_group_size(ctx, k, 256, &wg));
    groups = (N + wg - 1) / wg;
    OK(clc_kernel_set_buffer(k, 0, in));
    OK(clc_kernel_set_buffer(k, 1, dst));
    OK(clc_kernel_set_value(k, 2, sizeof n, &n));
    OK(clc_kernel_set_local(k, 3, wg * sizeof(int32_t)));
    OK(clc_kernel_launch(ctx, k, groups * wg, wg));
    OK(clc_finish(ctx));
    OK(clc_buffer_read(ctx, dst, 0, groups * sizeof(int32_t), out));
    for (i = 0; i < groups; ++i) total += out[i];
    {
        int64_t expected = 0;
        for (i = 0; i < N; ++i) expected += host[i];
        CHECK(total == expected);
    }
    clc_kernel_release(k);
    k = NULL;
    CHECK(clc_kernel_create(ctx, "__kernel void f( {", "f", NULL, NULL, NULL, 0,
                            &k)
          == CLC_ERROR_BUILD);
    CHECK(k == NULL);
    CHECK(clc_kernel_create(ctx, square_source, "missing", NULL, names, values,
                            2, &k)
          == CLC_ERROR_OPENCL);

    clc_buffer_release(in);
    clc_buffer_release(dst);
    clc_buffer_release(empty);
}

static void test_file_io(clc_context ctx)
{
    enum
    {
        N = 300000
    };
    const char* path = "clc_capi_test.bin";
    int32_t* host = malloc(N * sizeof(int32_t));
    int32_t* back = malloc(N * sizeof(int32_t));
    clc_buffer buf = NULL, half = NULL;
    size_t done = 0, size = 0, i;
    for (i = 0; i < N; ++i) host[i] = (int32_t)(i * 2654435761u);
    OK(clc_buffer_alloc(ctx, N * sizeof(int32_t), &buf));
    OK(clc_buffer_alloc(ctx, N * sizeof(int32_t) / 2, &half));
    OK(clc_buffer_write(ctx, buf, 0, N * sizeof(int32_t), host));
    OK(clc_file_write(ctx, path, 1, 0, buf, 0, N * sizeof(int32_t), &done));
    CHECK(done == N * sizeof(int32_t));
    OK(clc_file_size(path, &size));
    CHECK(size == N * sizeof(int32_t));
    /* Second half of the file into a smaller buffer. */
    OK(clc_file_read(ctx, path, N * sizeof(int32_t) / 2, half, 0,
                     N * sizeof(int32_t) / 2, &done));
    CHECK(done == N * sizeof(int32_t) / 2);
    OK(clc_buffer_read(ctx, half, 0, N * sizeof(int32_t) / 2, back));
    CHECK(memcmp(back, host + N / 2, N * sizeof(int32_t) / 2) == 0);
    /* Reading past the end returns what is there. */
    OK(clc_file_read(ctx, path, N * sizeof(int32_t) - 8, half, 0, 64, &done));
    CHECK(done == 8);
    CHECK(clc_file_read(ctx, "/nonexistent/dir/x.bin", 0, half, 0, 4, &done)
          == CLC_ERROR_IO);
    CHECK(strstr(clc_last_error(), "nonexistent") != NULL);
    CHECK(clc_file_read(ctx, path, 0, half, 0, N * sizeof(int32_t), &done)
          == CLC_ERROR_INVALID_ARGUMENT);
    remove(path);
    clc_buffer_release(buf);
    clc_buffer_release(half);
    free(host);
    free(back);
}

int main(void)
{
    clc_context ctx = NULL, missing = NULL;
    clc_device_info info;
    const char* d = getenv("CL_COMPUTE_TEST_DEVICE");
    size_t index = d != NULL ? (size_t)atoi(d) : CLC_DEFAULT_DEVICE;
    clc_buffer survivor = NULL;
    int32_t v = 42, back = 0;

    test_devices();
    if (clc_context_create(index, &ctx) != CLC_SUCCESS)
    {
        fprintf(stderr, "no context: %s\n", clc_last_error());
        return EXIT_FAILURE;
    }
    CHECK(clc_context_create(1000000, &missing) == CLC_ERROR_NO_DEVICE);
    CHECK(missing == NULL);
    OK(clc_context_get_info(ctx, &info));
    printf("Device: %s (%s), fp64=%d\n", info.name, info.platform,
           info.has_fp64);

    test_buffers(ctx);
    test_algorithms(ctx, info.has_fp64);
    test_kernels(ctx);
    test_file_io(ctx);

    /* Buffers keep the memory resource alive after the context is gone. */
    OK(clc_buffer_alloc(ctx, sizeof v, &survivor));
    OK(clc_buffer_write(ctx, survivor, 0, sizeof v, &v));
    OK(clc_buffer_read(ctx, survivor, 0, sizeof back, &back));
    CHECK(back == 42);
    clc_context_release(ctx);
    clc_buffer_release(survivor);

    printf("%d/%d checks passed\n", checks - failures, checks);
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
