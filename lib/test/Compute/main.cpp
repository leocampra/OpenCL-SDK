// Tests for the OpenCL Compute library. Every device result is compared
// against a host reference. Runs on the default OpenCL device; set
// CL_COMPUTE_TEST_PLATFORM / CL_COMPUTE_TEST_DEVICE to pick another one.

#include <CL/Compute/Compute.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace cc = cl::compute;

namespace {

int failures = 0;
int checks = 0;

#define CHECK(cond)                                                            \
    do                                                                         \
    {                                                                          \
        ++checks;                                                              \
        if (!(cond))                                                           \
        {                                                                      \
            ++failures;                                                        \
            std::cerr << "FAILED: " #cond " at " << __FILE__ << ":"            \
                      << __LINE__ << std::endl;                                \
        }                                                                      \
    } while (0)

template <typename T>
bool equal(const std::vector<T>& a, const std::vector<T>& b)
{
    return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
}

std::vector<cl_uint> random_uints(std::size_t n, cl_uint max, unsigned seed)
{
    std::mt19937 rng(seed);
    std::uniform_int_distribution<cl_uint> dist(0, max);
    std::vector<cl_uint> v(n);
    for (auto& x : v) x = dist(rng);
    return v;
}

const std::size_t sizes[] = { 0, 1, 7, 255, 256, 1000, 4097, 100003, 1 << 20 };

void test_memory_resource(cc::compute_context& ctx)
{
    auto upstream = std::make_shared<cc::buffer_resource>(ctx.context());
    auto stats = std::make_shared<cc::statistics_adaptor>(upstream);
    cc::pool_resource pool(stats);

    CHECK(pool.size_class(1) == 256);
    CHECK(pool.size_class(300) == 512);
    CHECK(pool.size_class((16u << 20) + 1) == (17u << 20));

    cl::Buffer a = pool.allocate(1000);
    cl::Buffer b = pool.allocate(1000);
    CHECK(stats->statistics().current_allocations == 2);
    pool.deallocate(a, 1000);
    CHECK(pool.cached_bytes() == 1024);
    cl::Buffer c = pool.allocate(900); // same size class: served from cache
    CHECK(c() == a());
    CHECK(stats->statistics().total_allocations == 2);
    pool.deallocate(b, 1000);
    pool.deallocate(c, 900);
    pool.release();
    CHECK(pool.cached_bytes() == 0);
    CHECK(stats->statistics().current_allocations == 0);
    CHECK(stats->statistics().peak_bytes == 2048);

    // device_buffer returns its block on destruction and supports moves.
    auto pooled = std::make_shared<cc::pool_resource>(upstream);
    {
        cc::device_buffer<cl_float> x(pooled, 100);
        cc::device_buffer<cl_float> y(std::move(x));
        CHECK(x.empty() && y.size() == 100);
    }
    CHECK(pooled->cached_bytes() == 512);
}

void test_program_cache(cc::compute_context& ctx)
{
    cc::program_cache& cache = ctx.cache();
    cache.add_header("test/twice.h", "#define TWICE(v) ((v) * 2)\n");
    const std::string src =
        "#include \"test/twice.h\"\n"
        "#include <test/twice.h>\n"
        "__kernel void k(__global int* p) { p[0] = TWICE(VALUE); }\n";
    const std::string expanded = cache.preprocess(src, { { "VALUE", "21" } });
    CHECK(expanded.find("#define VALUE 21") == 0);
    CHECK(expanded.find("#include") == std::string::npos);

    const auto before = cache.stats();
    cl::Kernel k1 = cache.kernel(src, "k", "", { { "VALUE", "21" } });
    cl::Kernel k2 = cache.kernel(src, "k", "", { { "VALUE", "21" } });
    cl::Kernel k3 = cache.kernel(src, "k", "", { { "VALUE", "5" } });
    const auto after = cache.stats();
    CHECK(after.builds - before.builds == 2);
    CHECK(after.memory_hits - before.memory_hits == 1);

    cl::Buffer out(ctx.context(), CL_MEM_READ_WRITE, sizeof(cl_int) * 2);
    k1.setArg(0, out);
    ctx.queue().enqueueTask(k1);
    cl_int r1 = 0;
    ctx.queue().enqueueReadBuffer(out, CL_TRUE, 0, sizeof(r1), &r1);
    k3.setArg(0, out);
    ctx.queue().enqueueTask(k3);
    cl_int r3 = 0;
    ctx.queue().enqueueReadBuffer(out, CL_TRUE, 0, sizeof(r3), &r3);
    CHECK(r1 == 42 && r3 == 10);

    bool threw = false;
    try
    {
        cache.get("__kernel void broken( { }");
    } catch (const cc::build_error& e)
    {
        threw = std::string(e.what()).find("build log") != std::string::npos;
    }
    CHECK(threw);

    // Disk cache: a second cache over the same directory loads binaries.
    const char* tmp = std::getenv("TMPDIR");
    const std::string dir = std::string(tmp ? tmp : "/tmp");
    {
        cc::program_cache writer(ctx.context(), dir);
        writer.add_header("test/twice.h", "#define TWICE(v) ((v) * 2)\n");
        writer.get(src, "-DUNIQUE_DISK_TEST", { { "VALUE", "7" } });
        cc::program_cache reader(ctx.context(), dir);
        reader.add_header("test/twice.h", "#define TWICE(v) ((v) * 2)\n");
        cl::Kernel k(
            reader.get(src, "-DUNIQUE_DISK_TEST", { { "VALUE", "7" } }), "k");
        CHECK(reader.stats().disk_hits + reader.stats().builds == 1);
        k.setArg(0, out);
        ctx.queue().enqueueTask(k);
        cl_int r = 0;
        ctx.queue().enqueueReadBuffer(out, CL_TRUE, 0, sizeof(r), &r);
        CHECK(r == 14);
        std::cout << "  disk cache hits: " << reader.stats().disk_hits
                  << std::endl;
    }
}

void test_elementwise(cc::compute_context& ctx)
{
    for (std::size_t n : sizes)
    {
        cc::device_buffer<cl_int> seq(ctx, n);
        cc::sequence(ctx, seq, cl_int(5), cl_int(3));
        std::vector<cl_int> expected(n);
        for (std::size_t i = 0; i < n; ++i) expected[i] = 5 + 3 * cl_int(i);
        CHECK(equal(seq.to_host(ctx), expected));

        cc::device_buffer<cl_float> out(ctx, n);
        cc::transform(ctx, seq, out, "(float)x * 0.5f");
        std::vector<cl_float> half(n);
        for (std::size_t i = 0; i < n; ++i) half[i] = float(expected[i]) * 0.5f;
        CHECK(equal(out.to_host(ctx), half));

        cc::device_buffer<cl_int> sum(ctx, n);
        cc::transform(ctx, seq, seq, sum, "x * y - 1");
        std::vector<cl_int> squares(n);
        for (std::size_t i = 0; i < n; ++i)
            squares[i] = expected[i] * expected[i] - 1;
        CHECK(equal(sum.to_host(ctx), squares));

        cc::fill(ctx, out, 2.5f);
        CHECK(equal(out.to_host(ctx), std::vector<cl_float>(n, 2.5f)));
    }
}

void test_reduce(cc::compute_context& ctx)
{
    for (std::size_t n : sizes)
    {
        auto host = random_uints(n, 1000, unsigned(n));
        auto buf = cc::device_buffer<cl_uint>::from_host(ctx, host);
        const cl_uint sum =
            std::accumulate(host.begin(), host.end(), cl_uint(7));
        CHECK(cc::reduce(ctx, buf, cc::ops::plus(), cl_uint(7)) == sum);
        if (n != 0)
        {
            CHECK(cc::reduce(ctx, buf, cc::ops::maximum<cl_uint>(), cl_uint(0))
                  == *std::max_element(host.begin(), host.end()));
            CHECK(cc::reduce(ctx, buf, cc::ops::minimum<cl_uint>(),
                             cl_uint(0xFFFFFFFFu))
                  == *std::min_element(host.begin(), host.end()));
        }
    }
    std::vector<cl_float> f(10000);
    for (std::size_t i = 0; i < f.size(); ++i) f[i] = float(i % 17) - 8.0f;
    auto fb = cc::device_buffer<cl_float>::from_host(ctx, f);
    CHECK(cc::reduce(ctx, fb, cc::ops::maximum<cl_float>(), -INFINITY) == 8.0f);
    CHECK(cc::reduce(ctx, fb, cc::ops::minimum<cl_float>(), INFINITY) == -8.0f);
}

void test_scan(cc::compute_context& ctx)
{
    for (std::size_t n : sizes)
    {
        auto host = random_uints(n, 100, unsigned(n) + 1);
        auto in = cc::device_buffer<cl_uint>::from_host(ctx, host);
        cc::device_buffer<cl_uint> out(ctx, n);

        std::vector<cl_uint> inc(n), exc(n);
        cl_uint run = 0;
        for (std::size_t i = 0; i < n; ++i)
        {
            exc[i] = run;
            run += host[i];
            inc[i] = run;
        }
        cc::inclusive_scan(ctx, in, out);
        CHECK(equal(out.to_host(ctx), inc));
        cc::exclusive_scan(ctx, in, out);
        CHECK(equal(out.to_host(ctx), exc));
        cc::inclusive_scan(ctx, in, in); // in place
        CHECK(equal(in.to_host(ctx), inc));
    }

    // Non-commutative operator: the right projection "b" scans to the input
    // itself only if operands are always combined in sequence order.
    std::vector<cl_int> v(5000);
    for (std::size_t i = 0; i < v.size(); ++i) v[i] = cl_int(i * 7 % 101);
    auto vb = cc::device_buffer<cl_int>::from_host(ctx, v);
    cc::device_buffer<cl_int> vo(ctx, v.size());
    cc::inclusive_scan(ctx, vb, vo, cc::binary_op{ "b", "0" });
    CHECK(equal(vo.to_host(ctx), v));
    cc::inclusive_scan(ctx, vb, vo, cc::ops::maximum<cl_int>());
    std::vector<cl_int> mx(v.size());
    cl_int m = INT32_MIN;
    for (std::size_t i = 0; i < v.size(); ++i) mx[i] = m = std::max(m, v[i]);
    CHECK(equal(vo.to_host(ctx), mx));
}

void test_sort(cc::compute_context& ctx)
{
    for (std::size_t n : sizes)
    {
        auto keys = random_uints(n, 0xFFFFFFFFu, unsigned(n) + 2);
        std::vector<cl_uint> idx(n);
        std::iota(idx.begin(), idx.end(), 0u);
        auto kb = cc::device_buffer<cl_uint>::from_host(ctx, keys);
        auto vb = cc::device_buffer<cl_uint>::from_host(ctx, idx);
        cc::sort_by_key(ctx, kb, vb);

        std::vector<std::pair<cl_uint, cl_uint>> ref(n);
        for (std::size_t i = 0; i < n; ++i) ref[i] = { keys[i], cl_uint(i) };
        std::stable_sort(ref.begin(), ref.end(),
                         [](const std::pair<cl_uint, cl_uint>& a,
                            const std::pair<cl_uint, cl_uint>& b) {
                             return a.first < b.first;
                         });
        std::vector<cl_uint> rk(n), rv(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            rk[i] = ref[i].first;
            rv[i] = ref[i].second;
        }
        CHECK(equal(kb.to_host(ctx), rk));
        CHECK(equal(vb.to_host(ctx), rv));
    }

    // Stability with many duplicates.
    auto dup = random_uints(50000, 15, 99);
    std::vector<cl_uint> idx(dup.size());
    std::iota(idx.begin(), idx.end(), 0u);
    auto kb = cc::device_buffer<cl_uint>::from_host(ctx, dup);
    auto vb = cc::device_buffer<cl_uint>::from_host(ctx, idx);
    cc::sort_by_key(ctx, kb, vb);
    auto sk = kb.to_host(ctx);
    auto sv = vb.to_host(ctx);
    bool stable = true;
    for (std::size_t i = 1; i < sk.size(); ++i)
        if (sk[i - 1] > sk[i] || (sk[i - 1] == sk[i] && sv[i - 1] > sv[i]))
            stable = false;
    CHECK(stable);

    // Signed and floating-point keys, ascending and descending.
    std::vector<cl_int> ints{ 5, -3, 0, INT32_MIN, 7, -1, INT32_MAX, 2 };
    auto ib = cc::device_buffer<cl_int>::from_host(ctx, ints);
    cc::sort(ctx, ib);
    auto si = ints;
    std::sort(si.begin(), si.end());
    CHECK(equal(ib.to_host(ctx), si));

    std::vector<cl_float> floats(3000);
    std::mt19937 rng(5);
    std::uniform_real_distribution<float> dist(-1e6f, 1e6f);
    for (auto& f : floats) f = dist(rng);
    floats[0] = -0.0f;
    floats[1] = INFINITY;
    floats[2] = -INFINITY;
    auto fb = cc::device_buffer<cl_float>::from_host(ctx, floats);
    cc::sort(ctx, fb, true);
    auto sf = floats;
    std::sort(sf.begin(), sf.end(), [](float a, float b) { return a > b; });
    auto got = fb.to_host(ctx);
    bool ok = got.size() == sf.size();
    for (std::size_t i = 0; ok && i < sf.size(); ++i) ok = got[i] == sf[i];
    CHECK(ok);
}

void test_copy_if_and_histogram(cc::compute_context& ctx)
{
    for (std::size_t n : sizes)
    {
        auto host = random_uints(n, 1000, unsigned(n) + 3);
        auto in = cc::device_buffer<cl_uint>::from_host(ctx, host);
        cc::device_buffer<cl_uint> out(ctx, n);
        const std::size_t count = cc::copy_if(ctx, in, out, "(x % 3u) == 0u");
        std::vector<cl_uint> ref;
        for (cl_uint x : host)
            if (x % 3 == 0) ref.push_back(x);
        CHECK(count == ref.size());
        auto got = out.to_host(ctx);
        got.resize(count);
        CHECK(equal(got, ref));

        cc::device_buffer<cl_uint> hist(ctx, 10);
        cc::histogram_even(ctx, in, hist, 0.0, 1000.0);
        std::vector<cl_uint> href(10, 0);
        for (cl_uint x : host)
            if (x < 1000) ++href[x / 100];
        CHECK(equal(hist.to_host(ctx), href));
    }
}

void test_static_map(cc::compute_context& ctx)
{
    const std::size_t n = 200000;
    std::vector<cl_uint> keys(n), values(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        keys[i] = cl_uint(i * 2654435761u % 0xFFFFFFF0u); // distinct
        values[i] = cl_uint(i);
    }
    cc::static_map map(ctx, n * 2);
    auto kb = cc::device_buffer<cl_uint>::from_host(ctx, keys);
    auto vb = cc::device_buffer<cl_uint>::from_host(ctx, values);
    CHECK(map.insert(kb, vb) == n);
    CHECK(map.size() == n);
    CHECK(map.insert(kb, vb) == 0); // duplicates are ignored

    // Query present keys and absent ones.
    std::vector<cl_uint> queries(keys.begin(), keys.begin() + 1000);
    for (cl_uint i = 0; i < 1000; ++i)
        queries.push_back(0xFFFFFFF1u + (i % 14));
    auto qb = cc::device_buffer<cl_uint>::from_host(ctx, queries);
    cc::device_buffer<cl_uint> found(ctx, queries.size());
    map.find(qb, found);
    auto fv = found.to_host(ctx);
    bool ok = true;
    for (std::size_t i = 0; i < 1000; ++i) ok = ok && fv[i] == values[i];
    for (std::size_t i = 1000; i < queries.size(); ++i)
        ok = ok && fv[i] == map.empty_value();
    CHECK(ok);

    cc::device_buffer<cl_uchar> present(ctx, queries.size());
    map.contains(qb, present);
    auto pv = present.to_host(ctx);
    CHECK(std::count(pv.begin(), pv.end(), cl_uchar(1)) == 1000);

    // Overfilling a small table reports the overflow.
    cc::static_map tiny(ctx, 4);
    bool threw = false;
    try
    {
        tiny.insert(kb, vb, 10);
    } catch (const std::overflow_error&)
    {
        threw = true;
    }
    CHECK(threw && tiny.size() == 4);
    tiny.clear();
    CHECK(tiny.size() == 0);
}

void test_compression(cc::compute_context& ctx)
{
    std::mt19937 rng(11);
    for (std::size_t n : sizes)
    {
        // Slowly increasing timestamps: ideal for delta coding.
        std::vector<cl_uint> ts(n);
        cl_uint t = 1700000000u;
        for (auto& x : ts) x = t += rng() % 50;
        // Small-range values: ideal for plain frame-of-reference.
        auto small = random_uints(n, 300, unsigned(n) + 4);
        for (auto& x : small) x += 5000000u;
        // Random 32-bit data: incompressible.
        auto noise = random_uints(n, 0xFFFFFFFFu, unsigned(n) + 5);

        const std::vector<cl_uint>* inputs[] = { &ts, &small, &noise };
        for (const auto* input : inputs)
            for (bool delta : { false, true })
                for (std::size_t chunk :
                     { std::size_t(256), std::size_t(2048) })
                {
                    auto in =
                        cc::device_buffer<cl_uint>::from_host(ctx, *input);
                    cc::bitpack_options opt;
                    opt.delta = delta;
                    opt.chunk_size = chunk;
                    auto packed = cc::compress_bitpack(ctx, in, opt);
                    cc::device_buffer<cl_uint> out(ctx, n);
                    cc::decompress_bitpack(ctx, packed, out);
                    CHECK(equal(out.to_host(ctx), *input));

                    auto blob = packed.serialize(ctx);
                    auto restored =
                        cc::compressed_buffer::deserialize(ctx, blob);
                    cc::device_buffer<cl_uint> out2(ctx, n);
                    cc::decompress_bitpack(ctx, restored, out2);
                    CHECK(equal(out2.to_host(ctx), *input));
                }
        if (n == (1u << 20))
        {
            auto in = cc::device_buffer<cl_uint>::from_host(ctx, ts);
            cc::bitpack_options opt;
            opt.delta = true;
            auto packed = cc::compress_bitpack(ctx, in, opt);
            const double ratio = double(packed.uncompressed_bytes())
                / double(packed.compressed_bytes());
            std::cout << "  timestamps delta ratio: " << ratio << "x"
                      << std::endl;
            CHECK(ratio > 4.0);
        }
    }

    auto blob = std::vector<unsigned char>(64, 0);
    bool threw = false;
    try
    {
        cc::compressed_buffer::deserialize(ctx, blob);
    } catch (const std::invalid_argument&)
    {
        threw = true;
    }
    CHECK(threw);
}

void test_file_io(cc::compute_context& ctx)
{
    const char* tmp = std::getenv("TMPDIR");
    const std::string path =
        std::string(tmp ? tmp : "/tmp") + "/cl_compute_io_test.bin";
    const std::size_t n = (3u << 20) + 123; // not a multiple of the task size
    auto host = random_uints(n, 0xFFFFFFFFu, 21);
    const std::size_t bytes = n * sizeof(cl_uint);

    for (cc::io_mode mode : { cc::io_mode::mapped, cc::io_mode::bounce })
    {
        cc::io_options opt;
        opt.mode = mode;
        opt.task_size = 1 << 20;
        auto src = cc::device_buffer<cl_uint>::from_host(ctx, host);
        {
            cc::file_handle out(path, "w", opt);
            CHECK(out.write(ctx.queue(), src.get(), bytes) == bytes);
        }
        cc::file_handle in(path, "r", opt);
        CHECK(in.nbytes() == bytes);
        cc::device_buffer<cl_uint> dst(ctx, n);
        auto pending = in.pread(ctx.queue(), dst.get(), bytes);
        CHECK(pending.get() == bytes);
        CHECK(equal(dst.to_host(ctx), host));

        // Offsets on both sides, and a read running past end of file.
        cc::device_buffer<cl_uint> part(ctx, 1000);
        const std::size_t got =
            in.read(ctx.queue(), part.get(), 3600, bytes - 2000, 400);
        CHECK(got == 2000);
        std::vector<cl_uint> tail(500);
        part.copy_to_host(ctx.queue(), tail.data(), 500, 100);
        CHECK(std::equal(tail.begin(), tail.end(), host.end() - 500));

        std::vector<cl_uint> back(n);
        CHECK(in.pread_host(back.data(), bytes).get() == bytes);
        CHECK(equal(back, host));
    }
    std::remove(path.c_str());

    bool threw = false;
    try
    {
        cc::file_handle missing(path + ".missing", "r");
    } catch (const std::system_error&)
    {
        threw = true;
    }
    CHECK(threw);
}

cl::Device pick_device()
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty()) throw std::runtime_error("no OpenCL platform");
    const char* p = std::getenv("CL_COMPUTE_TEST_PLATFORM");
    const char* d = std::getenv("CL_COMPUTE_TEST_DEVICE");
    cl::Platform platform = platforms.at(p ? std::atoi(p) : 0);
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    return devices.at(d ? std::atoi(d) : 0);
}

}

int main()
{
    try
    {
        cl::Device device = pick_device();
        cl::Context context(device);
        cl::CommandQueue queue(context, device);
        cc::compute_context ctx(queue);
        std::cout << "Device: " << device.getInfo<CL_DEVICE_NAME>()
                  << std::endl;

        struct
        {
            const char* name;
            void (*run)(cc::compute_context&);
        } tests[] = {
            { "memory_resource", test_memory_resource },
            { "program_cache", test_program_cache },
            { "elementwise", test_elementwise },
            { "reduce", test_reduce },
            { "scan", test_scan },
            { "sort", test_sort },
            { "copy_if/histogram", test_copy_if_and_histogram },
            { "static_map", test_static_map },
            { "compression", test_compression },
            { "file_io", test_file_io },
        };
        for (const auto& test : tests)
        {
            const int before = failures;
            std::cout << "[" << test.name << "]" << std::endl;
            try
            {
                test.run(ctx);
            } catch (const cl::Error& e)
            {
                ++failures;
                std::cerr << "  OpenCL error " << e.err() << " in " << e.what()
                          << std::endl;
            } catch (const std::exception& e)
            {
                ++failures;
                std::cerr << "  exception: " << e.what() << std::endl;
            }
            std::cout << "  " << (failures == before ? "ok" : "FAILED")
                      << std::endl;
        }
        const auto stats = ctx.cache().stats();
        std::cout << "program cache: " << stats.builds << " builds, "
                  << stats.memory_hits << " hits" << std::endl;
    } catch (const cl::Error& e)
    {
        std::cerr << "OpenCL error " << e.err() << " in " << e.what()
                  << std::endl;
        return EXIT_FAILURE;
    } catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << checks - failures << "/" << checks << " checks passed"
              << std::endl;
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
