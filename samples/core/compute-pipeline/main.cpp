/*
 * Copyright (c) 2026 The Khronos Group Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// OpenCL SDK includes
#include <CL/SDK/Context.hpp>
#include <CL/SDK/Options.hpp>
#include <CL/SDK/CLI.hpp>

// OpenCL Compute includes
#include <CL/Compute/Compute.hpp>

// STL includes
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <tuple>
#include <vector>

// TCLAP includes
#include <tclap/CmdLine.h>

namespace cc = cl::compute;

// Sample-specific option
struct PipelineOptions
{
    size_t length;
};

template <> auto cl::sdk::parse<PipelineOptions>()
{
    return std::make_tuple(std::make_shared<TCLAP::ValueArg<size_t>>(
        "l", "length", "Number of sensor readings", false, 1'000'000,
        "positive integral"));
}
template <>
PipelineOptions cl::sdk::comprehend<PipelineOptions>(
    std::shared_ptr<TCLAP::ValueArg<size_t>> length_arg)
{
    return PipelineOptions{ length_arg->getValue() };
}

namespace {
void expect(bool condition, const char* what)
{
    if (!condition)
        throw std::runtime_error(std::string("Verification FAILED: ") + what);
}
}

int main(int argc, char* argv[])
{
    try
    {
        auto opts =
            cl::sdk::parse_cli<cl::sdk::options::Diagnostic,
                               cl::sdk::options::SingleDevice, PipelineOptions>(
                argc, argv);
        const auto& diag_opts = std::get<0>(opts);
        const auto& dev_opts = std::get<1>(opts);
        const size_t n = std::get<2>(opts).length;

        cl::Context context = cl::sdk::get_context(dev_opts.triplet);
        cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>().at(0);
        cl::CommandQueue queue{ context, device };
        if (!diag_opts.quiet)
            std::cout << "Selected device: " << device.getInfo<CL_DEVICE_NAME>()
                      << "\n"
                      << std::endl;

        // Memory: a pool behind a statistics adaptor (RMM-style stack).
        auto stats = std::make_shared<cc::statistics_adaptor>(
            std::make_shared<cc::buffer_resource>(context));
        auto pool = std::make_shared<cc::pool_resource>(stats);
        cc::compute_context ctx(queue, pool);

        // Synthetic sensor log: increasing timestamps, sensor ids and
        // temperatures in tenths of a degree Celsius.
        std::mt19937 rng(42);
        std::vector<cl_uint> timestamps(n), sensors(n), readings(n);
        cl_uint t = 1'700'000'000u;
        for (size_t i = 0; i < n; ++i)
        {
            timestamps[i] = t += rng() % 20;
            sensors[i] = rng() % 1000;
            readings[i] = 150 + rng() % 300 + (rng() % 1000 == 0 ? 600 : 0);
        }

        // 1. File -> device with parallel I/O (KvikIO-style).
        const char* tmp = std::getenv("TMPDIR");
        if (tmp == nullptr) tmp = std::getenv("TEMP");
        const std::string path =
            std::string(tmp != nullptr ? tmp : ".") + "/compute-pipeline.bin";
        {
            cc::file_handle out(path, "w");
            out.pwrite_host(readings.data(), n * sizeof(cl_uint)).get();
        }
        cc::device_buffer<cl_uint> d_readings(ctx, n);
        {
            cc::file_handle in(path, "r");
            const size_t got =
                in.read(queue, d_readings.get(), n * sizeof(cl_uint));
            expect(got == n * sizeof(cl_uint), "file read size");
        }
        std::remove(path.c_str());

        // 2. Compress the timestamps (nvcomp Cascaded-style).
        auto d_timestamps =
            cc::device_buffer<cl_uint>::from_host(ctx, timestamps);
        cc::bitpack_options bp;
        bp.delta = true;
        auto packed = cc::compress_bitpack(ctx, d_timestamps, bp);
        cc::device_buffer<cl_uint> d_restored(ctx, n);
        cc::decompress_bitpack(ctx, packed, d_restored);
        expect(d_restored.to_host(ctx) == timestamps, "bitpack round trip");

        // 3. Calibration table on the device (cuco::static_map-style):
        //    sensor id -> offset in tenths of a degree.
        std::vector<cl_uint> ids(1000), offsets(1000);
        for (cl_uint i = 0; i < 1000; ++i)
        {
            ids[i] = i;
            offsets[i] = i % 7;
        }
        cc::static_map calibration(ctx, 2048);
        calibration.insert(cc::device_buffer<cl_uint>::from_host(ctx, ids),
                           cc::device_buffer<cl_uint>::from_host(ctx, offsets));
        auto d_sensors = cc::device_buffer<cl_uint>::from_host(ctx, sensors);
        cc::device_buffer<cl_uint> d_offsets(ctx, n);
        calibration.find(d_sensors, d_offsets);

        // 4. Algorithms (Thrust/CUB-style): calibrate in place, then
        //    analyze the calibrated readings.
        cc::transform(ctx, d_readings, d_offsets, d_readings, "x + y");
        for (size_t i = 0; i < n; ++i)
            readings[i] += sensors[i] % 7; // host copy

        const cl_uint hottest =
            cc::reduce(ctx, d_readings, cc::ops::maximum<cl_uint>(), 0u);

        cc::device_buffer<cl_uint> histogram(ctx, 12);
        cc::histogram_even(ctx, d_readings, histogram, 0.0, 1200.0);

        cc::device_buffer<cl_uint> d_alarms(ctx, n);
        const size_t alarms =
            cc::copy_if(ctx, d_readings, d_alarms, "x >= 700u");

        cc::device_buffer<cl_float> d_fahrenheit(ctx, n);
        cc::transform(ctx, d_readings, d_fahrenheit, "x * 0.18f + 32.0f");

        cc::device_buffer<cl_uint> d_order(ctx, n);
        cc::sequence(ctx, d_order);
        cc::sort_by_key(ctx, d_readings, d_order, true);
        const auto top = d_order.to_host(ctx);

        // Host reference.
        const cl_uint ref_hottest =
            *std::max_element(readings.begin(), readings.end());
        const size_t ref_alarms = static_cast<size_t>(
            std::count_if(readings.begin(), readings.end(),
                          [](cl_uint x) { return x >= 700u; }));
        std::vector<cl_uint> ref_hist(12, 0);
        for (cl_uint x : readings)
            if (x < 1200) ++ref_hist[x / 100];
        expect(hottest == ref_hottest, "reduce");
        expect(alarms == ref_alarms, "copy_if");
        expect(histogram.to_host(ctx) == ref_hist, "histogram");
        expect(n == 0 || readings[top[0]] == ref_hottest, "sort_by_key");

        if (!diag_opts.quiet)
        {
            const auto mem = stats->statistics();
            const auto jit = ctx.cache().stats();
            std::cout << "Readings:            " << n << "\n"
                      << "Hottest (0.1 C):     " << hottest << "\n"
                      << "Alarms (>= 70.0 C):  " << alarms << "\n"
                      << "Timestamp ratio:     "
                      << double(packed.uncompressed_bytes())
                    / double(std::max<size_t>(packed.compressed_bytes(), 1))
                      << "x\n"
                      << "Device allocations:  " << mem.total_allocations
                      << " (peak " << mem.peak_bytes / 1024 << " KiB)\n"
                      << "Programs built:      " << jit.builds
                      << " (cache hits " << jit.memory_hits << ")\n"
                      << std::endl;
        }
        std::cout << "Verification passed." << std::endl;
        return 0;
    } catch (cl::BuildError& e)
    {
        std::cerr << "OpenCL build error: " << e.what() << std::endl;
        std::exit(e.err());
    } catch (cl::Error& e)
    {
        std::cerr << "OpenCL runtime error: " << e.what() << std::endl;
        std::exit(e.err());
    } catch (std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
