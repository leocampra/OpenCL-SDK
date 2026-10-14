#pragma once

// OpenCL Compute includes
#include "OpenCLCompute_Export.h"
#include <CL/Compute/Context.hpp>
#include <CL/Compute/DeviceBuffer.hpp>
#include <CL/Compute/ProgramCache.hpp>

// STL includes
#include <cstddef>
#include <string>

/*! \file Algorithm.hpp
 *  \brief Device-wide parallel algorithms (Thrust / CUB DeviceXxx analogue).
 *
 *  Every algorithm enqueues its kernels on ctx.queue(), takes temporary
 *  storage from ctx.resource() and compiles specialized kernels through
 *  ctx.cache(). Operators and predicates are OpenCL C expressions: binary
 *  operators use the operands a and b, unary expressions and predicates use x.
 */

namespace cl {
namespace compute {

    //! Associative binary operator: an expression of a and b plus its
    //! identity element, both in OpenCL C.
    struct binary_op
    {
        std::string expression;
        std::string identity;
    };

    namespace ops {
        template <typename T> struct limits;
#define CL_COMPUTE_LIMITS(HOST, LOWEST, HIGHEST)                               \
    template <> struct limits<HOST>                                            \
    {                                                                          \
        static const char* lowest() { return LOWEST; }                         \
        static const char* highest() { return HIGHEST; }                       \
    };
        CL_COMPUTE_LIMITS(cl_char, "CHAR_MIN", "CHAR_MAX")
        CL_COMPUTE_LIMITS(cl_uchar, "0", "UCHAR_MAX")
        CL_COMPUTE_LIMITS(cl_short, "SHRT_MIN", "SHRT_MAX")
        CL_COMPUTE_LIMITS(cl_ushort, "0", "USHRT_MAX")
        CL_COMPUTE_LIMITS(cl_int, "INT_MIN", "INT_MAX")
        CL_COMPUTE_LIMITS(cl_uint, "0", "UINT_MAX")
        CL_COMPUTE_LIMITS(cl_long, "LONG_MIN", "LONG_MAX")
        CL_COMPUTE_LIMITS(cl_ulong, "0", "ULONG_MAX")
        CL_COMPUTE_LIMITS(cl_float, "(-INFINITY)", "INFINITY")
        CL_COMPUTE_LIMITS(cl_double, "(-INFINITY)", "INFINITY")
#undef CL_COMPUTE_LIMITS

        inline binary_op plus() { return { "(a + b)", "0" }; }
        inline binary_op multiplies() { return { "(a * b)", "1" }; }
        inline binary_op bit_and() { return { "(a & b)", "(~0)" }; }
        inline binary_op bit_or() { return { "(a | b)", "0" }; }
        inline binary_op bit_xor() { return { "(a ^ b)", "0" }; }
        template <typename T> binary_op minimum()
        {
            return { "((b < a) ? b : a)", limits<T>::highest() };
        }
        template <typename T> binary_op maximum()
        {
            return { "((a < b) ? b : a)", limits<T>::lowest() };
        }
    }

    namespace detail {
        struct type_info
        {
            const char* name;
            std::size_t size;
        };
        template <typename T> type_info info_of()
        {
            return { type_name<T>::get(), sizeof(T) };
        }

        COMPUTE_EXPORT void fill(compute_context& ctx, const cl::Buffer& out,
                                 std::size_t n, type_info t, const void* value);
        COMPUTE_EXPORT void sequence(compute_context& ctx,
                                     const cl::Buffer& out, std::size_t n,
                                     type_info t, const void* start,
                                     const void* step);
        COMPUTE_EXPORT void transform(compute_context& ctx,
                                      const cl::Buffer& in, type_info tin,
                                      const cl::Buffer& out, type_info tout,
                                      std::size_t n,
                                      const std::string& expression);
        COMPUTE_EXPORT void transform2(compute_context& ctx,
                                       const cl::Buffer& in1, type_info t1,
                                       const cl::Buffer& in2, type_info t2,
                                       const cl::Buffer& out, type_info tout,
                                       std::size_t n,
                                       const std::string& expression);
        COMPUTE_EXPORT void reduce(compute_context& ctx, const cl::Buffer& in,
                                   std::size_t n, type_info t,
                                   const binary_op& op, const void* init,
                                   void* result);
        COMPUTE_EXPORT void scan(compute_context& ctx, const cl::Buffer& in,
                                 const cl::Buffer& out, std::size_t n,
                                 type_info t, const binary_op& op,
                                 bool exclusive);
        COMPUTE_EXPORT void radix_sort(compute_context& ctx,
                                       const cl::Buffer& keys, type_info tkey,
                                       const cl::Buffer* values, std::size_t n,
                                       bool descending);
        COMPUTE_EXPORT std::size_t copy_if(compute_context& ctx,
                                           const cl::Buffer& in,
                                           const cl::Buffer& out, std::size_t n,
                                           type_info t,
                                           const std::string& predicate);
        COMPUTE_EXPORT void
        histogram_even(compute_context& ctx, const cl::Buffer& in,
                       std::size_t n, type_info t, const cl::Buffer& histogram,
                       std::size_t num_bins, double lower, double upper);
    }

    //! out[i] = value (thrust::fill).
    template <typename T>
    void fill(compute_context& ctx, device_buffer<T>& out, T value)
    {
        detail::fill(ctx, out.get(), out.size(), detail::info_of<T>(), &value);
    }

    //! out[i] = start + i * step (thrust::sequence).
    template <typename T>
    void sequence(compute_context& ctx, device_buffer<T>& out, T start = T(0),
                  T step = T(1))
    {
        detail::sequence(ctx, out.get(), out.size(), detail::info_of<T>(),
                         &start, &step);
    }

    //! out[i] = expression(in[i]); expression uses x (thrust::transform).
    template <typename In, typename Out>
    void transform(compute_context& ctx, const device_buffer<In>& in,
                   device_buffer<Out>& out, const std::string& expression)
    {
        if (out.size() < in.size())
            throw std::invalid_argument("transform: output too small");
        detail::transform(ctx, in.get(), detail::info_of<In>(), out.get(),
                          detail::info_of<Out>(), in.size(), expression);
    }

    //! out[i] = expression(in1[i], in2[i]); expression uses x and y; out may
    //! alias an input (binary thrust::transform).
    template <typename In1, typename In2, typename Out>
    void transform(compute_context& ctx, const device_buffer<In1>& in1,
                   const device_buffer<In2>& in2, device_buffer<Out>& out,
                   const std::string& expression)
    {
        if (in2.size() < in1.size() || out.size() < in1.size())
            throw std::invalid_argument("transform: size mismatch");
        detail::transform2(ctx, in1.get(), detail::info_of<In1>(), in2.get(),
                           detail::info_of<In2>(), out.get(),
                           detail::info_of<Out>(), in1.size(), expression);
    }

    //! Folds in with op, starting from init (thrust::reduce,
    //! cub::DeviceReduce::Reduce). As in Thrust, the order of combination is
    //! unspecified, so op must be associative and commutative.
    template <typename T>
    T reduce(compute_context& ctx, const device_buffer<T>& in,
             const binary_op& op = ops::plus(), T init = T(0))
    {
        T result;
        detail::reduce(ctx, in.get(), in.size(), detail::info_of<T>(), op,
                       &init, &result);
        return result;
    }

    //! out[i] = in[0] op ... op in[i]; out may alias in
    //! (thrust::inclusive_scan, cub::DeviceScan::InclusiveScan).
    template <typename T>
    void inclusive_scan(compute_context& ctx, const device_buffer<T>& in,
                        device_buffer<T>& out,
                        const binary_op& op = ops::plus())
    {
        if (out.size() < in.size())
            throw std::invalid_argument("inclusive_scan: output too small");
        detail::scan(ctx, in.get(), out.get(), in.size(), detail::info_of<T>(),
                     op, false);
    }

    //! out[i] = identity op in[0] op ... op in[i-1]; out may alias in
    //! (thrust::exclusive_scan, cub::DeviceScan::ExclusiveScan).
    template <typename T>
    void exclusive_scan(compute_context& ctx, const device_buffer<T>& in,
                        device_buffer<T>& out,
                        const binary_op& op = ops::plus())
    {
        if (out.size() < in.size())
            throw std::invalid_argument("exclusive_scan: output too small");
        detail::scan(ctx, in.get(), out.get(), in.size(), detail::info_of<T>(),
                     op, true);
    }

    /*! Stable LSD radix sort of 32-bit keys in place (cub::DeviceRadixSort).
     *  Key may be cl_uint, cl_int or cl_float.
     */
    template <typename Key>
    void sort(compute_context& ctx, device_buffer<Key>& keys,
              bool descending = false)
    {
        static_assert(sizeof(Key) == 4, "sort: keys must be 32-bit");
        detail::radix_sort(ctx, keys.get(), detail::info_of<Key>(), nullptr,
                           keys.size(), descending);
    }

    //! Sorts keys and permutes 32-bit values alongside
    //! (thrust::sort_by_key, cub::DeviceRadixSort::SortPairs).
    template <typename Key, typename Value>
    void sort_by_key(compute_context& ctx, device_buffer<Key>& keys,
                     device_buffer<Value>& values, bool descending = false)
    {
        static_assert(sizeof(Key) == 4, "sort_by_key: keys must be 32-bit");
        static_assert(sizeof(Value) == 4, "sort_by_key: values must be 32-bit");
        if (values.size() < keys.size())
            throw std::invalid_argument("sort_by_key: too few values");
        detail::radix_sort(ctx, keys.get(), detail::info_of<Key>(),
                           &values.get(), keys.size(), descending);
    }

    /*! Copies the elements satisfying predicate (an expression of x) to the
     *  front of out, preserving order, and returns how many were copied
     *  (thrust::copy_if, cub::DeviceSelect::If).
     */
    template <typename T>
    std::size_t copy_if(compute_context& ctx, const device_buffer<T>& in,
                        device_buffer<T>& out, const std::string& predicate)
    {
        if (out.size() < in.size())
            throw std::invalid_argument("copy_if: output too small");
        return detail::copy_if(ctx, in.get(), out.get(), in.size(),
                               detail::info_of<T>(), predicate);
    }

    /*! Counts samples into histogram.size() equal-width bins over
     *  [lower, upper); samples outside are ignored
     *  (cub::DeviceHistogram::HistogramEven).
     */
    template <typename T>
    void histogram_even(compute_context& ctx, const device_buffer<T>& in,
                        device_buffer<cl_uint>& histogram, double lower,
                        double upper)
    {
        detail::histogram_even(ctx, in.get(), in.size(), detail::info_of<T>(),
                               histogram.get(), histogram.size(), lower, upper);
    }

}
}
