#pragma once

// OpenCL Compute includes
#include "OpenCLCompute_Export.h"

// OpenCL includes
#include <CL/opencl.hpp>

// STL includes
#include <cstddef>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cl {
namespace compute {

    //! Ordered list of macro definitions used to specialize a program, the
    //! OpenCL C counterpart of jitify's template instantiation.
    using defines = std::vector<std::pair<std::string, std::string>>;

    /*! \brief Thrown when a program fails to build; what() includes the
     *  build log of every device.
     */
    class COMPUTE_EXPORT build_error : public std::runtime_error {
    public:
        build_error(cl_int err, const std::string& message)
            : std::runtime_error(message), err_(err)
        {}
        cl_int err() const { return err_; }

    private:
        cl_int err_;
    };

    /*! \brief Runtime compilation cache (jitify analogue).
     *
     *  - Resolves #include "x" / #include <x> against headers registered with
     *    add_header() before handing the source to the driver, so programs
     *    never depend on the driver's include search path. Each header is
     *    expanded at most once per program. The built-in clc/workgroup.h,
     *    clc/atomic.h and clc/hash.h headers are always registered.
     *  - Specializes sources with macro definitions (see defines).
     *  - Memoizes built programs in memory, keyed by a hash of the expanded
     *    source, build options and the identity of every device.
     *  - Optionally persists device binaries on disk, in the directory given
     *    to the constructor or in CL_COMPUTE_CACHE_DIR, and rebuilds from
     *    source whenever a cached binary is rejected.
     *
     *  Thread-safe.
     */
    class COMPUTE_EXPORT program_cache {
    public:
        struct statistics
        {
            std::size_t memory_hits = 0;
            std::size_t disk_hits = 0;
            std::size_t builds = 0;
        };

        /*! \param disk_cache_dir directory for persisted binaries; when empty
         *         the CL_COMPUTE_CACHE_DIR environment variable is used, and
         *         when that is unset binaries are not persisted.
         */
        explicit program_cache(cl::Context context,
                               std::string disk_cache_dir = std::string());

        void add_header(const std::string& name, const std::string& content);

        //! Expands registered headers and prepends the definitions.
        std::string preprocess(const std::string& source,
                               const defines& defs = defines()) const;

        cl::Program get(const std::string& source,
                        const std::string& options = std::string(),
                        const defines& defs = defines());

        //! Returns a fresh cl::Kernel, so concurrent callers never share
        //! kernel arguments.
        cl::Kernel kernel(const std::string& source, const std::string& name,
                          const std::string& options = std::string(),
                          const defines& defs = defines());

        statistics stats() const;
        const std::string& disk_cache_dir() const { return disk_dir_; }

        //! Drops in-memory programs; persisted binaries are kept.
        void clear();

    private:
        std::string key_for(const std::string& expanded,
                            const std::string& options) const;
        bool load_from_disk(const std::string& key, const std::string& options,
                            cl::Program& program);
        void store_to_disk(const std::string& key,
                           const cl::Program& program) const;
        void expand(const std::string& source, std::string& out,
                    std::vector<std::string>& included, int depth) const;

        cl::Context context_;
        std::vector<cl::Device> devices_;
        std::string device_identity_;
        std::string disk_dir_;
        std::map<std::string, std::string> headers_;
        std::map<std::string, cl::Program> programs_;
        statistics stats_;
        mutable std::mutex mutex_;
    };

    //! Name of the OpenCL C type corresponding to a host type.
    template <typename T> struct type_name;
#define CL_COMPUTE_TYPE_NAME(HOST, DEVICE)                                     \
    template <> struct type_name<HOST>                                         \
    {                                                                          \
        static const char* get() { return DEVICE; }                            \
    };
    CL_COMPUTE_TYPE_NAME(cl_char, "char")
    CL_COMPUTE_TYPE_NAME(cl_uchar, "uchar")
    CL_COMPUTE_TYPE_NAME(cl_short, "short")
    CL_COMPUTE_TYPE_NAME(cl_ushort, "ushort")
    CL_COMPUTE_TYPE_NAME(cl_int, "int")
    CL_COMPUTE_TYPE_NAME(cl_uint, "uint")
    CL_COMPUTE_TYPE_NAME(cl_long, "long")
    CL_COMPUTE_TYPE_NAME(cl_ulong, "ulong")
    CL_COMPUTE_TYPE_NAME(cl_float, "float")
    CL_COMPUTE_TYPE_NAME(cl_double, "double")
#undef CL_COMPUTE_TYPE_NAME

    //! Built-in OpenCL C headers registered in every program_cache.
    COMPUTE_EXPORT const std::vector<std::pair<std::string, std::string>>&
    builtin_headers();

}
}
