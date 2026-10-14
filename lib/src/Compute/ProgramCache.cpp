// OpenCL Compute includes
#include <CL/Compute/ProgramCache.hpp>

// STL includes
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <sstream>

namespace cl {
namespace compute {

    namespace {
        std::uint64_t fnv1a(const std::string& data,
                            std::uint64_t hash = 14695981039346656037ull)
        {
            for (unsigned char c : data)
            {
                hash ^= c;
                hash *= 1099511628211ull;
            }
            return hash;
        }

        std::string hex(std::uint64_t value)
        {
            char text[17];
            std::snprintf(text, sizeof(text), "%016llx",
                          static_cast<unsigned long long>(value));
            return text;
        }

        std::string trim_left(const std::string& s)
        {
            std::size_t i = s.find_first_not_of(" \t");
            return i == std::string::npos ? std::string() : s.substr(i);
        }

        // Parses `#include "name"` / `#include <name>`; returns false for
        // any other line.
        bool parse_include(const std::string& line, std::string& name)
        {
            std::string s = trim_left(line);
            if (s.empty() || s[0] != '#') return false;
            s = trim_left(s.substr(1));
            if (s.compare(0, 7, "include") != 0) return false;
            s = trim_left(s.substr(7));
            if (s.empty()) return false;
            const char close = s[0] == '"' ? '"' : (s[0] == '<' ? '>' : 0);
            if (close == 0) return false;
            const std::size_t end = s.find(close, 1);
            if (end == std::string::npos) return false;
            name = s.substr(1, end - 1);
            return true;
        }

        std::string build_logs(const cl::Program& program,
                               const std::vector<cl::Device>& devices)
        {
            std::string logs;
            for (const auto& device : devices)
            {
                std::string log;
                try
                {
                    log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
                } catch (const cl::Error&)
                {
                    log = "<build log unavailable>";
                }
                logs += "\n--- build log for "
                    + device.getInfo<CL_DEVICE_NAME>() + " ---\n" + log;
            }
            return logs;
        }
    }

    program_cache::program_cache(cl::Context context,
                                 std::string disk_cache_dir)
        : context_(std::move(context)),
          devices_(context_.getInfo<CL_CONTEXT_DEVICES>()),
          disk_dir_(std::move(disk_cache_dir))
    {
        if (disk_dir_.empty())
        {
            const char* env = std::getenv("CL_COMPUTE_CACHE_DIR");
            if (env != nullptr) disk_dir_ = env;
        }
        for (const auto& device : devices_)
        {
            cl::Platform platform(device.getInfo<CL_DEVICE_PLATFORM>());
            device_identity_ += platform.getInfo<CL_PLATFORM_NAME>() + '\n'
                + platform.getInfo<CL_PLATFORM_VERSION>() + '\n'
                + device.getInfo<CL_DEVICE_NAME>() + '\n'
                + device.getInfo<CL_DEVICE_VERSION>() + '\n'
                + device.getInfo<CL_DRIVER_VERSION>() + '\n';
        }
        for (const auto& header : builtin_headers())
            headers_[header.first] = header.second;
    }

    void program_cache::add_header(const std::string& name,
                                   const std::string& content)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        headers_[name] = content;
    }

    void program_cache::expand(const std::string& source, std::string& out,
                               std::vector<std::string>& included,
                               int depth) const
    {
        if (depth > 64)
            throw std::runtime_error(
                "program_cache: #include nesting too deep");
        std::istringstream lines(source);
        std::string line, name;
        while (std::getline(lines, line))
        {
            if (parse_include(line, name))
            {
                auto it = headers_.find(name);
                if (it != headers_.end())
                {
                    if (std::find(included.begin(), included.end(), name)
                        == included.end())
                    {
                        included.push_back(name);
                        expand(it->second, out, included, depth + 1);
                    }
                    continue;
                }
            }
            out += line;
            out += '\n';
        }
    }

    std::string program_cache::preprocess(const std::string& source,
                                          const defines& defs) const
    {
        std::string out;
        for (const auto& def : defs)
            out += "#define " + def.first + ' ' + def.second + '\n';
        std::vector<std::string> included;
        std::lock_guard<std::mutex> lock(mutex_);
        expand(source, out, included, 0);
        return out;
    }

    std::string program_cache::key_for(const std::string& expanded,
                                       const std::string& options) const
    {
        std::uint64_t h = fnv1a(expanded);
        h = fnv1a(std::string(1, '\0') + options, h);
        h = fnv1a(std::string(1, '\0') + device_identity_, h);
        return hex(h);
    }

    bool program_cache::load_from_disk(const std::string& key,
                                       const std::string& options,
                                       cl::Program& program)
    {
        if (disk_dir_.empty()) return false;
        cl::Program::Binaries binaries;
        for (std::size_t i = 0; i < devices_.size(); ++i)
        {
            std::ifstream file(disk_dir_ + "/" + key + "-" + std::to_string(i)
                                   + ".clbin",
                               std::ios::binary);
            if (!file) return false;
            binaries.emplace_back(std::istreambuf_iterator<char>(file),
                                  std::istreambuf_iterator<char>());
            if (binaries.back().empty()) return false;
        }
        try
        {
            program = cl::Program(context_, devices_, binaries);
            program.build(devices_, options.c_str());
            return true;
        } catch (const cl::Error&)
        {
            // Stale or foreign binary: rebuild from source.
            return false;
        }
    }

    void program_cache::store_to_disk(const std::string& key,
                                      const cl::Program& program) const
    {
        if (disk_dir_.empty()) return;
        try
        {
            auto binaries = program.getInfo<CL_PROGRAM_BINARIES>();
            for (std::size_t i = 0; i < binaries.size(); ++i)
            {
                if (binaries[i].empty()) return;
                const std::string path =
                    disk_dir_ + "/" + key + "-" + std::to_string(i) + ".clbin";
                const std::string tmp = path + ".tmp";
                {
                    std::ofstream file(tmp, std::ios::binary);
                    if (!file) return;
                    file.write(
                        reinterpret_cast<const char*>(binaries[i].data()),
                        static_cast<std::streamsize>(binaries[i].size()));
                    if (!file) return;
                }
                std::remove(path.c_str());
                std::rename(tmp.c_str(), path.c_str());
            }
        } catch (const cl::Error&)
        {
            // Persisting is best effort.
        }
    }

    cl::Program program_cache::get(const std::string& source,
                                   const std::string& options,
                                   const defines& defs)
    {
        const std::string expanded = preprocess(source, defs);
        const std::string key = key_for(expanded, options);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = programs_.find(key);
            if (it != programs_.end())
            {
                ++stats_.memory_hits;
                return it->second;
            }
        }

        // Build outside the lock so independent programs compile in
        // parallel; a racing duplicate build is harmless.
        cl::Program program;
        bool from_disk = load_from_disk(key, options, program);
        if (!from_disk)
        {
            program = cl::Program(context_, expanded);
            try
            {
                program.build(devices_, options.c_str());
            } catch (const cl::Error& e)
            {
                throw build_error(e.err(),
                                  "program_cache: build failed ("
                                      + std::to_string(e.err()) + ")"
                                      + build_logs(program, devices_));
            }
            store_to_disk(key, program);
        }

        std::lock_guard<std::mutex> lock(mutex_);
        ++(from_disk ? stats_.disk_hits : stats_.builds);
        return programs_.emplace(key, program).first->second;
    }

    cl::Kernel program_cache::kernel(const std::string& source,
                                     const std::string& name,
                                     const std::string& options,
                                     const defines& defs)
    {
        return cl::Kernel(get(source, options, defs), name.c_str());
    }

    program_cache::statistics program_cache::stats() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return stats_;
    }

    void program_cache::clear()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        programs_.clear();
    }

}
}
