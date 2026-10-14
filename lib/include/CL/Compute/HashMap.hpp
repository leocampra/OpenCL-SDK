#pragma once

// OpenCL Compute includes
#include "OpenCLCompute_Export.h"
#include <CL/Compute/Context.hpp>
#include <CL/Compute/DeviceBuffer.hpp>

// STL includes
#include <cstddef>

namespace cl {
namespace compute {

    /*! \brief Fixed-capacity GPU hash map from 32-bit keys to 32-bit values
     *  (cuco::static_map analogue).
     *
     *  Open addressing with linear probing over a power-of-two table; slots
     *  are claimed with atomic_cmpxchg, so bulk operations run fully in
     *  parallel. empty_key marks free slots and may not be inserted;
     *  empty_value is returned by find() for missing keys. Insertion keeps
     *  the first value seen for a key, like cuco::static_map::insert.
     *
     *  Bulk operations are ordered on ctx.queue(): a find() enqueued after an
     *  insert() observes all of its entries, while lookups concurrent with
     *  insertion inside one kernel are not supported.
     */
    class COMPUTE_EXPORT static_map {
    public:
        /*! \param capacity minimum number of slots; rounded up to a power of
         *         two. Keep the load factor below ~0.7 for short probes.
         */
        static_map(compute_context& ctx, std::size_t capacity,
                   cl_uint empty_key = 0xFFFFFFFFu,
                   cl_uint empty_value = 0xFFFFFFFFu);

        //! Inserts n pairs; returns how many keys were new. Throws when the
        //! table is full.
        std::size_t insert(const device_buffer<cl_uint>& keys,
                           const device_buffer<cl_uint>& values, std::size_t n);
        std::size_t insert(const device_buffer<cl_uint>& keys,
                           const device_buffer<cl_uint>& values)
        {
            return insert(keys, values, keys.size());
        }

        //! values[i] = map[keys[i]], or empty_value when absent.
        void find(const device_buffer<cl_uint>& keys,
                  device_buffer<cl_uint>& values) const;

        //! found[i] = 1 when keys[i] is present, else 0.
        void contains(const device_buffer<cl_uint>& keys,
                      device_buffer<cl_uchar>& found) const;

        //! Removes every entry.
        void clear();

        std::size_t size() const { return size_; }
        std::size_t capacity() const { return capacity_; }
        double load_factor() const
        {
            return static_cast<double>(size_) / static_cast<double>(capacity_);
        }
        cl_uint empty_key() const { return empty_key_; }
        cl_uint empty_value() const { return empty_value_; }

        const device_buffer<cl_uint>& slot_keys() const { return keys_; }
        const device_buffer<cl_uint>& slot_values() const { return values_; }

    private:
        compute_context& ctx_;
        std::size_t capacity_;
        cl_uint empty_key_;
        cl_uint empty_value_;
        std::size_t size_ = 0;
        device_buffer<cl_uint> keys_;
        device_buffer<cl_uint> values_;
        device_buffer<cl_uint> counters_;
    };

}
}
