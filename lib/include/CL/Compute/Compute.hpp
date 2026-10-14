#pragma once

// OpenCL Compute: device-wide algorithms, memory resources, JIT program
// cache, hash map, compression and file I/O for OpenCL. See lib/Compute.md.

#include "OpenCLCompute_Export.h"

#include <CL/Compute/MemoryResource.hpp>
#include <CL/Compute/Context.hpp>
#include <CL/Compute/DeviceBuffer.hpp>
#include <CL/Compute/ProgramCache.hpp>
#include <CL/Compute/Algorithm.hpp>
#include <CL/Compute/HashMap.hpp>
#include <CL/Compute/Compression.hpp>
#include <CL/Compute/FileIO.hpp>
