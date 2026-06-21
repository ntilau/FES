#ifndef MEM_H
#define MEM_H

#include <string>
#include <iomanip>
#include <ostream>
#include <fstream>
#include <sstream>
#include <cstdint>

// Platform detection for memory reporting
#if defined(__APPLE__)
  #include <mach/mach.h>
  #include <sys/sysctl.h>
#elif defined(__linux__)
  #include <unistd.h>
  #include <sys/resource.h>
#endif

class mem_stat {
public:
    static std::ostream& print(std::ostream& out) {
#if defined(__APPLE__)
        struct mach_task_basic_info info;
        mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
        if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                      (task_info_t)&info, &count) == KERN_SUCCESS) {
            out << "+Memory: " << (double)info.resident_size / (1024.0 * 1024.0) << " MB\n";
        } else {
            out << "Memory stats: Failed\n";
        }
#elif defined(__linux__)
        // Read RSS from /proc/self/status on Linux
        std::ifstream status("/proc/self/status");
        std::string line;
        long rss_kb = 0;
        while(std::getline(status, line)) {
            if(line.compare(0, 6, "VmRSS:") == 0) {
                std::istringstream iss(line.substr(6));
                iss >> rss_kb;
                break;
            }
        }
        out << "+Memory: " << (double)rss_kb / 1024.0 << " MB\n";
#else
        out << "Memory stats: N/A\n";
#endif
        return out;
    }

    static std::ostream& AvailableMemory(std::ostream& out) {
#if defined(__APPLE__)
        int64_t memsize = 0;
        size_t len = sizeof(memsize);
        if (sysctlbyname("hw.memsize", &memsize, &len, NULL, 0) == 0) {
            out << "Available memory: " << (double)memsize / (1024.0 * 1024.0) << " MB\n";
        } else {
            out << "Available memory: N/A\n";
        }
#elif defined(__linux__)
        long pages = sysconf(_SC_PHYS_PAGES);
        long page_size = sysconf(_SC_PAGE_SIZE);
        if(pages > 0 && page_size > 0) {
            out << "Available memory: " << (double)(pages * page_size) / (1024.0 * 1024.0) << " MB\n";
        } else {
            out << "Available memory: N/A\n";
        }
#else
        out << "Available memory: N/A\n";
#endif
        return out;
    }
};

#endif
