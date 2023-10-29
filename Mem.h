#ifndef MEM_H
#define MEM_H

#include <winsock2.h>
#include <string>
#include <stdlib.h>
#include <windows.h>
#include <psapi.h>
#include <iomanip>
#include <ostream>

class MemStat {
public:
    static std::ostream& print(std::ostream& out) {
        HANDLE hProcess = GetCurrentProcess();
        if(NULL == hProcess) {
            return out << "Memory stats: Failed to acquire process handle\n";
        }
        PROCESS_MEMORY_COUNTERS pmc;
        if(!GetProcessMemoryInfo(hProcess, &pmc, sizeof(pmc))) {
            out << "Memory stats: Failed to acquire process memory information\n";
        } else {
            double physiPeak = (double) pmc.PeakWorkingSetSize;
            double physiPres = (double) pmc.WorkingSetSize;
            out << "+Memory: " << physiPres/1048576 << " MB |" << physiPeak/1048576 << "|\n";
        }
        CloseHandle(hProcess);
        return out;
    }

    static std::ostream& AvailableMemory(std::ostream& out) {
        MEMORYSTATUSEX statex;
        statex.dwLength = sizeof(statex);
        if(GlobalMemoryStatusEx(&statex)) {
            out << "Available memory: " << statex.ullAvailPhys/1048576 << " MB\n";
        } else {
            out << "No memory information available\n";
        }
        return out;
    }
};

#endif
