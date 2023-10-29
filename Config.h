#ifndef CONFIG_H
#define CONFIG_H

#ifdef _WIN32
#include <winsock2.h>
#include <windows.h>
#include <psapi.h>
#include <iphlpapi.h>
#endif

#include <string>
#include <stdlib.h>
#include <iomanip>
#include <ostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>

class Config
{
public:
    /// PRIORITY
    inline static void SetPriorityRealTime()
    {
        SetPriorityClass(GetCurrentProcess(), REALTIME_PRIORITY_CLASS);
    }
    inline static void SetPriorityHigh()
    {
        SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    }
    inline static std::ostream& GetPriority(std::ostream& out)
    {
        DWORD dwPriClass = GetPriorityClass(GetCurrentProcess());
        if(dwPriClass == REALTIME_PRIORITY_CLASS)
        {
            return out << "Running with realtime priority...\n";
        }
        else if(dwPriClass == HIGH_PRIORITY_CLASS)
        {
            return out << "Running with high priority...\n";
        }
        else if(dwPriClass == NORMAL_PRIORITY_CLASS)
        {
            return out << "Running with normal priority...\n";
        }
        else
        {
            return out << "Priority = " <<  dwPriClass << "...\n";
        }
    }
    /// ENV VARS
    inline static const std::string& ComputerInfo()
    {
        static std::string tag;
        static const std::string name = GetVar("COMPUTERNAME");
        if(name.size() != 0)
        {
            tag = tag + "COMPUTERNAME         = " + std::string(name) + "\n";
        }
        else
        {
            tag = tag + "COMPUTERNAME         = ?\n";
        }
        static const std::string cores = GetVar("NUMBER_OF_PROCESSORS");
        if(cores.size() != 0)
        {
            tag = tag + "NUMBER_OF_PROCESSORS = " + std::string(cores) + "\n";
        }
        else
        {
            tag = tag + "NUMBER_OF_PROCESSORS = ?\n";
        }
        static const std::string threads = GetVar("OMP_NUM_THREADS");
        if(threads.size() != 0)
        {
            tag = tag + "OMP_NUM_THREADS      = " + std::string(threads) + "\n";
        }
        else
        {
            tag = tag + "OMP_NUM_THREADS      = ?\n";
        }
        return tag;
    }

    inline static const int GetNumProc()
    {
        return atoi(GetVar("NUMBER_OF_PROCESSORS").data());
    }

    inline static const std::string GetVar(const std::string name)
    {
        char* ptr = getenv(name.c_str());
        std::string ret;
        if(ptr == NULL)
        {
            ret = std::string("");
        }
        else
        {
            ret = std::string(ptr);
        }
        return ret;
    }
    inline static const int GetInt(const std::string name)
    {
        const std::string data = GetVar(name);
        int ret = -1;
        if(data.size() != 0)
        {
            ret = atoi(data.c_str());
        }
        return ret;
    }
    /// COMPUTER IDS
    inline static const int get_computer_name(BYTE* computer_name, DWORD* computer_name_lg)
    {
        HKEY hKey;
        if(RegOpenKeyEx(HKEY_LOCAL_MACHINE,
                        "SYSTEM\\CurrentControlSet\\Control\\ComputerName\\ComputerName",
                        0, KEY_QUERY_VALUE, &hKey) != ERROR_SUCCESS)
        {
            return FALSE;
        }
        if(RegQueryValueEx(hKey, "ComputerName", NULL, NULL,
                           (LPBYTE) computer_name,
                           (LPDWORD) computer_name_lg) != ERROR_SUCCESS)
        {
            RegCloseKey(hKey);
            return FALSE;
        }
        RegCloseKey(hKey);
        return TRUE;
    }
    /// MAC
    inline static const void getdMacAddresses(std::vector<std::string>& vMacAddresses)
    {
        vMacAddresses.clear();
        IP_ADAPTER_INFO AdapterInfo[32];       // Allocate information for up to 32 NICs
        DWORD dwBufLen = sizeof(AdapterInfo);  // Save memory size of buffer
        DWORD dwStatus = GetAdaptersInfo(      // Call GetAdapterInfo
                             AdapterInfo,                 // [out] buffer to receive data
                             &dwBufLen);                  // [in] size of receive data buffer
        //No network card? Other error?
        if(dwStatus != ERROR_SUCCESS)
        {
            return;
        }
        PIP_ADAPTER_INFO pAdapterInfo = AdapterInfo;
        char szBuffer[512];
        while(pAdapterInfo)
        {
            if(pAdapterInfo->Type == MIB_IF_TYPE_ETHERNET)
            {
                snprintf(szBuffer, sizeof(szBuffer), "%.2x-%.2x-%.2x-%.2x-%.2x-%.2x"
                         , pAdapterInfo->Address[0]
                         , pAdapterInfo->Address[1]
                         , pAdapterInfo->Address[2]
                         , pAdapterInfo->Address[3]
                         , pAdapterInfo->Address[4]
                         , pAdapterInfo->Address[5]
                        );
                vMacAddresses.push_back(std::string(szBuffer));
            }
            pAdapterInfo = pAdapterInfo->Next;
        }
    }
};


//class MemStat {
//public:
//    static std::ostream& print(std::ostream& out) {
//        HANDLE hProcess = GetCurrentProcess();
//        if(NULL == hProcess) {
//            return out << "Memory stats: Failed to acquire process handle\n";
//        }
//        PROCESS_MEMORY_COUNTERS pmc;
//        if(!GetProcessMemoryInfo(hProcess, &pmc, sizeof(pmc))) {
//            out << "Memory stats: Failed to acquire process memory information\n";
//        } else {
//            double physiPeak = (double) pmc.PeakWorkingSetSize;
//            double physiPres = (double) pmc.WorkingSetSize;
//            out << "+Memory: " << physiPres/1048576 << " MB |" << physiPeak/1048576 << "|\n";
//        }
//        CloseHandle(hProcess);
//        return out;
//    }
//
//    static std::ostream& AvailableMemory(std::ostream& out) {
//        MEMORYSTATUSEX statex;
//        statex.dwLength = sizeof(statex);
//        if(GlobalMemoryStatusEx(&statex)) {
//            out << "Available memory: " << statex.ullAvailPhys/1048576 << " MB\n";
//        } else {
//            out << "No memory information available\n";
//        }
//        return out;
//    }
//};

#endif

