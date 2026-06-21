#ifndef CONFIG_H
#define CONFIG_H

#include <string>
#include <stdlib.h>
#include <iomanip>
#include <ostream>
#include <stdio.h>
#include <vector>
#include <iostream>

class config
{
public:
    inline static void SetPriorityRealTime() {}
    inline static void SetPriorityHigh() {}
    inline static std::ostream& GetPriority(std::ostream& out)
    {
        return out << "Running with normal priority...\n";
    }
    inline static std::string ComputerInfo()
    {
        std::string tag;
        tag = tag + "COMPUTERNAME         = " + std::string(GetVar("USER") + "\n");
        tag = tag + "OMP_NUM_THREADS      = 1\n";
        return tag;
    }
    inline static const int GetNumProc() { return 1; }
    inline static const std::string GetVar(const std::string name)
    {
        char* ptr = getenv(name.c_str());
        if(ptr == NULL) return std::string("");
        return std::string(ptr);
    }
    inline static const int GetInt(const std::string name)
    {
        const std::string data = GetVar(name);
        if(data.size() != 0) return atoi(data.c_str());
        return -1;
    }
    inline static const void getdMacAddresses(std::vector<std::string>& vMacAddresses)
    {
        vMacAddresses.clear();
    }
};

#endif
