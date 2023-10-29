#ifndef MAC_H
#define MAC_H

#include <iostream>

// Stores allowed MAC address of current executable

class MAC
{
public:
    static const bool CheckMAC()
    {
        bool FoundMac = false;
        std::vector<std::string> vMacAddresses;
        Config::getdMacAddresses(vMacAddresses);
        //std::cout << vMacAddresses.size() << "\n";
        for(size_t i; i< vMacAddresses.size(); i++)
        {
            //std::cout << vMacAddresses[i] << "\n";
            if(vMacAddresses[i] == "00-23-8b-c9-2b-70")   // ln-laptop LAN1
            {
                std::cout << "Welcome Laurent's Laptop!\n";
                FoundMac |= true;
            }
            else if(vMacAddresses[i] == "20-cf-30-e4-23-45")     // ln-workstation LAN1
            {
                std::cout << "Welcome Laurent's Workstation!\n";
                FoundMac |= true;
            }
            else if(vMacAddresses[i] == "40-2c-f4-e9-34-85")     // CEM ThinkStation LAN1
            {
                std::cout << "Welcome ThinkStation!\n";
                FoundMac |= true;
            }
            else if(vMacAddresses[i] == "00-1f-c6-4a-18-bb")     // Elson LAN1 - Wifi: 00-15-af-82-b1-e6
            {
                std::cout << "Welcome Elson's Laptop!\n";
                FoundMac |= true;
            }
        }
        return FoundMac;
    }
};


#endif // MAC_H
