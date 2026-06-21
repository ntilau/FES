#include <time.h>
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <armadillo>

#include "configuration.h"
#include "memory.h"
#include "option.h"
#include "project.h"
#include "equation_system.h"

int main(int argc, char* argv[])
{
    time_t t;
    option opt;
    opt.set(argc, argv);
    opt.apply_cli();
    if(opt.name.empty()) {
        opt.print_usage(std::cout);
        return EXIT_FAILURE;
    }
    std::ofstream logFile(std::string(opt.name + ".log").data(), std::ios::app);
    try
    {
        std::cout << "----------------------------------------\n";
        std::cout << "                FES\n";
        std::cout << "----------------------------------------\n";
        
        t = time(NULL);
        logFile << "# START: " << asctime(localtime(&t)) << config::ComputerInfo();
        mem_stat::AvailableMemory(logFile);
        mem_stat::AvailableMemory(std::cout);
        arma::wall_clock totTimer;
        totTimer.tic();
        project prj(logFile, opt);
        eq_sys cSys(logFile, &prj);
        mem_stat::print(std::cout);
        std::cout << "++ " << totTimer.toc() << " s\n";
        logFile << "++ " << totTimer.toc() << " s\n";
    }
    catch(const std::exception& err)
    {
        t = time(NULL);
        std::cout << "### ERROR: " << err.what() << " ###\n";
        logFile << "### ERROR: " << err.what() << " ###\n";
        logFile << "# END: " << asctime(localtime(&t)) << "\n";
        logFile.close();
        std::cout << "\a\a\a\a\a";
        return EXIT_FAILURE;
    }
    t = time(NULL);
    logFile << "# END: " << asctime(localtime(&t)) << "\n";
    logFile.close();
    return EXIT_SUCCESS;
}
