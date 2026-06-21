#include "post_processor.h"
#include "equation_system.h"
#include "field.h"
#include "radiation.h"
#include "constants.h"
#include "memory.h"
#include "mesh.h"
#include "boundary_condition.h"
#include "option.h"

#include <iomanip>
#include <map>
#include <cmath>
post_processor::post_processor(eq_sys& s, std::ofstream& l) : sys(s), log(l) {}

void post_processor::save_data()
{
    option* opt = sys.opt;
    log << "% Saving data:\n";
    tt.tic();
    if(sys.get_wave_ports_num() > 0)
    {
            std::cout << "Saving S-parameters\n";
        lt.tic();
        int nPorts = (int)sys.get_wave_ports_num();
        std::string sParamFilename = opt->name + ".s" + std::to_string(nPorts) + "p";

        // Read existing Touchstone entries to overwrite duplicate frequencies
        std::map<double, std::vector<std::complex<double>>> freqData;
        {
            std::ifstream inFile(sParamFilename);
            if(inFile.is_open()) {
                std::string line;
                while(std::getline(inFile, line)) {
                    if(line.empty() || line[0] == '!' || line[0] == '#') continue;
                    std::istringstream iss(line);
                    double fGHz;
                    if(iss >> fGHz) {
                        std::vector<std::complex<double>> vals;
                        double mag, ang;
                        while(iss >> mag >> ang) {
                            vals.push_back(std::polar(mag, ang * consts::pi / 180.0));
                        }
                        if(!vals.empty()) freqData[fGHz] = vals;
                    }
                }
            }
        }

        // Compute S-parameters for current frequency
        arma::cx_mat sParams;
        if(opt->nl) {
            sParams = sys.Sp_mat()(arma::span(0,sys.get_wave_ports_num()-1),0);
            sParams(0) -= 1.0;
        } else {
            sParams = sys.Sp_mat()(arma::span(0,sys.get_wave_ports_num()-1),arma::span(0,sys.get_wave_ports_num()-1)) -
                      arma::eye<arma::mat>(sys.get_wave_ports_num(),sys.get_wave_ports_num());
        }
        sParams *= std::sqrt(opt->power);
        double freqGHz = sys.get_freq() / 1e9;

        // Update/add current frequency (column-major: S11, S21, ..., S12, S22, ...)
        std::vector<std::complex<double>> curVals;
        for(int i=0; i < nPorts; i++)
            for(int j=0; j < nPorts; j++)
                curVals.push_back(sParams(j,i));
        freqData[freqGHz] = curVals;

        // Write complete Touchstone file
        std::ofstream sParamFile(sParamFilename);
        sParamFile << "! FES S-parameters\n";
        sParamFile << "! " << opt->name << "\n";
        sParamFile << "# GHz S MA R 50\n";
        for(const auto& entry : freqData) {
            sParamFile << std::setw(10) << std::scientific << std::left << std::setprecision(8) << entry.first;
            for(const auto& val : entry.second) {
                double mag = std::abs(val);
                double ang = std::arg(val) * 180.0 / consts::pi;
                sParamFile << " " << std::setw(14) << std::scientific << std::setprecision(8) << mag
                           << " " << std::setw(14) << std::scientific << std::setprecision(8) << ang;
            }
            sParamFile << "\n";
        }
        sParamFile.close();
        log << "\tS parameters: " << lt.toc() << " s\n";
            std::cout << "|S| =\n" << 20*arma::log10(arma::abs(sParams));
        {
            // Check passivity:  I - S^H S must be positive semidefinite
            // Equivalently: all singular values of S ≤ 1
            arma::cx_mat shS = sParams.t() * sParams;
            arma::cx_mat IminusShS = arma::eye<arma::cx_mat>(sParams.n_rows, sParams.n_rows) - shS;

            arma::vec eigval;
            arma::eig_sym(eigval, IminusShS);

            // max singular value of S = sqrt(max eigenvalue of S^H S)
            arma::vec shsEigval;
            arma::eig_sym(shsEigval, shS);
            double maxSingular = std::sqrt(shsEigval.max());

            std::cout << "\n--- Passivity check (f = " << sys.get_freq() << " Hz) ---\n";
            std::cout << "  I - S^H S min eigenvalue: " << eigval.min() << "\n";
            std::cout << "  Max singular value of S:  " << maxSingular << "\n";
            if(eigval.min() >= -1e-12 && maxSingular <= 1.0 + 1e-12) {
                std::cout << "  PASSIVE ✓\n";
            } else {
                std::cout << "  NOT PASSIVE ✗\n";
            }
        }
        sParams.clear();
    }
    for(size_t i=0; i<sys.port_ampl_vec().size(); i++)
    {
        sys.Sol_mat().col(i) *= sys.port_ampl_vec()[i];
    }
    if(opt->field)
    {
            std::cout << "Saving fields\n";
        lt.tic();
        field(sys.prj, sys.Sol_mat(), sys.get_freq());
        log << "\tfields: " << lt.toc() << " s\n";
    }
    if(opt->rad)
    {
            std::cout << "Saving radiation: ";
        lt.tic();
        double Pref = 0.0;
        double Pinc = 0.0;
        double Pacc = 0.0;
        for(int i=0; i < sys.Sp_mat().n_rows; i++)
        {
            Pref += std::pow(std::abs(sys.Sp_mat()(i,i) - 1.0),2);
            Pinc += 1.0;
        }
        Pacc = Pinc - Pref;
        rad(sys.prj, sys.Sol_mat(), sys.get_freq(), Pacc);
        log << "\tradiation: " << lt.toc() << " s\n";
    }
    log << "+" << tt.toc() << " s\n";
}
