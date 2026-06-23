#include "equation_system.h"
#include <stdexcept>
#include "post_processor.h"
#include "quadrature.h"
#include "constants.h"
#include "memory.h"
#include "field.h"
#include "mesh.h"
#include "radiation.h"

#include "assembler.h"

#include "solver.h"

#include "degree_of_freedom.h"
#include "eigen_solver.h"
#include "element_matrix.h"

#include <set>
#include <map>
#include <fstream>
#include <complex>

namespace {
    constexpr double nonlinear_conv_tol = 1e-5;
    constexpr int nonlinear_max_iter = 50;
}

eq_sys::eq_sys(std::ofstream& logFile, project* pPrj) : prj(pPrj), msh(pPrj->msh), opt(pPrj->opt),
    quadr(new quad(pPrj->opt->p_ord+1)), symm_flag(2), error(1.0), wave_portsNum(0)
{
        std::cout << "Assembly and solution:\n";
    arma::vec freqs = arma::linspace<arma::vec>(opt->l_freq, opt->h_freq, opt->n_freqs);
    if(opt->n_freqs < 2)
    {
        freqs.fill(opt->freq);
    }
    for(size_t kf=0; kf<freqs.n_rows; kf++)
    {
        freq = freqs(kf);
        prj->freq = freq;
        logFile << "--- Frequency = " << freq << " ---\n";
            std::cout << "--- Frequency = " << freq << " ---\n";
            switch(opt->assembly)
            {
            case option::em_e_fd:
                assembler::create(assembler::em_e_fd)->assemble(logFile, *this);
                mem_stat::print(logFile);
                    mem_stat::print(std::cout);
                break;
            case option::em_e_fd_nl:
                if(opt->nl)
                {
                    iter = 1;
                    std::ofstream resFile(std::string(opt->name + "_Res.txt").c_str(), std::ios::app);
                    resFile << "\nproject: " << opt->name << "\n";
                    resFile << "Frequency: " << freq << "\n";
                    resFile << "Harmonics: " << opt->n_harm << "\n";
                    resFile.close();
                }
                do
                {
                    std::ofstream resFile(std::string(opt->name + "_Res.txt").c_str(),std::ios::app);
                    logFile << "### Iter = " << iter << "\n";
                        std::cout << "### Iter = " << iter << "\n";
                    assembler::create(assembler::em_e_fd_nl)->assemble(logFile, *this);
                    mem_stat::print(logFile);
                        mem_stat::print(std::cout);
                    if(iter == 1)
                    {
                        sol_prev.resize(dofnum*opt->n_harm, 1);
                        sp_prev.resize(wave_portsNum, 1);
                        sol_prev.fill(0);
                        sp_prev.fill(0);
                    }
                    solver::create(*opt)->solve(*this, logFile);
                    error = arma::norm(arma::abs(Sp)-arma::abs(sp_prev), 2);
                    sp_prev = Sp;
                    sol_prev *= 1.0-opt->relax;
                    sol_prev += Sol*opt->relax;
                    mem_stat::print(logFile);
                    post_processor(*this, logFile).save_data();
                    logFile << "### Error = " << error << "\n";
                        std::cout << "### Error = " << error << "\n";
                    resFile << iter << " = " << error << "\n";
                    resFile.close();
                    iter++;
                }
                while(error > nonlinear_conv_tol && iter < nonlinear_max_iter);
                break;
            case option::em_e_qs:
                    std::cout << "Only Electrostatic at this stage\n";
                assembler::create(assembler::em_e_qs)->assemble(logFile, *this);
                mem_stat::print(logFile);
                mem_stat::print(std::cout);
                break;
            case option::em_e_fd_dd:
                if(opt->dds)
                {
                    assembler::create(assembler::em_e_fd_dd_schur)->assemble(logFile, *this);
                }
                else
                {
                    assembler::create(assembler::em_e_fd_dd)->assemble(logFile, *this);
                    //gmm::scale(PR,consts::c0/2.0*consts::pi*freq);
                }
                mem_stat::print(logFile);
                mem_stat::print(std::cout);
                break;
            case option::em_ez_fd:
                std::cout << "2D TMz formulation\n";
                assembler::create(assembler::em_ez_fd)->assemble(logFile, *this);
                mem_stat::print(logFile);
                mem_stat::print(std::cout);
                break;
            case option::em_e_tl_eig:
                assembler::create(assembler::em_e_tl_eig)->assemble(logFile, *this);
                continue; // skip solver + post-processing for eigenmode-only run
            default:
                throw std::runtime_error("Formulation not available for this version");
            }
            if(!opt->nl)
            {
                switch(opt->solver)
                {
                case option::direct:
                    if(opt->dd)
                    {
                        if(!opt->dds)
                        {
                            eq_sys::mat_row_type Adiag(dofreal, dofreal);
                            for(size_t i=0; i < dofreal; i++)
                            {
                                Adiag(i,i) = A(i,i);
                                A(i,i) = 0;
                            }
                            A += A.t();
                            A += Adiag;
                            A += PR;
                            symm_flag = 0;
                        }
                    }
                    solver::create(*opt)->solve(*this, logFile);
                    mem_stat::print(logFile);
                    mem_stat::print(std::cout);
                    post_processor(*this, logFile).save_data();
                    break;
                case option::gmres:
                    if(opt->dds)
                    {
                        PR.set_size(dofreal, dofreal);
                        {
                            size_t r0 = doflevel[0], r1 = doflevel[1];
                            size_t c0 = doflevel[1], c1 = doflevel[doflevel.size()-1];
                            for(auto it = A.begin(); it != A.end(); ++it) {
                                size_t r = it.row(), c = it.col();
                                if(r >= r0 && r < r1 && c >= c0 && c < c1) {
                                    PR(r, c) += *it;
                                    PR(c, r) += *it;
                                    A(r, c) = std::complex<double>(0,0);
                                }
                            }
                        }
                        mat_row_type SchurCompl(doflevel[1],doflevel[1]);
                        for(size_t i=1; i<doflevel.size()-2; i++)
                        {
                            size_t curSize =  doflevel[i+1]-doflevel[i];
                            size_t curLev = doflevel[i];
                        }
                    }
                    solver::create(*opt)->solve(*this, logFile);
                    mem_stat::print(logFile);
                    mem_stat::print(std::cout);
                    post_processor(*this, logFile).save_data();
                    break;
                default:
                    throw std::runtime_error("solver not implemented yet");
                }
            }
        }
    }

eq_sys::~eq_sys()
{
    delete quadr;
}
