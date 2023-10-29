#include "EqSys.h"
#include "Quad.h"
#include "Const.h"
#include "Mem.h"
#include "Field.h"
#include "Mesh.h"
#include "Rad.h"

#include "AssElStat.h"
#include "AssLin.h"
#include "AssNL.h"
#include "AssLinDD.h"
#include "AssLinSchur.h"

#include <smumps_c.h>
#include <dmumps_c.h>
#include <cmumps_c.h>
#include <zmumps_c.h>
//#include "MUMPS.h"
#include "GMRES.h"

#include <iomanip>
#include <armadillo>

#include <cfloat> // for DBL_MIN


EqSys::EqSys(std::ofstream& logFile, Project* pPrj) : prj(pPrj), msh(pPrj->msh), opt(pPrj->opt),
    quad(new Quad(pPrj->opt->pOrd+1)), SymmFlag(2), error(1.0), WavePortsNum(0)
{
    if(opt->verbose)
    {
        std::cout << "Assembly and solution:\n";
    }
    arma::vec freqs = arma::linspace<arma::vec>(opt->lFreq, opt->hFreq, opt->nFreqs);
    if(opt->nFreqs < 2)
    {
        freqs.fill(opt->freq);
    }
    for(size_t kf=0; kf<freqs.n_rows; kf++)
    {
        freq = freqs(kf);
        prj->freq = freq;
        logFile << "--- Frequency = " << freq << " ---\n";
        if(opt->verbose)
        {
            std::cout << "--- Frequency = " << freq << " ---\n";
        }
        if(opt->LIMITED)
        {
            switch(opt->assembly)
            {
            case Option::LIN:
                AssLin(logFile, this);
                MemStat::print(logFile);
                if(opt->verbose)
                {
                    MemStat::print(std::cout);
                }
                break;
            default:
                throw std::string("Formulation not available for this version");
            }
            switch(opt->solver)
            {
            case Option::DIRECT:
                if(opt->dbl)
                {
                    SolveDoubleComplex(logFile);
                }
                else
                {
                    SolveSingleComplex(logFile);
                }
                MemStat::print(logFile);
                if(opt->verbose)
                {
                    MemStat::print(std::cout);
                }
                SaveData(logFile);
                break;
            default:
                throw std::string("Solver not available for this version");
            }
        }
        else
        {
            switch(opt->assembly)
            {
            case Option::LIN:
                AssLin(logFile, this);
                MemStat::print(logFile);
                if(opt->verbose)
                {
                    MemStat::print(std::cout);
                }
                break;
            case Option::NL:
                if(opt->nl)
                {
                    iter = 1;
                    std::ofstream resFile(std::string(opt->name + "_Res.txt").c_str(), std::ios::app);
                    resFile << "\nProject: " << opt->name << "\n";
                    resFile << "Frequency: " << freq << "\n";
                    resFile << "Harmonics: " << opt->nHarm << "\n";
                    resFile.close();
                }
                do
                {
                    std::ofstream resFile(std::string(opt->name + "_Res.txt").c_str(),std::ios::app);
                    logFile << "### Iter = " << iter << "\n";
                    if(opt->verbose)
                    {
                        std::cout << "### Iter = " << iter << "\n";
                    }
                    AssNL(logFile, this);
                    MemStat::print(logFile);
                    if(opt->verbose)
                    {
                        MemStat::print(std::cout);
                    }
                    if(iter == 1)
                    {
                        Solprev.resize(DoFnum*opt->nHarm, 1);
                        Spprev.resize(WavePortsNum, 1);
                        Solprev.fill(0);
                        Spprev.fill(0);
                    }
                    if(opt->dbl)
                    {
                        SolveDoubleComplex(logFile);
                    }
                    else
                    {
                        SolveSingleComplex(logFile);
                    }
                    error = arma::norm(arma::abs(Sp)-arma::abs(Spprev), 2);
                    //arma::mat SpErr = arma::max(arma::max(arma::abs(Sp)-arma::abs(Spprev),0),1);
                    //error = std::abs(SpErr(0,0));
                    //error = arma::norm(arma::abs(Solprev)-arma::abs(Sol),2) / arma::norm(Sol,2); // field based
                    Spprev = Sp;
                    Solprev *= 1.0-opt->relax;
                    Solprev += Sol*opt->relax;
                    MemStat::print(logFile);
                    SaveData(logFile);
                    logFile << "### Error = " << error << "\n";
                    if(opt->verbose)
                    {
                        std::cout << "### Error = " << error << "\n";
                    }
                    resFile << iter << " = " << error << "\n";
                    resFile.close();
                    iter++;
                }
                while(error > 1e-5 && iter < 50);
                //SaveSystem(logFile);
                break;
            case Option::STAT:
                if(opt->verbose)
                {
                    std::cout << "Only Electrostatic at this stage\n";
                }
                AssElStat(logFile, this);
                MemStat::print(logFile);
                if(opt->verbose)
                {
                    MemStat::print(std::cout);
                }
                break;
            case Option::DD:
                if(opt->dds)
                {
                    AssLinSchur(logFile, this);
                }
                else
                {
                    AssLinDD(logFile, this);
                    //gmm::scale(PR,Const::c0/2.0*Const::pi*freq);
                }
                MemStat::print(logFile);
                MemStat::print(std::cout);
                break;
            default:
                throw std::string("Formulation not available for this version");
            }
            if(!opt->nl)
            {
                switch(opt->solver)
                {
                case Option::DIRECT:
                    if(opt->dd)
                    {
                        if(!opt->dds)
                        {
                            EqSys::MatRowType Atmp(DoFreal, DoFreal);
                            #pragma omp parallel for
                            for(size_t i=0; i < DoFreal; i++)
                            {
                                Atmp(i,i) = A(i,i);
                                A(i,i) = 0;
                            }
                            gmm::add(gmm::transposed(A), Atmp);
                            gmm::add(A, Atmp);
                            std::swap(A, Atmp);
                            Atmp.clear_mat();
                            gmm::add(PR,A);
                            SymmFlag = 0;
                        }
                    }
                    if(opt->dbl)
                    {
                        SolveDoubleComplex(logFile);
                    }
                    else
                    {
                        SolveSingleComplex(logFile);
                    }
                    MemStat::print(logFile);
                    if(opt->verbose)
                    {
                        MemStat::print(std::cout);
                    }
                    SaveData(logFile);
                    break;
                case Option::MATLAB:
                    if(opt->dds)
                    {
//                            EqSys::MatRowType Atmp(DoFreal, DoFreal);
//                            #pragma omp parallel for
//                            for(size_t i=0; i < DoFreal; i++) {
//                                Atmp(i,i) = A(i,i);
//                                A(i,i) *= 0.0;
//                            }
//                            gmm::add(gmm::transposed(A), Atmp);
//                            gmm::add(A, Atmp);
//                            std::swap(A, Atmp);
                        gmm::resize(PR,DoFreal,DoFreal);
                        gmm::add(gmm::sub_matrix(A, gmm::sub_interval(DoFlevel[1], DoFlevel[DoFlevel.size()-1]-DoFlevel[1]),
                                                 gmm::sub_interval(DoFlevel[0], DoFlevel[1])),
                                 gmm::sub_matrix(PR, gmm::sub_interval(DoFlevel[1], DoFlevel[DoFlevel.size()-1]-DoFlevel[1]),
                                                 gmm::sub_interval(DoFlevel[0], DoFlevel[1])));
                    }
                    SaveSystem(logFile);
                    break;
                case Option::GMRES:
                    if(opt->dds)
                    {
//                            /// diagonal scaling
//                            gmm::resize(Adiag, DoFreal, DoFreal);
//                            #pragma omp parallel for
//                            for(size_t i=0; i < DoFreal; i++) {
//                                Adiag(i,i) = std::complex<double>(1.0,0.0)/A(i,i);
//                            }
//                            gmm::mult(gmm::conjugated(Adiag), A, A);
                        gmm::resize(PR,DoFreal,DoFreal);
                        // upper band PR
                        gmm::add(gmm::sub_matrix(A, gmm::sub_interval(DoFlevel[0], DoFlevel[1]),                                                     gmm::sub_interval(DoFlevel[1], DoFlevel[DoFlevel.size()-1]-DoFlevel[1])),
                                 gmm::sub_matrix(PR, gmm::sub_interval(DoFlevel[0], DoFlevel[1]),                                                     gmm::sub_interval(DoFlevel[1], DoFlevel[DoFlevel.size()-1]-DoFlevel[1])));
                        // lower band PR
                        gmm::add(gmm::transposed(gmm::sub_matrix(A, gmm::sub_interval(DoFlevel[0], DoFlevel[1]),                                                     gmm::sub_interval(DoFlevel[1], DoFlevel[DoFlevel.size()-1]-DoFlevel[1]))),
                                 gmm::sub_matrix(PR, gmm::sub_interval(DoFlevel[1], DoFlevel[DoFlevel.size()-1]-DoFlevel[1]),
                                                 gmm::sub_interval(DoFlevel[0], DoFlevel[1])));
                        // remove upper band on A
                        gmm::clear(gmm::sub_matrix(A, gmm::sub_interval(DoFlevel[0], DoFlevel[1]),                                                     gmm::sub_interval(DoFlevel[1], DoFlevel[DoFlevel.size()-1]-DoFlevel[1])));
                        // assemble schur complement in PR
                        MatRowType SchurCompl(DoFlevel[1],DoFlevel[1]);
                        for(size_t i=1; i<DoFlevel.size()-2; i++)
                        {
                            size_t curSize =  DoFlevel[i+1]-DoFlevel[i];
                            size_t curLev = DoFlevel[i];
                        }
                    }
                    SolveGmRes(logFile);
                    MemStat::print(logFile);
                    if(opt->verbose)
                    {
                        MemStat::print(std::cout);
                    }
                    SaveData(logFile);
                    break;
                default:
                    throw std::string("Solver not implemented yet");
                }
            }
        }
    }
}

void EqSys::SolveSingleComplex(std::ofstream& logFile)
{
    logFile << "% Single precision Complex Solver:\n";
    if(opt->verbose)
    {
        std::cout << "Single precision Complex Solver";
    }
    tt.tic();
    if(opt->verbose)
    {
        switch(SymmFlag)
        {
        case 0:
            logFile << " / Non-symmetric\n";
            std::cout << " / Non-symmetric\n";
            break;
        case 1:
            logFile << " / Symmetric\n";
            std::cout << " / Symmetric\n";
            break;
        case 2:
            logFile << " / General symmetric\n";
            std::cout << " / General symmetric\n";
            break;
        }
    }
    CMUMPS_STRUC_C* idz = new CMUMPS_STRUC_C;
    idz->job=-1; // init
    idz->par=1;
    idz->sym=(MUMPS_INT) SymmFlag;
    idz->comm_fortran=-987654; // use_comm_world
    cmumps_c(idz);
    if(opt->verbose)
    {
        MemStat::AvailableMemory(std::cout);
    }
    idz->n = (MUMPS_INT) DoFreal;
    idz->nz = (MUMPS_INT) gmm::nnz(A);
    idz->irn = new MUMPS_INT [idz->nz];
    idz->jcn = new MUMPS_INT [idz->nz];
    idz->a = new CMUMPS_COMPLEX [idz->nz];
    typename gmm::linalg_traits<MatRowType>::const_row_iterator it = mat_row_const_begin(A);
    typename gmm::linalg_traits<MatRowType>::const_row_iterator ite = mat_row_const_end(A);
    size_t i = 0,  idx = 0;
    for(; it != ite; ++it)
    {
        typename gmm::linalg_traits<MatRowType>::const_sub_row_type row = gmm::linalg_traits<MatRowType>::row(it);
        typename gmm::linalg_traits<VecType>::const_iterator itr = vect_const_begin(row);
        typename gmm::linalg_traits<VecType>::const_iterator itre = vect_const_end(row);
        for(; itr != itre; ++itr)
        {
            idz->a[idx].r = (CMUMPS_REAL) itr->real();
            idz->a[idx].i = (CMUMPS_REAL) itr->imag();
            idz->irn[idx] = (MUMPS_INT) i+1;
            idz->jcn[idx++] = (MUMPS_INT) itr.index()+1;
        }
        i++;
    }
    idz->icntl[0]=-1; //-1 6
    idz->icntl[1]=-1; //-1 6
    idz->icntl[2]=-1; //-1 1
    idz->icntl[3]= 0; //0 6
    idz->icntl[19]=1; // sparse right hand side
    idz->nrhs = (MUMPS_INT) gmm::mat_ncols(B);
    idz->lrhs = (MUMPS_INT) gmm::mat_nrows(B);
    idz->rhs = new CMUMPS_COMPLEX [idz->lrhs*idz->nrhs];
    idz->nz_rhs = (MUMPS_INT) gmm::nnz(B) ;
    idz->rhs_sparse = new CMUMPS_COMPLEX [idz->nz_rhs];
    idz->irhs_sparse = new MUMPS_INT [idz->nz_rhs];
    idz->irhs_ptr = new MUMPS_INT [idz->nrhs+1];
    it = mat_col_const_begin(B);
    ite = mat_col_const_end(B);
    i = 0,  idx = 0;
    for(; it != ite; ++it)
    {
        typename gmm::linalg_traits<MatRowType>::const_sub_row_type row = gmm::linalg_traits<MatRowType>::row(it);
        typename gmm::linalg_traits<VecType>::const_iterator itr = vect_const_begin(row);
        typename gmm::linalg_traits<VecType>::const_iterator itre = vect_const_end(row);
        idz->irhs_ptr[i++] = (MUMPS_INT) idx+1;
        for(; itr != itre; ++itr)
        {
            idz->irhs_sparse[idx] = (MUMPS_INT) itr.index()+1;
            idz->rhs_sparse[idx].r = (CMUMPS_REAL) itr->real();
            idz->rhs_sparse[idx++].i = (CMUMPS_REAL) itr->imag();
        }
    }
    idz->irhs_ptr[idz->nrhs] = (MUMPS_INT)(idz->nz_rhs+1);
    // freeing memory
    {
        MatRowType Atmp(idz->n, idz->n);
        MatColType Btmp(idz->n, idz->nrhs);
        std::swap(A, Atmp);
        std::swap(B, Btmp);
    }
    if(opt->verbose)
    {
        MemStat::print(std::cout);
    }
    idz->job=6; // factorization and solution
    if(opt->verbose)
    {
        std::cout << "Factor and Solve ";
    }
    lt.tic();
    cmumps_c(idz);
    if(opt->verbose)
    {
        std::cout << "+ " << idz->info[14] << " MB ";
    }
    logFile << "\tCommit memory: " << idz->info[14] << " MB\n";
    logFile << "\tFactor and Solve: " <<  lt.toc() << " s\n";
    if(opt->nl)
    {
        Sol.resize(DoFnum*opt->nHarm, 1);
        Sp.resize(WavePortsNum, 1);
    }
    else
    {
        Sp.resize(idz->nrhs, idz->nrhs);
        Sol.resize(DoFnum, idz->nrhs);
    }
    Sol.fill(0);
    Sp.fill(0);
    for(size_t col = 0; col < gmm::mat_ncols(B); col++)
    {
        size_t shift = col*idz->n;
        if(WavePortsNum > 0)
        {
            if(opt->tfe)
            {
                // saving ports scattering parameters
                for(size_t row = 0; row < WavePortsNum; row++)
                {
                    Sp(row,col) = std::complex<double>((double)idz->rhs[shift+row].r,(double)idz->rhs[shift+row].i);
                }
                // saving inner dof values
                std::complex<double> jk0z0(0.0, 2.0*Const::pi*freq/Const::c0*Const::z0*opt->power);
                for(size_t i=0; i<NonWavePortIds.n_rows; i++)
                {
                    if(NonWavePortIds(i) < UINT_MAX)
                    {
                        Sol(NonWavePortIds(i),col) =  std::complex<double>((double)idz->rhs[shift+i+WavePortsNum].r,
                                                      (double)idz->rhs[shift+i+WavePortsNum].i);
                    }
                }
                idx = 0;
                if(opt->nl)
                {
                    for(size_t bcid = 0; bcid < msh->facBC.size(); bcid++)
                    {
                        BC* bc = &(msh->facBC[bcid]);
                        if(bc->type == BC::WavePort)
                        {
                            for(size_t ih=0; ih<opt->nHarm; ih++)
                            {
                                for(size_t i=0; i<bc->numModes; i++)
                                {
                                    for(size_t j=0; j < bc->ModeVec.n_rows; j++)
                                    {
                                        Sol(bc->ModeVecDoF(j,ih),col) += std::sqrt(jk0z0/bc->ModeBeta(ih*bc->numModes+i))*
                                                                         bc->ModeVec(j,ih*bc->numModes+i)*Sp(idx,col);
                                    }
                                    idx++;
                                }
                            }
                        }
                    }
                }
                else
                {
                    for(size_t bcid = 0; bcid < msh->facBC.size(); bcid++)
                    {
                        BC* bc = &(msh->facBC[bcid]);
                        if(bc->type == BC::WavePort)
                        {
                            for(size_t i=0; i<bc->numModes; i++)
                            {
                                for(size_t j=0; j < bc->ModeVec.n_rows; j++)
                                {
                                    Sol(bc->ModeVecDoF(j),col) += std::sqrt(jk0z0/bc->ModeBeta(i))*bc->ModeVec(j,i)*Sp(idx,col);
                                }
                                idx++;
                            }
                        }
                    }
                }
            }
            else
            {
                // saving dof values
                for(size_t row = 0; row < (size_t) idz->n; row++)
                {
                    if(NonDirIds[row] < UINT_MAX)
                    {
                        Sol(NonDirIds[row], col) = std::complex<double>((double)idz->rhs[shift+row].r,(double)idz->rhs[shift+row].i);
                    }
                }
                // saving ports scattering parameters
                /// Multimode still to be fixed
                std::complex<double> jk0z0(0.0, 2.0*Const::pi*freq/Const::c0*Const::z0);
                arma::uvec jcol(1);
                jcol(0) = col;
                size_t row = 0;
                for(size_t bcid = 0; bcid < msh->facBC.size(); bcid++)
                {
                    BC* bc = &(msh->facBC[bcid]);
                    if(bc->type == BC::WavePort)
                    {
                        #pragma omp parallel for
                        for(size_t in = 0; in < bc->ModeBeta.size(); in++)
                        {
                            arma::cx_mat val;
                            val = (bc->ModeVecf.col(in).st()*
                                   Sol.elem(bc->ModeVecDoF, jcol))*bc->ModeBeta(in)/jk0z0;
                            Sp(row,col) = val(0,0);
                            row++;
                        }
                    }
                }
            }
        }
        else if(opt->einc || opt->stat)
        {
            for(size_t row = 0; row < idz->n; row++)
            {
                Sol(row,0) = std::complex<double>((double)idz->rhs[shift+row].r,(double)idz->rhs[shift+row].i);
            }
        }
    }
    idz->job=-2; // end
    cmumps_c(idz);
    delete idz->a;
    delete idz->rhs;
    delete idz->irn;
    delete idz->jcn;
    delete idz->rhs_sparse;
    delete idz->irhs_sparse;
    delete idz->irhs_ptr;
    delete idz;
    logFile << "+" << tt.toc() << " s\n";
    if(opt->verbose)
    {
        std::cout << tt.toc() << " s\n";
    }
}

void EqSys::SolveDoubleComplex(std::ofstream& logFile)
{
    logFile << "% Double precision Complex Solver ";
    if(opt->verbose)
    {
        std::cout << "Double precision Complex Solver";
    }
    tt.tic();
    if(opt->verbose)
    {
        switch(SymmFlag)
        {
        case 0:
            logFile << " / Non-symmetric\n";
            std::cout << " / Non-symmetric\n";
            break;
        case 1:
            logFile << " / Symmetric\n";
            std::cout << " / Symmetric\n";
            break;
        case 2:
            logFile << " / General symmetric\n";
            std::cout << " / General symmetric\n";
            break;
        }
    }
    ZMUMPS_STRUC_C* idz = new ZMUMPS_STRUC_C;
    idz->job=-1; // init
    idz->par=1;
    idz->sym=(MUMPS_INT) SymmFlag;
    idz->comm_fortran=-987654; // use_comm_world
    zmumps_c(idz);
    if(opt->verbose)
    {
        MemStat::AvailableMemory(std::cout);
    }
    idz->n = (MUMPS_INT) DoFreal;
    idz->nz = (MUMPS_INT) gmm::nnz(A);
    idz->irn = new MUMPS_INT [idz->nz];
    idz->jcn = new MUMPS_INT [idz->nz];
    idz->a = new ZMUMPS_COMPLEX [idz->nz];
    typename gmm::linalg_traits<MatRowType>::const_row_iterator it = mat_row_const_begin(A);
    typename gmm::linalg_traits<MatRowType>::const_row_iterator ite = mat_row_const_end(A);
    size_t i = 0,  idx = 0;
    for(; it != ite; ++it)
    {
        typename gmm::linalg_traits<MatRowType>::const_sub_row_type row = gmm::linalg_traits<MatRowType>::row(it);
        typename gmm::linalg_traits<VecType>::const_iterator itr = vect_const_begin(row);
        typename gmm::linalg_traits<VecType>::const_iterator itre = vect_const_end(row);
        for(; itr != itre; ++itr)
        {
            idz->a[idx].r = (ZMUMPS_REAL) itr->real();
            idz->a[idx].i = (ZMUMPS_REAL) itr->imag();
            idz->irn[idx] = (MUMPS_INT) i+1;
            idz->jcn[idx++] = (MUMPS_INT) itr.index()+1;
        }
        i++;
    }
    idz->icntl[0]=-1; //-1 6
    idz->icntl[1]=-1; //-1 6
    idz->icntl[2]=-1; //-1 1
    idz->icntl[3]= 0; //0 6
    idz->icntl[19]=1; // sparse right hand side
    idz->nrhs = (MUMPS_INT) gmm::mat_ncols(B);
    idz->lrhs = (MUMPS_INT) gmm::mat_nrows(B);
    idz->rhs = new ZMUMPS_COMPLEX [idz->lrhs*idz->nrhs];
    idz->nz_rhs = (MUMPS_INT) gmm::nnz(B) ;
    idz->rhs_sparse = new ZMUMPS_COMPLEX [idz->nz_rhs];
    idz->irhs_sparse = new MUMPS_INT [idz->nz_rhs];
    idz->irhs_ptr = new MUMPS_INT [idz->nrhs+1];
    it = mat_col_const_begin(B);
    ite = mat_col_const_end(B);
    i = 0,  idx = 0;
    for(; it != ite; ++it)
    {
        typename gmm::linalg_traits<MatRowType>::const_sub_row_type row = gmm::linalg_traits<MatRowType>::row(it);
        typename gmm::linalg_traits<VecType>::const_iterator itr = vect_const_begin(row);
        typename gmm::linalg_traits<VecType>::const_iterator itre = vect_const_end(row);
        idz->irhs_ptr[i++] = (MUMPS_INT) idx+1;
        for(; itr != itre; ++itr)
        {
            idz->irhs_sparse[idx] = (MUMPS_INT) itr.index()+1;
            idz->rhs_sparse[idx].r = (ZMUMPS_REAL) itr->real();
            idz->rhs_sparse[idx++].i = (ZMUMPS_REAL) itr->imag();
        }
    }
    idz->irhs_ptr[idz->nrhs] = (MUMPS_INT)(idz->nz_rhs+1);
    // freeing memory
    {
        MatRowType Atmp(idz->n, idz->n);
        MatColType Btmp(idz->n, idz->nrhs);
        std::swap(A, Atmp);
        std::swap(B, Btmp);
    }
    if(opt->verbose)
    {
        MemStat::print(std::cout);
    }
    idz->job=6; // factorization and solution
    if(opt->verbose)
    {
        std::cout << "Factor and Solve ";
    }
    lt.tic();
    zmumps_c(idz);
    if(opt->verbose)
    {
        std::cout << "+ " << idz->info[14] << " MB ";
    }
    logFile << "\tCommit memory: " << idz->info[14] << " MB\n";
    logFile << "\tFactor and Solve: " <<  lt.toc() << " s\n";
    if(opt->nl)
    {
        Sol.resize(DoFnum*opt->nHarm, 1);
        Sp.resize(WavePortsNum, 1);
    }
    else
    {
        Sp.resize(idz->nrhs, idz->nrhs);
        Sol.resize(DoFnum, idz->nrhs);
    }
    Sp.fill(0);
    Sol.fill(0);
    for(size_t col = 0; col < gmm::mat_ncols(B); col++)
    {
        size_t shift = col*idz->n;
        if(WavePortsNum > 0)
        {
            if(opt->tfe)
            {
                // saving ports scattering parameters
                for(size_t row = 0; row < WavePortsNum; row++)
                {
                    Sp(row,col) = std::complex<double>((double)idz->rhs[shift+row].r,(double)idz->rhs[shift+row].i);
                }
                // saving inner dof values
                std::complex<double> jk0z0(0.0, 2.0*Const::pi*freq/Const::c0*Const::z0*opt->power);
                for(size_t i=0; i<NonWavePortIds.n_rows; i++)
                {
                    if(NonWavePortIds(i) < UINT_MAX)
                    {
                        Sol(NonWavePortIds(i),col) =  std::complex<double>((double)idz->rhs[shift+i+WavePortsNum].r,
                                                      (double)idz->rhs[shift+i+WavePortsNum].i);
                    }
                }
                idx = 0;
                if(opt->nl)
                {
                    for(size_t bcid = 0; bcid < msh->facBC.size(); bcid++)
                    {
                        BC* bc = &(msh->facBC[bcid]);
                        if(bc->type == BC::WavePort)
                        {
                            for(size_t ih=0; ih<opt->nHarm; ih++)
                            {
                                for(size_t i=0; i<bc->numModes; i++)
                                {
                                    for(size_t j=0; j < bc->ModeVec.n_rows; j++)
                                    {
                                        Sol(bc->ModeVecDoF(j,ih),col) += std::sqrt(jk0z0/bc->ModeBeta(ih*bc->numModes+i))*
                                                                         bc->ModeVec(j,ih*bc->numModes+i)*Sp(idx,col);
                                    }
                                    idx++;
                                }
                            }
                        }
                    }
                }
                else
                {
                    for(size_t bcid = 0; bcid < msh->facBC.size(); bcid++)
                    {
                        BC* bc = &(msh->facBC[bcid]);
                        if(bc->type == BC::WavePort)
                        {
                            for(size_t i=0; i<bc->numModes; i++)
                            {
                                for(size_t j=0; j < bc->ModeVec.n_rows; j++)
                                {
                                    Sol(bc->ModeVecDoF(j),col) += std::sqrt(jk0z0/bc->ModeBeta(i))*bc->ModeVec(j,i)*Sp(idx,col);
                                }
                                idx++;
                            }
                        }
                    }
                }
            }
            else
            {
                // saving dof values
                for(size_t row = 0; row < (size_t) idz->n; row++)
                {
                    if(NonDirIds[row] < UINT_MAX)
                    {
                        Sol(NonDirIds[row], col) = std::complex<double>((double)idz->rhs[shift+row].r,(double)idz->rhs[shift+row].i);
                    }
                }
                // saving ports scattering parameters
                /// Multimode still to be fixed
                std::complex<double> jk0z0(0.0, 2.0*Const::pi*freq/Const::c0*Const::z0);
                arma::uvec jcol(1);
                jcol(0) = col;
                size_t row = 0;
                for(size_t bcid = 0; bcid < msh->facBC.size(); bcid++)
                {
                    BC* bc = &(msh->facBC[bcid]);
                    if(bc->type == BC::WavePort)
                    {
                        #pragma omp parallel for
                        for(size_t in = 0; in < bc->ModeBeta.size(); in++)
                        {
                            arma::cx_mat val;
                            val = (bc->ModeVecf.col(in).st()*
                                   Sol.elem(bc->ModeVecDoF, jcol)/(jk0z0/bc->ModeBeta(in)));
                            Sp(row,col) = val(0,0);
                            row++;
                        }
                    }
                }
            }
        }
        else if(opt->einc || opt->stat)
        {
            for(size_t row = 0; row < idz->n; row++)
            {
                Sol(row,0) = std::complex<double>((double)idz->rhs[shift+row].r,(double)idz->rhs[shift+row].i);
            }
        }
    }
    idz->job=-2; // end
    zmumps_c(idz);
    delete idz->a;
    delete idz->rhs;
    delete idz->irn;
    delete idz->jcn;
    delete idz->rhs_sparse;
    delete idz->irhs_sparse;
    delete idz->irhs_ptr;
    delete idz;
    logFile << "+" << tt.toc() << " s\n";
    if(opt->verbose)
    {
        std::cout << tt.toc() << " s\n";
    }
}

void EqSys::SolveGmRes(std::ofstream& logFile)
{
    logFile << "% GMRES solution:\n";
    std::cout << "GMRES: ";
    tt.tic();
    if(!opt->sparam || opt->einc || opt->stat)
    {
        Sol.resize(DoFnum, 1);
        Sol.fill(0);
    }
    else
    {
        Sp.resize(WavePortsNum, WavePortsNum);
        Sp.fill(0);
        Sol.resize(DoFnum, WavePortsNum);
        Sol.fill(0);
    }
    if(gmm::nnz(PR) == 0)
    {
        EqSys::MatRowType Atmp(DoFreal, DoFreal);
        #pragma omp parallel for
        for(size_t i=0; i < DoFreal; i++)
        {
            Atmp(i,i) = A(i,i);
            A(i,i) = 0;
        }
        gmm::add(gmm::transposed(A), Atmp);
        gmm::add(A, Atmp);
        std::swap(A, Atmp);
        // freeing memory
        {
            MatRowType Atmp2(DoFreal, DoFreal);
            std::swap(Atmp, Atmp2);
        }
        if(opt->verbose)
        {
            std::cout << "ILU preconditioning";
        }
        lt.tic();
        gmm::ilu_precond<MatRowType> P(A);
        //gmm::identity_matrix P;
        if(opt->verbose)
        {
            std::cout << " " << lt.toc() << " s\n";
        }
        if(WavePortsNum > 0)
        {
            for(size_t col = 0; col < WavePortsNum; col++)
            {
                std::vector<std::complex<double> > X(DoFreal), Brhs(DoFreal);
                #pragma omp parallel for
                for(size_t row = 0; row < DoFreal; row++)
                {
                    Brhs[row] = B(row,col);
                }
                gmm::iteration iter(opt->toll);
                gmm::gmres(A, X, Brhs, P, opt->niter, iter);
                std::cout << "Iters " << col << " ~ " << iter.get_iteration() << "\n";
                if(opt->tfe)
                {
                    // saving ports scattering parameters
                    for(size_t row = 0; row < WavePortsNum; row++)
                    {
                        Sp(row,col) = X[row];
                    }
                    // saving inner dof values
                    size_t shift = col*DoFreal;
                    std::complex<double> jk0z0(0.0, 2.0*Const::pi*freq/Const::c0*Const::z0*opt->power);
                    for(size_t i=0; i<NonWavePortIds.n_rows; i++)
                    {
                        if(NonWavePortIds(i) < UINT_MAX)
                        {
                            Sol(NonWavePortIds(i),col) = X[WavePortsNum+i];
                        }
                    }
                    size_t idx = 0;
                    if(opt->nl)
                    {
                        for(size_t bcid = 0; bcid < msh->facBC.size(); bcid++)
                        {
                            BC* bc = &(msh->facBC[bcid]);
                            if(bc->type == BC::WavePort)
                            {
                                for(size_t ih=0; ih<opt->nHarm; ih++)
                                {
                                    for(size_t i=0; i<bc->numModes; i++)
                                    {
                                        for(size_t j=0; j < bc->ModeVec.n_rows; j++)
                                        {
                                            Sol(bc->ModeVecDoF(j,ih),col) += std::sqrt(jk0z0/bc->ModeBeta(ih*bc->numModes+i))*
                                                                             bc->ModeVec(j,ih*bc->numModes+i)*Sp(idx,col);
                                        }
                                        idx++;
                                    }
                                }
                            }
                        }
                    }
                    else
                    {
                        for(size_t bcid = 0; bcid < msh->facBC.size(); bcid++)
                        {
                            BC* bc = &(msh->facBC[bcid]);
                            if(bc->type == BC::WavePort)
                            {
                                for(size_t i=0; i<bc->numModes; i++)
                                {
                                    for(size_t j=0; j < bc->ModeVec.n_rows; j++)
                                    {
                                        Sol(bc->ModeVecDoF(j),col) += std::sqrt(jk0z0/bc->ModeBeta(i))*bc->ModeVec(j,i)*Sp(idx,col);
                                    }
                                    idx++;
                                }
                            }
                        }
                    }
                }
                else
                {
                    // saving dof values
                    for(size_t row = 0; row < DoFreal; row++)
                    {
                        if(NonDirIds[row] < UINT_MAX)
                        {
                            Sol(NonDirIds[row],col) = X[row];
                        }
                    }
                    // saving ports scattering parameters
                    /// Multimode still to be fixed
                    std::complex<double> jk0z0(0.0, 2.0*Const::pi*freq/Const::c0*Const::z0);
                    arma::uvec jcol(1);
                    jcol(0) = col;
                    size_t row = 0;
                    for(size_t bcid = 0; bcid < msh->facBC.size(); bcid++)
                    {
                        BC* bc = &(msh->facBC[bcid]);
                        if(bc->type == BC::WavePort)
                        {
                            for(size_t in = 0; in < bc->ModeBeta.size(); in++)
                            {
                                arma::cx_mat val(bc->ModeVecf.col(in).st()*
                                                 Sol.elem(bc->ModeVecDoF, jcol)/(jk0z0/bc->ModeBeta(in)));
                                Sp(row,col) = val(0,0);
                                row++;
                            }
                        }
                    }
                }
            }
        }
        else if(opt->einc || opt->stat)
        {
            std::vector<std::complex<double> > X(DoFreal), Brhs(DoFreal);
            gmm::iteration iter(opt->toll);
            #pragma omp parallel for
            for(size_t row = 0; row < DoFreal; row++)
            {
                Brhs[row] = B(row,0);
            }
            gmm::gmres(A, X, Brhs, P, opt->niter, iter);
            std::cout << "Iters  ~ " << iter.get_iteration() << "\n";
            for(size_t row = 0; row < DoFreal; row++)
            {
                Sol(row,0) = X[row];
            }
        }
    }
    else     /// DD Preconditioned GMRES ///
    {
        MemStat::print(std::cout);
        if(WavePortsNum > 0)
        {
            if(!opt->sparam)
            {
                std::vector<std::complex<double> > X(DoFreal), Brhs(DoFreal);
                gmm::iteration iter(opt->toll, 0, -1, 10);
                for(size_t col = 0; col < WavePortsNum; col++)
                {
                    #pragma omp parallel for
                    for(size_t row = 0; row < DoFreal; row++)
                    {
                        Brhs[row] += B(row,col);
                    }
                }
                GMRES(A, PR, X, Brhs, DoFlevel, opt->niter, iter, logFile, opt);
                std::cout << "Iters ~ " << iter.get_iteration() << "\n";
                // saving dof values
                for(size_t row = 0; row < DoFreal; row++)
                {
                    if(NonDirIds[row] < UINT_MAX)
                    {
                        Sol(NonDirIds[row],0) = X[row];
                    }
                }
            }
            else
            {
                for(size_t col = 0; col < WavePortsNum; col++)
                {
                    std::vector<std::complex<double> > X(DoFreal), Brhs(DoFreal);
                    gmm::iteration iter(opt->toll, 0, -1, 10);
                    #pragma omp parallel for
                    for(size_t row = 0; row < DoFreal; row++)
                    {
                        Brhs[row] = B(row,col);
                    }
                    GMRES(A, PR, X, Brhs, DoFlevel, opt->niter, iter, logFile, opt);
                    std::cout << "Iters " << col << " ~ " << iter.get_iteration() << "\n";
                    // saving dof values
                    if(opt->tfe)
                    {
                        // saving ports scattering parameters
                        for(size_t row = 0; row < WavePortsNum; row++)
                        {
                            Sp(row,col) = X[row];
                        }
                        // saving inner dof values
                        size_t shift = col*DoFreal;
                        std::complex<double> jk0z0(0.0, 2.0*Const::pi*freq/Const::c0*Const::z0*opt->power);
                        for(size_t i=0; i<NonWavePortIds.n_rows; i++)
                        {
                            if(NonWavePortIds(i) < UINT_MAX)
                            {
                                Sol(NonWavePortIds(i),col) = X[WavePortsNum+i];
                            }
                        }
                        size_t idx = 0;
                        if(opt->nl)
                        {
                            for(size_t bcid = 0; bcid < msh->facBC.size(); bcid++)
                            {
                                BC* bc = &(msh->facBC[bcid]);
                                if(bc->type == BC::WavePort)
                                {
                                    for(size_t ih=0; ih<opt->nHarm; ih++)
                                    {
                                        for(size_t i=0; i<bc->numModes; i++)
                                        {
                                            for(size_t j=0; j < bc->ModeVec.n_rows; j++)
                                            {
                                                Sol(bc->ModeVecDoF(j,ih),col) += std::sqrt(jk0z0/bc->ModeBeta(ih*bc->numModes+i))*
                                                                                 bc->ModeVec(j,ih*bc->numModes+i)*Sp(idx,col);
                                            }
                                            idx++;
                                        }
                                    }
                                }
                            }
                        }
                        else
                        {
                            for(size_t bcid = 0; bcid < msh->facBC.size(); bcid++)
                            {
                                BC* bc = &(msh->facBC[bcid]);
                                if(bc->type == BC::WavePort)
                                {
                                    for(size_t i=0; i<bc->numModes; i++)
                                    {
                                        for(size_t j=0; j < bc->ModeVec.n_rows; j++)
                                        {
                                            Sol(bc->ModeVecDoF(j),col) += std::sqrt(jk0z0/bc->ModeBeta(i))*bc->ModeVec(j,i)*Sp(idx,col);
                                        }
                                        idx++;
                                    }
                                }
                            }
                        }
                    }
                    else
                    {
                        // saving dof values
                        for(size_t row = 0; row < DoFreal; row++)
                        {
                            if(NonDirIds[row] < UINT_MAX)
                            {
                                Sol(NonDirIds[row],col) = X[row];
                            }
                        }
                        // saving ports scattering parameters
                        /// Multimode still to be fixed
                        std::complex<double> jk0z0(0.0, 2.0*Const::pi*freq/Const::c0*Const::z0);
                        arma::uvec jcol(1);
                        jcol(0) = col;
                        size_t row = 0;
                        for(size_t bcid = 0; bcid < msh->facBC.size(); bcid++)
                        {
                            BC* bc = &(msh->facBC[bcid]);
                            if(bc->type == BC::WavePort)
                            {
                                for(size_t in = 0; in < bc->ModeBeta.size(); in++)
                                {
                                    arma::cx_mat val(bc->ModeVecf.col(in).st()*
                                                     Sol.elem(bc->ModeVecDoF, jcol)/(jk0z0/bc->ModeBeta(in)));
                                    Sp(row,col) = val(0,0);
                                    row++;
                                }
                            }
                        }
                    }
//                    for(size_t row = 0; row < DoFreal; row++) {
//                        if(NonDirIds[row] < UINT_MAX) {
//                            Sol(NonDirIds[row],col) = X[row];
//                        }
//                    }
//                    // saving ports scattering parameters
//                    /// Multimode still to be fixed
//                    std::complex<double> jk0z0(0.0, 2.0*Const::pi*freq/Const::c0*Const::z0);
//                    arma::uvec jcol(1);
//                    jcol(0) = col;
//                    size_t row = 0;
//                    for(size_t bcid = 0; bcid < msh->facBC.size(); bcid++) {
//                        BC* bc = &(msh->facBC[bcid]);
//                        if(bc->type == BC::WavePort) {
//                            #pragma omp parallel for
//                            for(size_t in = 0; in < bc->ModeBeta.size(); in++) {
//                                //std::cout << in << "\n";
//                                arma::cx_mat val(bc->ModeVecf.col(in).st()*
//                                                 Sol.elem(bc->ModeVecDoF, jcol)/(jk0z0/bc->ModeBeta(in)));
//                                Sp(row,col) = val(0,0);
//                                row++;
//                            }
//                        }
//                    }
                } // end col loop
            }
        }
        else if(opt->einc || opt->stat)
        {
            std::vector<std::complex<double> > X(DoFreal), Brhs(DoFreal);
            gmm::iteration iter(opt->toll, 0, -1, 10);
            #pragma omp parallel for
            for(size_t row = 0; row < DoFreal; row++)
            {
                Brhs[row] = B(row,0);
            }
            GMRES(A, PR, X, Brhs, DoFlevel, opt->niter, iter, logFile, opt);
            std::cout << "Iters  ~ " << iter.get_iteration() << "\n";
            for(size_t row = 0; row < DoFreal; row++)
            {
                if(InvDoFmapv(row) < UINT_MAX)
                {
                    Sol(InvDoFmapv(row),0) = X[row];
                }
            }
        }
    }
    logFile << "+" << tt.toc() << " s\n";
    std::cout << tt.toc() << " s\n";
}
void EqSys::SaveSystem(std::ofstream& logFile)
{
    if(DoFlevel.size() > 0)
    {
        logFile << "% Saving blocks sizes:\n";
        if(opt->verbose)
        {
            std::cout << "Blocks, ";
        }
        std::ofstream BlocksFile(std::string("Blocks.txt").c_str());
        for(int i=0; i< DoFlevel.size(); i++)
        {
            BlocksFile << DoFlevel[i] << "\n";
        }
        BlocksFile.close();
        if(opt->dds)
        {
            gmm::csc_matrix<std::complex<double> > AFFout;
            for(size_t i=0; i < AFF.size(); i++)
            {
                if(opt->verbose)
                {
                    std::cout << "AFF" << i << ".mm, ";
                }
                gmm::copy(AFF[i], AFFout);
                std::stringstream tmp;
                tmp << "AFF" << i << ".mm";
                gmm::MatrixMarket_save(std::string(tmp.str()).data(), AFFout);
            }
        }
    }
    logFile << "% Saving system matrices:\n";
    gmm::csc_matrix<std::complex<double> > Aout, Bout, Pout;
    if(gmm::nnz(PR) > 0)
    {
        if(opt->verbose)
        {
            std::cout << "P.mm, ";
        }
        gmm::copy(PR, Pout);
        gmm::MatrixMarket_save(std::string("P.mm").data(), Pout);
    }
    if(opt->verbose)
    {
        std::cout << "A.mm, ";
    }
    gmm::copy(A, Aout);
    gmm::MatrixMarket_save(std::string("A.mm").data(), Aout);
    if(opt->verbose)
    {
        std::cout << "B.mm\n";
    }
    gmm::copy(B, Bout);
    gmm::MatrixMarket_save(std::string("B.mm").data(), Bout);
    logFile << "+" << tt.toc() << " s\n";
}
void EqSys::SaveData(std::ofstream& logFile)
{
    logFile << "% Saving data:\n";
    tt.tic();
    if(opt->sparam)
    {
        if(opt->verbose)
        {
            std::cout << "Saving S-parameters\n";
        }
        lt.tic();
        std::ofstream sParamFile(std::string(opt->name + "_S.txt").c_str(), std::ios::app);
        arma::cx_mat sParams;
        if(opt->nl)
        {
            sParams = Sp(arma::span(0,WavePortsNum-1),0);
            sParams(0) -= 1.0;
        }
        else
        {
            sParams = Sp(arma::span(0,WavePortsNum-1),arma::span(0,WavePortsNum-1)) -
                      arma::eye<arma::mat>(WavePortsNum,WavePortsNum);
        }
        sParams *= std::sqrt(opt->power);
        sParamFile << std::setw(10) << std::scientific << std::left << std::setprecision(8) << freq;
        for(int i=0; i< sParams.n_cols; i++)
        {
            for(int j=0; j< sParams.n_rows; j++)
            {
                sParamFile << " " << std::setw(18) << std::scientific
                           << std::right << std::setprecision(15)
                           << 20*std::log10(std::abs(sParams(j,i)));
            }
        }
        sParamFile << "\n";
        sParamFile.close();
        logFile << "\tS parameters: " << lt.toc() << " s\n";
        if(opt->verbose)
        {
            std::cout << "|S| =\n" << 20*arma::log10(arma::abs(sParams));
        }
        sParams.clear();
    }
    if(opt->sol)
    {
        if(opt->verbose)
        {
            std::cout << "Saving solution vector\n";
        }
        lt.tic();
        std::ofstream solFile(std::string(opt->name + "_Sol.txt").c_str());//, std::ios::app);
        //solFile << "\n\n### iter " << iter << "\n";
        for(int i=0; i< Sol.n_rows; i++)
        {
            for(int j=0; j< Sol.n_cols; j++)
            {
                solFile << std::setw(23) << std::scientific
                        << std::right << std::setprecision(15)
                        << Sol(i,j).real() << " "
                        << std::setw(23) << std::scientific
                        << std::right << std::setprecision(15)
                        << Sol(i,j).imag() << " ";
            }
            solFile << "\n";
        }
        solFile.close();
    }
    for(size_t i=0; i<PortAmpl.size(); i++)
    {
        Sol.col(i) *= PortAmpl[i];
    }
    if(opt->field && opt->solver != Option::MATLAB)
    {
        if(opt->verbose)
        {
            std::cout << "Saving fields\n";
        }
        lt.tic();
        Field(prj, Sol, freq);
        logFile << "\tFields: " << lt.toc() << " s\n";
    }
    if(opt->rad && opt->solver != Option::MATLAB)
    {
        if(opt->verbose)
        {
            std::cout << "Saving radiation: ";
        }
        lt.tic();
        //std::cout << arma::abs(Sp);
        double Pref = 0.0;
        double Pinc = 0.0;
        double Pacc = 0.0;
        for(int i=0; i < Sp.n_rows; i++)
        {
            Pref += std::pow(std::abs(Sp(i,i) - 1.0),2);
            Pinc += 1.0;
        }
        Pacc = Pinc - Pref;
//        if(opt->sparam) {
        Rad(prj, Sol, freq, Pacc);
//        } else {
//            Rad(prj, Sol, freq, Pinc);
//        }
        logFile << "\tRadiation: " << lt.toc() << " s\n";
    }
    logFile << "+" << tt.toc() << " s\n";
}
EqSys::~EqSys()
{
    delete quad;
}
