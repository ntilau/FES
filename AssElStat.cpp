#include "AssElStat.h"

#include <armadillo>
#include "Config.h"
#include "Mem.h"
#include "EqSys.h"
#include "DoF.h"
#include "EleMat.h"
#include "Shape.h"

#include <cfloat>

AssElStat::AssElStat(std::ofstream& logFile, EqSys* sys): prj(sys->prj), msh(sys->msh), opt(sys->opt), quad(sys->quad)
{
    logFile << "% Assembly Electrostatic:\n";
    logFile << "In solids: ";
    arma::wall_clock tt, lt;
    tt.tic();
    sys->DoFnum = DoF(prj).DoFnums;
    if(opt->verbose)
    {
        std::cout << "FE DoF = " << sys->DoFnum << " ";
    }
    sys->SymmFlag = 0;
    sys->A.clear_mat();
    sys->B.clear_mat();
    gmm::resize(sys->A, sys->DoFnum, sys->DoFnum);
    gmm::resize(sys->B, sys->DoFnum, 1);
    lt.tic();
    #pragma omp parallel for
    for(size_t id = 0; id < msh->nTetras; id++)
    {
        Mtrl* cMtrl = &(msh->tetMtrl[msh->tetLab(id)]);
        EleMat lMat(opt->pOrd, 3, msh->tetGeo(id), quad, cMtrl, Shape::Hgrad);
        DoF cDoF(prj, 3, id);
        #pragma omp critical
        for(int i=0; i<cDoF.s.n_rows; i++)
        {
            for(int j=0; j<cDoF.s.n_rows; j++)
            {
                sys->A(cDoF.s(i),cDoF.s(j)) += lMat.S(i,j);
            }
        }
    }
    logFile << lt.toc() << " s\n";
    if(opt->verbose)
    {
        std::cout << lt.toc() << " s\n";
    }
    MemStat::print(logFile);
    if(opt->verbose)
    {
        MemStat::print(std::cout);
    }
    logFile << "On boundaries:\n";
    for(size_t bcid = 0; bcid < msh->facBC.size(); bcid++)
    {
        BC* bc = &(msh->facBC[bcid]);
        if(bc->type == BC::PerfectE)
        {
            lt.tic();
            double f = opt->Vbnd[bc->name];
            if(opt->verbose)
            {
                std::cout << bc->name << "(" << f << "V)";
            }
            logFile << "\t" << bc->name << ": ";
            #pragma omp parallel for
            for(size_t fid = 0; fid < bc->Faces.size(); fid++)
            {
                DoF cDoF(prj, 2, bc->Faces(fid));
                #pragma omp critical
                {
                    sys->DirDoFs = arma::join_cols(sys->DirDoFs, cDoF.s);
                    for(int i=0; i<cDoF.s.n_rows; i++)
                    {
                        sys->B(cDoF.s(i),0) = f;
                    }
                }
            }
            sys->DirDoFs = arma::unique(sys->DirDoFs);
            logFile << lt.toc() << " s\n";
            if(opt->verbose)
            {
                std::cout << " ";
            }
        }
    }
    sys->WavePortsNum = 0;
    sys->DoFreal = sys->DoFnum;
    if(opt->verbose)
    {
        std::cout << "\nSYS DoF = " << sys->DoFreal << "\n";
    }
    std::vector<size_t> DirIds(sys->DirDoFs.size());
    #pragma omp parallel for
    for(size_t i=0; i<sys->DirDoFs.size(); i++)
    {
        DirIds[i] = sys->DirDoFs(i);
    }
    EqSys::VecType Bnew(sys->DoFreal), Bcopy(sys->DoFreal);
//    EqSys::MatRowType Adiag(sys->DoFreal,sys->DoFreal);
//    #pragma omp parallel for
//    for(size_t i=0; i<sys->DoFreal; i++) {
//        Adiag(i,i) = -sys->A(i,i);
//    }
    gmm::copy(gmm::mat_col(sys->B,0), Bcopy);
    gmm::mult(sys->A, Bcopy, Bnew);
    //gmm::mult_add(gmm::transposed(sys->A), Bcopy, Bnew);
    //gmm::mult_add(Adiag, Bcopy, Bnew);
    gmm::clear(gmm::sub_vector(Bnew, gmm::sub_index(DirIds)));
    gmm::add(gmm::scaled(Bnew,-1.0), gmm::mat_col(sys->B,0));
    //gmm::copy(Bcopy, gmm::mat_col(sys->B,0));
    gmm::clear(gmm::sub_matrix(sys->A, gmm::sub_index(DirIds), gmm::sub_interval(0, sys->DoFreal)));
    gmm::clear(gmm::sub_matrix(sys->A, gmm::sub_interval(0, sys->DoFreal), gmm::sub_index(DirIds)));
    #pragma omp parallel for
    for(size_t i=0; i<DirIds.size(); i++)
    {
        sys->A(DirIds[i],DirIds[i]) = 1.0;
    }
    if(gmm::mat_euclidean_norm(sys->B) == 0)
    {
        throw std::string("Null Right Hand Side");
    }
    logFile << " " << lt.toc() << " s\n";
    logFile << "+" << tt.toc() << "s\n";
}

AssElStat::~AssElStat()
{
    //dtor
}

