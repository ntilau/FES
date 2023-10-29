#include "AssNL.h"

#include <armadillo>
#include "Const.h"
#include "Mem.h"
#include "EqSys.h"
#include "DoF.h"
#include "EleMat.h"
#include "Eigen.h"

#include <cfloat> // for DBL_MIN

AssNL::AssNL(std::ofstream& logFile, EqSys* sys): prj(sys->prj), msh(sys->msh), opt(sys->opt), quad(sys->quad)
{
    double hCoeff[] = {1,3,5,7,9,11,13};
    logFile << "% Assembly:\n";
    logFile << "In solids: ";
    arma::wall_clock tt, lt;
    tt.tic();
    double k0 = 2.0 * Const::pi * sys->freq / Const::c0;
    double kk = k0*k0;
    sys->DoFnum = DoF(prj).DoFnumv;
    logFile << "% Assembly nonlinear:\n";
    DoF* cDoF = new DoF(prj, 3, 0);
    size_t num3 = cDoF->v.n_rows;
    cDoF = new DoF(prj, 2, 0);
    size_t num2 = cDoF->v.n_rows;
    delete cDoF;
    std::cout << "FE DoF = " << sys->DoFnum* opt->nHarm << " ";
    sys->SymmFlag = 0;
    sys->A.clear_mat();
    sys->B.clear_mat();
    sys->PR.clear_mat();
    gmm::resize(sys->A, opt->nHarm*sys->DoFnum, opt->nHarm*sys->DoFnum);
    if(sys->Solprev.n_rows==0)
    {
        sys->Solprev.resize(opt->nHarm*sys->DoFnum, 1);
        sys->Solprev.fill(0);
    }
    sys->Sol.clear();
    sys->Sp.clear();
    lt.tic();
    arma::vec k0Coeff(opt->nHarm*num3);
    arma::vec kkCoeff(opt->nHarm*num3);
    k0Coeff.fill(k0);
    kkCoeff.fill(kk);
    for(size_t ih=1; ih < opt->nHarm; ih++)
    {
        for(size_t i=0; i < num3; i++)
        {
            k0Coeff(ih*num3+i) = hCoeff[ih]*k0;
            kkCoeff(ih*num3+i) = std::pow(hCoeff[ih],2)*kk;
        }
    }
    #pragma omp parallel for
    for(size_t id = 0; id < msh->nTetras; id++)
    {
        DoF cDoF(prj, 3, id);
        cDoF.v.resize(opt->nHarm*num3);
        for(size_t ih=1; ih < opt->nHarm; ih++)
        {
            for(size_t i=0; i < num3; i++)
            {
                cDoF.v(ih*num3+i) = ih*sys->DoFnum + cDoF.v(i);
            }
        }
        Mtrl* cMtrl = &(msh->tetMtrl[msh->tetLab(id)]);
        EleMat lMat(opt->pOrd, 3, msh->tetGeo(id), quad, cMtrl, &cDoF, sys->Solprev.col(0), opt->nHarm, sys->DoFnum, sys->freq);
        #pragma omp critical
        for(size_t i=0; i<cDoF.v.n_rows; i++)
        {
            for(size_t j=0; j<cDoF.v.n_rows; j++)
            {
                sys->A(cDoF.v(i),cDoF.v(j)) += lMat.S(i,j) + k0Coeff(i)*lMat.Z(i,j) - kkCoeff(i)*lMat.T(i,j);
            }
        }
    }
    logFile << lt.toc() << " s\n";
    if(opt->verbose)
    {
        std::cout << lt.toc() << " s\n";
    }
    MemStat::print(logFile);
    MemStat::print(std::cout);
    logFile << "On boundaries:\n";
    for(size_t bcid = 0; bcid < msh->facBC.size(); bcid++)
    {
        BC* bc = &(msh->facBC[bcid]);
        switch(bc->type)
        {
        case BC::PerfectE :
            lt.tic();
            std::cout << bc->name;
            logFile << "\t" << bc->name << ": ";
            for(size_t fid = 0; fid < bc->Faces.size(); fid++)
            {
                DoF cDoF(prj, 2, bc->Faces(fid));
                cDoF.v.resize(opt->nHarm*num2);
                for(size_t ih=1; ih < opt->nHarm; ih++)
                {
                    for(size_t i=0; i < num2; i++)
                    {
                        cDoF.v(ih*num2+i) = ih*sys->DoFnum + cDoF.v(i);
                    }
                }
                sys->DirDoFs = arma::join_cols(sys->DirDoFs, cDoF.s);
                sys->DirDoFv = arma::join_cols(sys->DirDoFv, cDoF.v);
            }
            sys->DirDoFs = arma::unique(sys->DirDoFs);
            sys->DirDoFv = arma::unique(sys->DirDoFv);
            logFile << lt.toc() << " s\n";
            std::cout << " ";
            break;
        case BC::WavePort:
            break;
        case BC::Radiation:
            lt.tic();
            std::cout << bc->name;
            logFile << "\t" << bc->name << ": ";
            for(size_t ih=0; ih < opt->nHarm; ih++)
            {
                arma::mat centroid(1,3);
                centroid.fill(0.25);
                for(size_t fid = 0; fid < bc->Faces.size(); fid++)
                {
                    arma::uvec adjTet = msh->facAdjTet(bc->Faces(fid));
                    Mtrl* cMtrl = &(msh->tetMtrl[msh->tetLab(adjTet(0))]);
                    cMtrl->updMtrl(sys->freq);
                    std::complex<double> epsr(cMtrl->epsr, cMtrl->epsr2);
                    double mur = cMtrl->mur;
                    if(cMtrl->kerr != 0.0)
                    {
                        DoF cDoF3(prj, 3, adjTet(0));
                        Jacobian cJac(3, msh->tetGeo(adjTet(0)));
                        Shape cShp(opt->pOrd, 3, Shape::Hcurl, centroid.row(0), &cJac);
                        arma::uvec tDoFs = cDoF3.v + (size_t)(sys->DoFnum*ih);
                        arma::cx_vec tSol = arma::cx_vec(sys->Solprev.col(0)).elem(tDoFs);
                        double normE = arma::norm(cShp.Nv*tSol,2);
                        epsr *= (1 + cMtrl->kerr*std::pow(normE,2));
                    }
                    EleMat lMat(opt->pOrd, 2, msh->facGeo(bc->Faces(fid)), quad,
                                cMtrl, msh->intNode(bc->Faces(fid)));
                    DoF cDoF(prj, 2, bc->Faces(fid));
                    for(int i=0; i<cDoF.v.n_rows; i++)
                    {
                        for(int j=0; j<cDoF.v.n_rows; j++)
                        {
                            sys->A(ih*sys->DoFnum + cDoF.v(i),ih*sys->DoFnum + cDoF.v(j)) +=
                                std::complex<double>(0.0,(hCoeff[ih])*k0)*lMat.Tt(i,j)*std::sqrt(epsr/mur);
                        }
                    }
                }
            }
            logFile << lt.toc() << " s\n";
            std::cout << " ";
            break;
        case BC::PerfectH:
            lt.tic();
            std::cout << bc->name;
            logFile << "\t" << bc->name << ": ";
            logFile << lt.toc() << " s\n";
            std::cout << " ";
            break;
        default:
            throw std::string("Wrong boundary type");
        }
    }
    sys->WavePortsNum = 0;
    sys->WavePortsDoFnum = 0;
    sys->WavePortIds.reset();
    for(size_t bcid = 0; bcid < msh->facBC.size(); bcid++)
    {
        BC* bc = &(msh->facBC[bcid]);
        if(bc->type == BC::WavePort)
        {
            lt.tic();
            if(opt->verbose)
            {
                std::cout << bc->name;
            }
            logFile << "\t" << bc->name << ": ";
            arma::uvec tmpEdges, tmpDoFv, tmpDirDoFv;
            arma::uvec tmpNodes, tmpDoFs, tmpDirDoFs;
            arma::uvec tmp;
            size_t numt, numz, numtot;
            tmpDoFs = arma::zeros<arma::uvec>(DoF(prj).DoFnums);
            tmpDoFv = arma::zeros<arma::uvec>(DoF(prj).DoFnumv);
            for(size_t fid = 0; fid < bc->Faces.size(); fid++)
            {
                DoF cDoF(prj, 2, bc->Faces(fid));
                tmpEdges = arma::join_cols(tmpEdges, cDoF.v);
                tmpNodes = arma::join_cols(tmpNodes, cDoF.s);
            }
            tmpNodes = arma::unique(tmpNodes);
            tmpEdges = arma::unique(tmpEdges);
            tmp.reset();
            tmp = arma::uvec(tmpEdges.n_rows);
            tmp.fill(0);
            for(size_t i = 0; i < tmpEdges.n_rows; i++)
            {
                tmpDoFv(tmpEdges(i)) = i;
                arma::uvec pD = arma::find(sys->DirDoFv == tmpEdges(i));
                if(pD.n_rows == 0)
                {
                    tmp(i) = 1;
                }
            }
            tmpDirDoFv = arma::find(tmp > 0);
            tmp.reset();
            tmp = arma::uvec(tmpNodes.n_rows);
            tmp.fill(0);
            for(size_t i = 0; i < tmpNodes.n_rows; i++)
            {
                tmpDoFs(tmpNodes(i)) = i;
                arma::uvec pD = arma::find(sys->DirDoFs == tmpNodes(i));
                if(pD.n_rows == 0)
                {
                    tmp(i) = 1;
                }
            }
            tmpDirDoFs = arma::find(tmp > 0);
            arma::mat centroid(1,3);
            centroid.fill(0.25);
            for(size_t ih=0; ih < opt->nHarm; ih++)
            {
                arma::cx_mat tmpSt(tmpEdges.n_rows,tmpEdges.n_rows);
                arma::cx_mat tmpTt(tmpEdges.n_rows,tmpEdges.n_rows);
                arma::cx_mat tmpTt2(tmpEdges.n_rows,tmpEdges.n_rows);
                arma::cx_mat tmpSz(tmpNodes.n_rows,tmpNodes.n_rows);
                arma::cx_mat tmpTz(tmpNodes.n_rows,tmpNodes.n_rows);
                arma::cx_mat tmpG(tmpEdges.n_rows,tmpNodes.n_rows);
                tmpSt.fill(0);
                tmpTt.fill(0);
                tmpTt2.fill(0);
                tmpSz.fill(0);
                tmpTz.fill(0);
                tmpG.fill(0);
                double maxepsr = DBL_MIN;
                double maxmur = DBL_MIN;
                for(size_t fid = 0; fid < bc->Faces.size(); fid++)
                {
                    arma::uvec adjTet = msh->facAdjTet(bc->Faces(fid));
                    Mtrl* cMtrl = &(msh->tetMtrl[msh->tetLab(adjTet(0))]);
                    double cFreq = (hCoeff[ih])*sys->freq;
                    cMtrl->updMtrl(cFreq);
                    std::complex<double> epsr(cMtrl->epsr, cMtrl->epsr2);
                    double mur = cMtrl->mur;
                    if(cMtrl->kerr != 0.0)
                    {
                        DoF cDoF3(prj, 3, adjTet(0));
                        Jacobian cJac(3, msh->tetGeo(adjTet(0)));
                        Shape cShp(opt->pOrd, 3, Shape::Hcurl, centroid.row(0), &cJac);
                        arma::uvec tDoFs = cDoF3.v + (size_t)(sys->DoFnum*ih);
                        arma::cx_vec tSol = arma::cx_vec(sys->Solprev.col(0)).elem(tDoFs);
                        double normE = arma::norm(cShp.Nv*tSol,2);
                        epsr *=  std::complex<double>(1 + cMtrl->kerr*std::pow(normE,2),0.0);
                    }
                    maxepsr = std::max(maxepsr, std::real(epsr));
                    maxmur = std::max(maxmur, mur);
                    EleMat lMat(opt->pOrd, 2, msh->facGeo(bc->Faces(fid)), quad,
                                cMtrl, msh->intNode(bc->Faces(fid)));
                    DoF cDoF(prj, 2, bc->Faces(fid));
                    for(int i=0; i<cDoF.v.n_rows; i++)
                    {
                        for(int j=0; j<cDoF.v.n_rows; j++)
                        {
                            tmpSt(tmpDoFv(cDoF.v(i)),tmpDoFv(cDoF.v(j))) += lMat.St(i,j)/mur;
                            tmpTt(tmpDoFv(cDoF.v(i)),tmpDoFv(cDoF.v(j))) += lMat.Tt(i,j)*epsr;
                            tmpTt2(tmpDoFv(cDoF.v(i)),tmpDoFv(cDoF.v(j))) += lMat.Tt(i,j)/mur;
                        }
                    }
                    for(int i=0; i<cDoF.s.n_rows; i++)
                    {
                        for(int j=0; j<cDoF.s.n_rows; j++)
                        {
                            tmpSz(tmpDoFs(cDoF.s(i)),tmpDoFs(cDoF.s(j))) += lMat.Sz(i,j)/mur;
                            tmpTz(tmpDoFs(cDoF.s(i)),tmpDoFs(cDoF.s(j))) += lMat.Tz(i,j)*epsr;
                        }
                    }
                    for(int i=0; i<cDoF.v.n_rows; i++)
                    {
                        for(int j=0; j<cDoF.s.n_rows; j++)
                        {
                            tmpG(tmpDoFv(cDoF.v(i)),tmpDoFs(cDoF.s(j))) += lMat.G(i,j)/mur;
                        }
                    }
                }
                tmpSt = tmpSt(tmpDirDoFv,tmpDirDoFv);
                tmpTt = tmpTt(tmpDirDoFv,tmpDirDoFv);
                tmpTt2 = tmpTt2(tmpDirDoFv,tmpDirDoFv);
                tmpSz = tmpSz(tmpDirDoFs,tmpDirDoFs);
                tmpTz = tmpTz(tmpDirDoFs,tmpDirDoFs);
                tmpG = tmpG(tmpDirDoFv,tmpDirDoFs);
                if(ih == 0)
                {
                    numt = tmpSt.n_rows;
                    numz = tmpSz.n_rows;
                    numtot = numt + numz;
                    bc->ModeBeta.resize(opt->nHarm*bc->numModes);
                    bc->ModeVec.resize(numt, opt->nHarm*bc->numModes);
                    bc->ModeVecDoF.resize(numt, opt->nHarm);
                }
                arma::cx_mat tmpA(numtot,numtot);
                arma::cx_mat tmpB(numtot,numtot);
                tmpA.fill(0);
                tmpB.fill(0);
                tmpA(arma::span(0,numt-1),arma::span(0,numt-1)) = tmpSt-kk*std::pow(hCoeff[ih],2)*tmpTt;
                tmpB(arma::span(0,numt-1),arma::span(0,numt-1)) = tmpTt2;
                if(numz>0)
                {
                    tmpB(arma::span(numt,numtot-1), arma::span(numt,numtot-1)) = tmpSz-kk*std::pow(hCoeff[ih],2)*tmpTz;
                    tmpB(arma::span(0,numt-1), arma::span(numt,numtot-1)) = tmpG;
                    tmpB(arma::span(numt,numtot-1),arma::span(0,numt-1)) = tmpG.st();
                }
                tmpSt.clear();
                tmpTt.clear();
                tmpTt2.clear();
                tmpSz.clear();
                tmpTz.clear();
                tmpG.clear();
                {
                    double shift = -kk*std::pow(hCoeff[ih],2)*maxepsr*maxmur;
                    Eigen cEigen(tmpA, tmpB, numt, numz, shift, bc->numModes);
                    bc->ModeBeta.rows(ih*bc->numModes,(ih+1)*bc->numModes-1) = cEigen.modeBeta;
                    if(opt->verbose)
                    {
                        for(int i=0; i<bc->numModes; i++)
                        {
                            std::cout << bc->ModeBeta(ih*bc->numModes+i);
                        }
                    }
                    bc->ModeVec.cols(ih*bc->numModes,(ih+1)*bc->numModes-1) = cEigen.modeVec.rows(0,numt-1)*
                            arma::inv(arma::sqrt(cEigen.modeVec.st()*tmpB*cEigen.modeVec));
                    bc->ModeVecDoF.col(ih) = (sys->DoFnum*ih) + tmpEdges.elem(tmpDirDoFv);
                }
                tmpA.clear();
                tmpB.clear();
                sys->WavePortsNum += bc->numModes;
                sys->WavePortsDoFnum += bc->ModeVec.n_rows;
                sys->WavePortIds = arma::join_cols(sys->WavePortIds,bc->ModeVecDoF.col(ih));
            }
            if(opt->verbose)
            {
                std::cout << " ";
            }
            logFile << lt.toc() << " s\n";
        }
    }
    MemStat::print(logFile);
    logFile << "Finishing:\n";
    lt.tic();
    if(sys->WavePortsNum > 0)
    {
        gmm::row_matrix<std::vector<std::complex<double> > > Meig, MAcoeff;
        gmm::resize(Meig, sys->WavePortsNum, sys->WavePortsDoFnum);
        gmm::resize(MAcoeff, sys->WavePortsNum, sys->WavePortsNum);
        gmm::resize(sys->B, sys->WavePortsNum, sys->WavePortsNum);
        size_t idx = 0;
        size_t jdx = 0;
        for(size_t bcid = 0; bcid < msh->facBC.size(); bcid++)
        {
            BC* bc = &(msh->facBC[bcid]);
            if(bc->type == BC::WavePort)
            {
                for(size_t ih = 0; ih < opt->nHarm; ih++)
                {
                    for(size_t i = 0; i < bc->numModes; i++)
                    {
                        std::complex<double> sqrtBeta(std::sqrt(std::complex<double>(0.0, k0*Const::z0*opt->power)/bc->ModeBeta(ih*bc->numModes+i)));
                        MAcoeff(idx,idx) = std::complex<double>(0.0, hCoeff[ih]*k0*Const::z0*opt->power);
                        // j 2 k0 z0
                        sys->B(idx,idx) = std::complex<double>(0.0, 2*hCoeff[ih]*k0*Const::z0*opt->power);
                        for(size_t j=0; j< bc->ModeVec.n_rows; j++)
                        {
                            Meig(idx,jdx+j) = sqrtBeta * bc->ModeVec(j,ih*bc->numModes+i);
                        }
                        idx++;
                    }
                    jdx += bc->ModeVec.n_rows;
                }
            }
        }
        sys->NonWavePortIds.resize(sys->DoFnum*opt->nHarm-(sys->WavePortIds.n_rows+sys->DirDoFv.n_rows));
        idx = 0;
        arma::uvec tmp;
        for(size_t i = 0; i<sys->DoFnum*opt->nHarm; i++)
        {
            tmp = arma::find(sys->WavePortIds == i);
            tmp = arma::join_cols(tmp, arma::find(sys->DirDoFv == i));
            if(tmp.n_rows<1)
            {
                sys->NonWavePortIds(idx++) = i;
            }
        }
        tmp.clear();
        sys->DoFreal = sys->NonWavePortIds.n_rows + sys->WavePortsNum;
        std::cout << "\nSYS DoF = " << sys->DoFreal << "\n";
        logFile << "\tSYS DoF = " << sys->DoFreal << ", ";
        gmm::resize(sys->B, sys->DoFreal, 1);
        //sys->Sp.resize(sys->WavePortsNum, sys->WavePortsNum);
        lt.tic();
        std::vector<size_t> nnWPids(sys->NonWavePortIds.n_rows);
        std::vector<size_t> WPids(sys->WavePortIds.n_rows);
        #pragma omp parallel for
        for(size_t i=0; i<sys->NonWavePortIds.n_rows; i++)
        {
            nnWPids[i] = sys->NonWavePortIds(i);
        }
        #pragma omp parallel for
        for(size_t i=0; i<sys->WavePortIds.n_rows; i++)
        {
            WPids[i] = sys->WavePortIds(i);
        }
        EqSys::MatRowType Anew(sys->DoFreal, sys->DoFreal), Atired(sys->WavePortsNum,nnWPids.size());
        EqSys::MatRowType Attred1(sys->WavePortsNum,WPids.size());
        // (k0 z0 / Beta) Mt Att M
        gmm::mult(Meig, gmm::sub_matrix(sys->A, gmm::sub_index(WPids)), Attred1);
        gmm::mult(Attred1, gmm::transposed(Meig), gmm::sub_matrix(Anew, gmm::sub_interval(0, sys->WavePortsNum)));
        // Mt Att M + j k0 z0
        gmm::add(MAcoeff, gmm::sub_matrix(Anew, gmm::sub_interval(0, sys->WavePortsNum)));
        // j sqrt(k0 z0 / Beta) Mt Ati
        gmm::mult(Meig, gmm::sub_matrix(sys->A, gmm::sub_index(WPids), gmm::sub_index(nnWPids)), Atired);
        //gmm::clean(Atired, 1e-10);
        gmm::copy(Atired, gmm::sub_matrix(Anew, gmm::sub_interval(0, sys->WavePortsNum),  gmm::sub_interval(sys->WavePortsNum, nnWPids.size())));
        // j sqrt(k0 z0 / Beta) Ait M
        gmm::copy(gmm::transposed(Atired), gmm::sub_matrix(Anew, gmm::sub_interval(sys->WavePortsNum, nnWPids.size()),
                  gmm::sub_interval(0, sys->WavePortsNum)));
        // Aii
        gmm::copy(gmm::sub_matrix(sys->A, gmm::sub_index(nnWPids)), gmm::sub_matrix(Anew, gmm::sub_interval(sys->WavePortsNum, nnWPids.size())));
        //gmm::clean(Anew,1e-10);
        std::swap(sys->A, Anew);
        Anew.clear_mat();
        Atired.clear_mat();
        Meig.clear_mat();
        MAcoeff.clear_mat();
        nnWPids.clear();
        WPids.clear();
    }
    else if(opt->einc)
    {
        sys->Sol.clear();
        sys->Sol.resize(sys->DoFnum*opt->nHarm,1);
        sys->Sol.fill(0);
        arma::vec kEinc(3), polEinc(3);
        kEinc(0) = opt->k[0];
        kEinc(1) = opt->k[1];
        kEinc(2) = opt->k[2];
        polEinc(0) = opt->E[0];
        polEinc(1) = opt->E[1];
        polEinc(2) = opt->E[2];
        kEinc *= k0;
        polEinc /= std::sqrt(2.0); // Vrms
        sys->DoFreal = sys->DoFnum;
        std::cout << "\nSYS DoF = " << std::setw(8) << std::right << sys->DoFreal << "\n";
        logFile << "\tSYS DoF = " << std::setw(8) << std::right << sys->DoFreal << ", ";
        gmm::resize(sys->B, sys->DoFnum, 1);
        for(size_t bcid = 0; bcid < msh->facBC.size(); bcid++)
        {
            BC* bc = &(msh->facBC[bcid]);
            if(bc->type == BC::Radiation)
            {
                #pragma omp parallel for
                for(size_t fid = 0; fid < bc->Faces.size(); fid++)
                {
                    arma::uvec adjTet = msh->facAdjTet(bc->Faces(fid));
                    double epsr = msh->tetMtrl[msh->tetLab(adjTet(0))].epsr;
                    double mur = msh->tetMtrl[msh->tetLab(adjTet(0))].mur;
                    EleMat lMat(opt->pOrd, 2, msh->facGeo(bc->Faces(fid)), quad,
                                &(msh->tetMtrl[msh->tetLab(adjTet(0))]),
                                msh->intNode(bc->Faces(fid)), kEinc, polEinc);
                    DoF cDoF(prj, 2, bc->Faces(fid));
                    for(int i=0; i<cDoF.v.n_rows; i++)
                    {
                        #pragma omp critical
                        sys->B(cDoF.v(i),0) += lMat.f(i) * std::complex<double>(0.0,k0);
                    }
                }
            }
        }
        std::vector<size_t> DirIds(sys->DirDoFv.size());
        #pragma omp parallel for
        for(size_t i=0; i<sys->DirDoFv.size(); i++)
        {
            DirIds[i] = sys->DirDoFv(i);
        }
        gmm::clear(gmm::sub_matrix(sys->A, gmm::sub_index(DirIds), gmm::sub_interval(0, sys->DoFreal)));
        gmm::clear(gmm::sub_matrix(sys->A, gmm::sub_interval(0, sys->DoFreal), gmm::sub_index(DirIds)));
        #pragma omp parallel for
        for(size_t i=0; i<DirIds.size(); i++)
        {
            sys->A(DirIds[i],DirIds[i]) = 1.0;
        }
    }
    if(gmm::mat_euclidean_norm(sys->B) == 0)
    {
        throw std::string("Null Right Hand Side");
    }
//    EqSys::MatRowType Anew(sys->A);
//    gmm::add(gmm::transposed(gmm::scaled(sys->A, -1.0)), Anew);
//    if(gmm::mat_euclidean_norm(Anew) < 1e-9) {
//        sys->SymmFlag = 2;
//    } else {
    sys->SymmFlag = 0;
//    }
//    Anew.clear_mat();
    logFile << " " << lt.toc() << " s\n";
    logFile << "+" << tt.toc() << "s\n";
}

AssNL::~AssNL()
{
    //dtor
}
