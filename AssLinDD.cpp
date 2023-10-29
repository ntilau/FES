#include "AssLinDD.h"

#include <armadillo>
#include "Const.h"
#include "Mem.h"
#include "EqSys.h"
#include "DoF.h"
#include "EleMat.h"
#include "Eigen.h"

#include <cfloat> // for DBL_MIN

AssLinDD::AssLinDD(std::ofstream& logFile, EqSys* sys): prj(sys->prj), msh(sys->msh), opt(sys->opt), quad(sys->quad)
{
    //
    double tmpDbl;
    std::string line;
    std::ifstream fileName(std::string(opt->name + "_Ports.txt").c_str(), std::ios::in | std::ios::binary);
    if(fileName.is_open())
    {
        std::cout << "Provided port amplitudes: ";
        while(getline(fileName,line))
        {
            std::istringstream iss(line);
            iss >> tmpDbl;
            sys->PortAmpl.push_back(tmpDbl);
            std::cout << tmpDbl << " ";
        }
        std::cout << "\n";
    }
    opt->tfe = false;
    logFile << "In solids: ";
    arma::wall_clock tt, lt;
    tt.tic();
    double k0 = 2.0 * Const::pi * sys->freq / Const::c0;
    double kk = k0*k0;
    sys->DoFnum = DoF(prj).DoFnumv;
    if(opt->verbose)
    {
        std::cout << "RAW FE DoF = " << sys->DoFnum << "\n";
    }
    ///
    arma::field<arma::uvec> BndDoFv, BndDoFs, DomDoFmap, BndDoFmapJ;
    arma::field<std::vector<bool> > IntDoFv;
    IntDoFv.set_size(msh->nDomains);
    BndDoFv.set_size(msh->nDomains);
    DomDoFmap.set_size(msh->nDomains);
    BndDoFmapJ.set_size(msh->nDomains);
    lt.tic();
    for(size_t did = 0; did < msh->nDomains; did++)
    {
        std::vector<bool> DoFinternal(sys->DoFnum, true);
        arma::uvec bndFaces = msh->domFaces(did);
        for(size_t fif = 0; fif < bndFaces.n_rows; fif++)
        {
            DoF cDoF(prj, 2, bndFaces(fif));
            BndDoFv(did) = arma::unique(arma::join_cols(BndDoFv(did), cDoF.v));
            for(size_t id=0; id<cDoF.v.n_rows; id++)
            {
                DoFinternal[cDoF.v(id)] = false;
            }
        }
        IntDoFv(did) = DoFinternal;
    }
    sys->DoFlevel.clear();
    size_t gidx = 0;
    sys->DoFlevel.push_back(gidx);
    for(size_t did = 0; did < msh->nDomains; did++)
    {
        arma::uvec intTetras = msh->domTetras(did);
        arma::uvec domDoFmap(sys->DoFnum);
        arma::uvec bndJDoFmap(sys->DoFnum);
        arma::uvec bndRDoFmap(sys->DoFnum);
        domDoFmap.fill(UINT_MAX);
        bndJDoFmap.fill(UINT_MAX);
        bndRDoFmap.fill(UINT_MAX);
        std::vector<bool> DoFinternal = IntDoFv(did);
        std::cout << "Domain " << did << ": " << gidx << "-";
        for(size_t tit = 0; tit < intTetras.n_rows; tit++)
        {
            DoF cDoF(prj, 3, intTetras(tit));
            for(size_t id=0; id<cDoF.v.n_rows; id++)
            {
                if(DoFinternal[cDoF.v(id)])
                {
                    domDoFmap(cDoF.v(id)) = gidx++;
                    DoFinternal[cDoF.v(id)] = false;
                }
            }
        }
        std::cout << gidx-1 << ", ";
        std::cout << gidx << "-";
        arma::uvec domBndDoFv = BndDoFv(did);
        for(size_t id = 0; id < domBndDoFv.n_rows; id++)
        {
            domDoFmap(domBndDoFv(id)) = gidx++;
        }
        DomDoFmap(did) = domDoFmap;
        std::cout << gidx-1;
        if(!opt->ddn)
        {
            std::cout << ", " << gidx << "-";
            for(size_t id = 0; id < domBndDoFv.n_rows; id++)
            {
                bndJDoFmap(domBndDoFv(id)) = gidx++;
            }
            BndDoFmapJ(did) = bndJDoFmap;
            std::cout << gidx-1;
        }
        sys->DoFlevel.push_back(gidx);
        std::cout << "\n";
    }
    size_t DoFnum = gidx;
    sys->InvDoFmapv.resize(DoFnum);
    sys->InvDoFmapv.fill(UINT_MAX);
    sys->DoFmapv.resize(DoFnum);
    sys->DoFmapv.fill(UINT_MAX);
    std::vector<bool> mapped(sys->DoFnum, true);
    for(size_t did = 0; did < msh->nDomains; did++)
    {
        arma::uvec domDoFmap = DomDoFmap(did);
        for(size_t i = 0; i < sys->DoFnum; i++)
        {
            if(mapped[i])
            {
                if(domDoFmap(i) < UINT_MAX)
                {
                    sys->InvDoFmapv(domDoFmap(i)) = i;
                    sys->DoFmapv(i) = domDoFmap(i);
                    mapped[i] = false;
                }
            }
        }
    }
    ///
    std::vector<bool> DoFtoLeave(DoFnum, true);
    if(opt->verbose)
    {
        std::cout << "FE DoF = " << DoFnum << " ";
    }
    sys->SymmFlag = 0;
    sys->A.clear_mat();
    sys->PR.clear_mat();
    sys->B.clear_mat();
    sys->Sol.clear();
    sys->Sp.clear();
    gmm::resize(sys->A, DoFnum, DoFnum);
    gmm::resize(sys->PR, DoFnum, DoFnum);
    lt.tic();
    #pragma omp parallel for
    for(size_t id = 0; id < msh->nTetras; id++)
    {
        Mtrl* cMtrl = &(msh->tetMtrl[msh->tetLab(id)]);
        EleMat lMat(opt->pOrd, 3, msh->tetGeo(id), quad, cMtrl, Shape::Hcurl);
        DoF cDoF(prj, 3, id);
        cDoF.v = DomDoFmap(msh->tetDom(id)).elem(cDoF.v);
        #pragma omp critical
        for(int i=0; i<cDoF.v.n_rows; i++)
        {
            for(int j=0; j<cDoF.v.n_rows; j++)
            {
                if(cDoF.v(i)<=cDoF.v(j))
                {
                    sys->A(cDoF.v(i),cDoF.v(j)) += lMat.S(i,j) + k0*lMat.Z(i,j) - kk*lMat.T(i,j);
                }
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
        switch(bc->type)
        {
        case BC::PerfectE :
            lt.tic();
            if(opt->verbose)
            {
                std::cout << bc->name;
            }
            logFile << "\t" << bc->name << ": ";
            #pragma omp parallel for
            for(size_t fid = 0; fid < bc->Faces.size(); fid++)
            {
                DoF cDoF(prj, 2, bc->Faces(fid));
                arma::uvec adjTet = msh->facAdjTet(bc->Faces(fid));
                cDoF.v = DomDoFmap(msh->tetDom(adjTet(0))).elem(cDoF.v);
                #pragma omp critical
                {
                    sys->DirDoFs = arma::join_cols(sys->DirDoFs, cDoF.s);
                    sys->DirDoFv = arma::join_cols(sys->DirDoFv, cDoF.v);
                    for(size_t dofid=0; dofid < cDoF.v.n_rows; dofid++)
                    {
                        DoFtoLeave[cDoF.v(dofid)] = false;
                    }
                }
            }
            sys->DirDoFs = arma::unique(sys->DirDoFs);
            sys->DirDoFv = arma::unique(sys->DirDoFv);
            logFile << lt.toc() << " s\n";
            if(opt->verbose)
            {
                std::cout << " ";
            }
            break;
        case BC::WavePort:
            break;
        case BC::Radiation:
            lt.tic();
            if(opt->verbose)
            {
                std::cout << bc->name;
            }
            logFile << "\t" << bc->name << ": ";
            #pragma omp parallel for
            for(size_t fid = 0; fid < bc->Faces.size(); fid++)
            {
                arma::uvec adjTet = msh->facAdjTet(bc->Faces(fid));
                Mtrl* cMtrl = &(msh->tetMtrl[msh->tetLab(adjTet(0))]);
                std::complex<double> epsr(cMtrl->epsr, cMtrl->CalcEpsr2(sys->freq));
                double mur = cMtrl->mur;
                EleMat lMat(opt->pOrd, 2, msh->facGeo(bc->Faces(fid)), quad,
                            cMtrl, msh->intNode(bc->Faces(fid)));
                DoF cDoF(prj, 2, bc->Faces(fid));
                cDoF.v = DomDoFmap(msh->tetDom(adjTet(0))).elem(cDoF.v);
                #pragma omp critical
                for(int i=0; i<cDoF.v.n_rows; i++)
                {
                    for(int j=0; j<cDoF.v.n_rows; j++)
                    {
                        if(cDoF.v(i)<=cDoF.v(j))
                        {
                            sys->A(cDoF.v(i),cDoF.v(j)) += std::complex<double>(0.0,k0)*lMat.Tt(i,j)*std::sqrt(epsr/mur);
                        }
                    }
                }
            }
            logFile << lt.toc() << " s\n";
            if(opt->verbose)
            {
                std::cout << " ";
            }
            break;
        case BC::PerfectH:
            lt.tic();
            if(opt->verbose)
            {
                std::cout << bc->name;
            }
            logFile << "\t" << bc->name << ": ";
            logFile << lt.toc() << " s\n";
            if(opt->verbose)
            {
                std::cout << " ";
            }
            break;
        default:
            throw std::string("Wrong boundary type");
        }
    }
    sys->WavePortsNum = 0;
    sys->WavePortsDoFnum = 0;
    sys->WavePortIds.reset();
    size_t idx = 0;
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
            arma::uvec tmpShared, tmpSharedDD, tmpSharedReg;
            tmpDoFs = arma::zeros<arma::uvec>(DoF(prj).DoFnums);
            tmpDoFv = arma::zeros<arma::uvec>(DoFnum);
            for(size_t fid = 0; fid < bc->Faces.size(); fid++)
            {
                DoF cDoF(prj, 2, bc->Faces(fid));
                tmpShared = arma::join_cols(tmpShared, cDoF.v);
                arma::uvec adjTet = msh->facAdjTet(bc->Faces(fid));
                cDoF.v = DomDoFmap(msh->tetDom(adjTet(0))).elem(cDoF.v);
                tmpSharedDD = arma::join_cols(tmpSharedDD, cDoF.v);
                tmpSharedReg = arma::join_cols(tmpSharedReg, 0*cDoF.v + msh->tetDom(adjTet(0)));
                tmpEdges = arma::join_cols(tmpEdges, cDoF.v);
                tmpNodes = arma::join_cols(tmpNodes, cDoF.s);
            }
            arma::uvec sharedRegs = arma::unique(tmpSharedReg);
            arma::umat sharedDoFv;
            for(size_t i = 0; i< tmpShared.size(); i++)
            {
                arma::uvec dofs = arma::sort(arma::find(tmpShared == tmpShared(i)));
                if(dofs.size() == 2)
                    if(tmpSharedReg(dofs(0)) != tmpSharedReg(dofs(1)))
                    {
                        arma::uvec DoFv = arma::sort(tmpSharedDD.elem(dofs));
                        if(sharedDoFv.n_rows > 0)
                        {
                            tmp = arma::find(sharedDoFv.col(0) == DoFv(0));
                            if(tmp.size() == 0)
                            {
                                sharedDoFv = arma::join_cols(sharedDoFv, DoFv.st());
                            }
                        }
                        else
                        {
                            sharedDoFv = arma::join_cols(sharedDoFv, DoFv.st());
                        }
                    }
            }
            for(size_t i=0; i< sharedDoFv.n_rows; i++)
            {
                tmp = arma::find(tmpEdges == sharedDoFv(i,1));
                if(tmp.size()>0)
                {
                    for(size_t j=0; j<tmp.n_rows; j++)
                    {
                        tmpEdges(tmp(j)) = arma::min(sharedDoFv.row(i));
                    }
                }
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
                std::complex<double> epsr(cMtrl->epsr, cMtrl->CalcEpsr2(sys->freq));
                double mur = cMtrl->mur;
                maxepsr = std::max(maxepsr, std::real(epsr));
                maxmur = std::max(maxmur, mur);
                EleMat lMat(opt->pOrd, 2, msh->facGeo(bc->Faces(fid)), quad,
                            cMtrl, msh->intNode(bc->Faces(fid)));
                DoF cDoF(prj, 2, bc->Faces(fid));
                cDoF.v = DomDoFmap(msh->tetDom(adjTet(0))).elem(cDoF.v);
                // fixing for shared DoFs
                for(size_t i=0; i< sharedDoFv.n_rows; i++)
                {
                    tmp = arma::find(cDoF.v  == sharedDoFv(i,1));
                    if(tmp.size()>0)
                        for(size_t j=0; j<tmp.n_rows; j++)
                        {
                            cDoF.v(tmp(j)) = sharedDoFv(i,0);
                        }
                }
                // assembly
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
            size_t numt = tmpSt.n_rows;
            size_t numz = tmpSz.n_rows;
            size_t numtot = numt + numz;
            arma::cx_mat tmpA(numtot,numtot);
            arma::cx_mat tmpB(numtot,numtot);
            tmpA.fill(0);
            tmpB.fill(0);
            tmpA(arma::span(0,numt-1),arma::span(0,numt-1)) = tmpSt-kk*tmpTt;
            tmpB(arma::span(0,numt-1),arma::span(0,numt-1)) = tmpTt2;
            if(numz>0)
            {
                tmpB(arma::span(numt,numtot-1), arma::span(numt,numtot-1)) = tmpSz-kk*tmpTz;
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
                double shift = -kk*maxepsr*maxmur;
                Eigen cEigen(tmpA, tmpB, numt, numz, shift, bc->numModes);
                bc->ModeBeta = cEigen.modeBeta;
                if(opt->verbose)
                {
                    for(int i=0; i<bc->numModes; i++)
                    {
                        std::cout << bc->ModeBeta(i);
                    }
                }
                bc->ModeVec = cEigen.modeVec.rows(0,numt-1)*arma::inv(arma::sqrt(cEigen.modeVec.st()*tmpB*cEigen.modeVec));
                arma::cx_mat coeff(bc->ModeBeta.size(),bc->ModeBeta.size());
                coeff.diag() = arma::sqrt(std::complex<double>(0.0, k0*Const::z0)/bc->ModeBeta);
                if(sys->PortAmpl.size() > 0)
                {
                    bc->ModeVecf = sys->PortAmpl[idx++] * (tmpB * (cEigen.modeVec * coeff *
                                                           arma::inv(arma::sqrt(cEigen.modeVec.st()*tmpB*cEigen.modeVec))));
                }
                else
                {
                    bc->ModeVecf = (tmpB * (cEigen.modeVec * coeff *
                                            arma::inv(arma::sqrt(cEigen.modeVec.st()*tmpB*cEigen.modeVec))));
                }
                bc->ModeVecf = bc->ModeVecf.rows(0,numt-1);
                bc->ModeVecDoF = tmpEdges.elem(tmpDirDoFv);
            }
            tmpA.clear();
            tmpB.clear();
            sys->WavePortsNum += bc->numModes;
            sys->WavePortsDoFnum += bc->ModeVec.n_rows;
            sys->WavePortIds = arma::join_cols(sys->WavePortIds,bc->ModeVecDoF);
            if(opt->verbose)
            {
                std::cout << " ";
            }
            logFile << lt.toc() << " s\n";
        }
    }
    size_t reg1, reg2;
    std::complex<double> epsrd1, epsrd2, murd1, murd2;
    for(size_t did = 0; did < msh->nDomains; did++)
    {
        arma::uvec bndFaces = msh->domFaces(did);
        for(size_t fif = 0; fif < bndFaces.n_rows; fif++)
        {
            arma::uvec adjTet = msh->facAdjTet(bndFaces(fif));
            Mtrl* cMtrl0 = &(msh->tetMtrl[msh->tetLab(adjTet(0))]);
            Mtrl* cMtrl1 = &(msh->tetMtrl[msh->tetLab(adjTet(1))]);
            EleMat lMat(opt->pOrd, 2, msh->facGeo(bndFaces(fif)), quad,
                        cMtrl0, msh->intNode(bndFaces(fif)));
            reg1 = did;
            if(msh->tetDom(adjTet(0)) == reg1)
            {
                epsrd1 = std::complex<double>(cMtrl0->epsr, cMtrl0->CalcEpsr2(sys->freq));
                murd1 = std::complex<double>(cMtrl0->mur,0.0);
                epsrd2 = std::complex<double>(cMtrl1->epsr, cMtrl1->CalcEpsr2(sys->freq));
                murd2 = std::complex<double>(cMtrl1->mur,0.0);
                reg2 = msh->tetDom(adjTet(1));
            }
            else
            {
                epsrd2 = std::complex<double>(cMtrl0->epsr, cMtrl0->CalcEpsr2(sys->freq));
                murd2 = std::complex<double>(cMtrl0->mur,0.0);
                epsrd1 = std::complex<double>(cMtrl1->epsr, cMtrl1->CalcEpsr2(sys->freq));
                murd1 = std::complex<double>(cMtrl1->mur,0.0);
                reg2 = msh->tetDom(adjTet(0));
            }
            DoF cDoF1(prj, 2, bndFaces(fif));
            cDoF1.v = DomDoFmap(reg1).elem(cDoF1.v);
            DoF cDoF2(prj, 2, bndFaces(fif));
            cDoF2.v = DomDoFmap(reg2).elem(cDoF2.v);
            if(opt->ddn)
            {
                Jacobian* cJac30 = new Jacobian(3, prj->msh->tetGeo(adjTet(0)));
                Jacobian* cJac31 = new Jacobian(3, prj->msh->tetGeo(adjTet(1)));
                arma::mat cGeo = prj->msh->facGeo(bndFaces(fif));
                size_t RefFace = 0;
                arma::mat intNode = prj->msh->intNode(bndFaces(fif), RefFace);
                //std::cout << RefFace << "\n";
                arma::mat RefGeo(3,3);
                RefGeo.fill(0);
                arma::uvec map3to2(3); //first order
                switch(RefFace)
                {
                case 0:
                    RefGeo(0,0) = 1.0;
                    RefGeo(1,0) = -1.0;
                    RefGeo(1,1) = 1.0;
                    RefGeo(2,0) = -1.0;
                    RefGeo(2,2) = 1.0;
                    map3to2(0) = 5;
                    map3to2(1) = 4;
                    map3to2(2) = 3;
                    break;
                case 1:
                    RefGeo(1,1) = 1.0;
                    RefGeo(2,2) = 1.0;
                    map3to2(0) = 5;
                    map3to2(1) = 2;
                    map3to2(2) = 1;
                    break;
                case 2:
                    RefGeo(1,0) = 1.0;
                    RefGeo(2,2) = 1.0;
                    map3to2(0) = 4;
                    map3to2(1) = 2;
                    map3to2(2) = 0;
                    break;
                case 3:
                    RefGeo(1,0) = 1.0;
                    RefGeo(2,1) = 1.0;
                    map3to2(0) = 3;
                    map3to2(1) = 1;
                    map3to2(2) = 0;
                    break;
                }
                arma::vec v0 = cGeo.row(0).st();
                arma::vec v1 = cGeo.row(1).st();
                arma::vec v2 = cGeo.row(2).st();
                arma::vec vIn = ((v0+v1+v2)/3.0) - intNode;
                vIn /= arma::norm(vIn,2);
                v1 -= v0;
                v2 -= v0;
                arma::vec u = v1 / arma::norm(v1,2);
                arma::vec n = arma::cross(v1,v2);
                n *= arma::dot(n,vIn);
                n /= arma::norm(n,2);
                arma::vec v = arma::cross(n,u);
                arma::mat cGeo2(3,2);
                cGeo2.fill(0);
                cGeo2(1,0) = arma::dot(v1, u);
                cGeo2(2,0) = arma::dot(v2, u);
                cGeo2(2,1) = arma::dot(v2, v);
                Jacobian* cJac2 = new Jacobian(2, cGeo2);
                arma::cx_mat Dij, Dji, S, T;
                switch(opt->pOrd)
                {
                case 1:
                    Dij.resize(6,6);
                    Dji.resize(6,6);
                    S.resize(6,6);
                    T.resize(6,6);
                    break;
                case 2:
                    Dij.resize(20,20);
                    S.resize(6,6);
                    break;
                case 3:
                    Dij.resize(45,45);
                    S.resize(6,6);
                    break;
                default:
                    throw std::string("2D EleMat order not yet implemented");
                }
                for(size_t iq=0; iq< quad->wq2.n_rows; iq++)
                {
                    arma::rowvec LocPnt =  RefGeo.row(0) +
                                           RefGeo.row(1)*quad->xq2(iq,0) +
                                           RefGeo.row(2)*quad->xq2(iq,1);
                    //arma::vec GlobPnt =  v0 + v1*quad->xq2(iq,0) + v2*quad->xq2(iq,1);
                    Shape shp0(opt->pOrd, 3, Shape::Hcurl, LocPnt, cJac30);
                    Shape shp1(opt->pOrd, 3, Shape::Hcurl, LocPnt, cJac31);
                    //arma::mat dshape(2,6);
                    for(size_t i=0; i < shp0.Nv.n_cols; i++)
                    {
                        //shape(0,i) = arma::dot(n, shp.dNv.col(i));
                        shp0.Nv.col(i) = arma::cross(arma::cross(n, shp0.Nv.col(i)),n);
                        shp1.Nv.col(i) = arma::cross(arma::cross(n, shp1.Nv.col(i)),n);
                        shp0.dNv.col(i) = -n% shp0.dNv.col(i);
                        shp1.dNv.col(i) = -n% shp1.dNv.col(i);
                    }
//                    S += arma::conv_to<arma::cx_mat>::from(shp.dNv.t()*shp.dNv)*quad->wq2(iq);
                    //Dij += arma::conv_to<arma::cx_mat>::from(shp.Nv.t()*shp.dNv)*quad->wq2(iq);
//                    Dji += arma::conv_to<arma::cx_mat>::from(shp.dNv.t()*shp.Nv)*quad->wq2(iq);
//                    T += arma::conv_to<arma::cx_mat>::from(shp.Nv.t()*shp.Nv)*quad->wq2(iq);
//                    Dij += arma::conv_to<arma::cx_mat>::from(shp.dNv.t()*shp.Nv + shp.Nv.t()*shp.dNv)*quad->wq2(iq);
                    Dij += arma::conv_to<arma::cx_mat>::from(shp1.dNv.t()*shp0.Nv + shp0.Nv.t()*shp1.dNv)*quad->wq2(iq);
                }
                Dij *= cJac2->detJ;
//                Dji *= cJac2->detJ;
//                S *= cJac2->detJ;
//                T *= cJac2->detJ;
                Dij = /*arma::diagmat*/(Dij.submat(map3to2,map3to2));
//                Dji = Dji.submat(map3to2,map3to2);
//                S = S.submat(map3to2,map3to2);
//                T = T.submat(map3to2,map3to2);
                std::cout << "Dij:\n" << arma::real(Dij) ;
                //std::cout << "ST:\n" << arma::real(D) << "\nT2:\n" << arma::real(lMat.St) << "\n";
                for(int i=0; i<cDoF1.v.n_rows; i++)
                {
                    for(int j=0; j<cDoF1.v.n_rows; j++)
                    {
                        if(cDoF1.v(i)<=cDoF1.v(j))
                        {
                            sys->A(cDoF1.v(i),cDoF1.v(j)) += lMat.Tt(i,j) * std::complex<double>(0.0, k0);// +
                            //Dij(i,j) + Dji(j,i) + S(i,j) *std::complex<double>(0.0, 1.0/(k0));
                        }
                        sys->PR(cDoF2.v(i),cDoF1.v(j)) -= lMat.Tt(i,j) * std::complex<double>(0.0, k0) + Dij(i,j);// + Dji(i,j) - S(i,j);
                        // *std::complex<double>(0.0, 1.0/(k0));
                    }
                }
            }
            else
            {
                DoF cDoF1j(prj, 2, bndFaces(fif));
                cDoF1j.v = BndDoFmapJ(reg1).elem(cDoF1j.v);
                DoF cDoF2j(prj, 2, bndFaces(fif));
                cDoF2j.v = BndDoFmapJ(reg2).elem(cDoF2j.v);
                for(int i=0; i<cDoF1.v.n_rows; i++)
                {
                    for(int j=0; j<cDoF1.v.n_rows; j++)
                    {
                        if(cDoF1.v(i)<=cDoF1j.v(j))
                        {
                            sys->A(cDoF1.v(i),cDoF1j.v(j)) += lMat.Tt(i,j) * k0;// * std::sqrt(murd1*epsrd1);
                        }
                        if(cDoF1j.v(i)<=cDoF1.v(j))
                        {
                            sys->A(cDoF1j.v(i),cDoF1.v(j)) += lMat.Tt(j,i) * k0;// * std::sqrt(murd1*epsrd1);
                        }
                        if(cDoF1j.v(i)<=cDoF1j.v(j))
                        {
                            sys->A(cDoF1j.v(i),cDoF1j.v(j)) += lMat.Tt(i,j) * std::complex<double>(0.0, k0);
                        }
                        sys->PR(cDoF1j.v(i),cDoF2.v(j)) -= lMat.Tt(i,j) * k0;
                        sys->PR(cDoF1j.v(i),cDoF2j.v(j)) += lMat.Tt(i,j) * std::complex<double>(0.0, k0);
                    }
                }
            }
        }
    }
    MemStat::print(logFile);
    logFile << "Finishing:\n";
    lt.tic();
    if(sys->WavePortsNum > 0)
    {
        sys->DoFreal = DoFnum - sys->DirDoFv.n_rows;
        std::cout << "\nSYS DoF = " << sys->DoFreal << "\n";
        logFile << "\tSYS DoF = " << sys->DoFreal << ", ";
        gmm::resize(sys->B, DoFnum, sys->WavePortsNum);
        size_t idx = 0;
        for(size_t bcid = 0; bcid < msh->facBC.size(); bcid++)
        {
            BC* bc = &(msh->facBC[bcid]);
            if(bc->type == BC::WavePort)
            {
                std::complex<double> coeff = bc->ModeBeta(0);
                #pragma omp parallel for
                for(size_t fid = 0; fid < bc->Faces.size(); fid++)
                {
                    arma::uvec adjTet = msh->facAdjTet(bc->Faces(fid));
                    Mtrl* cMtrl = &(msh->tetMtrl[msh->tetLab(adjTet(0))]);
                    std::complex<double> epsr(cMtrl->epsr, cMtrl->CalcEpsr2(sys->freq));
                    double mur = cMtrl->mur;
                    EleMat lMat(opt->pOrd, 2, msh->facGeo(bc->Faces(fid)), quad,
                                cMtrl, msh->intNode(bc->Faces(fid)));
                    DoF cDoF(prj, 2, bc->Faces(fid));
                    cDoF.v = DomDoFmap(msh->tetDom(adjTet(0))).elem(cDoF.v);
                    #pragma omp critical
                    for(int i=0; i<cDoF.v.n_rows; i++)
                    {
                        for(int j=0; j<cDoF.v.n_rows; j++)
                        {
                            if(cDoF.v(i)<=cDoF.v(j))
                            {
                                sys->A(cDoF.v(i),cDoF.v(j)) += lMat.Tt(i,j)*coeff;
                            }
                        }
                    }
                }
                for(size_t in = 0; in < bc->ModeBeta.size(); in++)
                {
                    std::complex<double> coeff = bc->ModeBeta(in);
                    for(size_t i=0; i<bc->ModeVecDoF.size(); i++)
                    {
                        sys->B(bc->ModeVecDoF(i),idx) = 2.0 * coeff * bc->ModeVecf(i,in);
                    }
                    idx++;
                }
                for(size_t i=0; i<bc->ModeVecDoF.size(); i++)
                {
                    bc->ModeVecDoF(i) = sys->InvDoFmapv(bc->ModeVecDoF(i));
                }
            }
        }
        sys->NonDirIds.resize(sys->DoFreal);
        idx = 0;
        for(size_t i = 0; i<DoFnum; i++)
        {
            if(DoFtoLeave[i] == true)
            {
                sys->NonDirIds[idx++] = i;
            }
        }
        EqSys::MatRowType Anew(sys->DoFreal, sys->DoFreal), PRnew(sys->DoFreal, sys->DoFreal);
        EqSys::MatColType Bnew(sys->DoFreal, sys->WavePortsNum);
        gmm::copy(gmm::sub_matrix(sys->A, gmm::sub_index(sys->NonDirIds)), Anew);
        gmm::copy(gmm::sub_matrix(sys->PR, gmm::sub_index(sys->NonDirIds)), PRnew);
        gmm::copy(gmm::sub_matrix(sys->B, gmm::sub_index(sys->NonDirIds), gmm::sub_interval(0,sys->WavePortsNum)), Bnew);
        std::swap(sys->A, Anew);
        std::swap(sys->B, Bnew);
        std::swap(sys->PR, PRnew);
        Anew.clear_mat();
        Bnew.clear_mat();
        PRnew.clear_mat();
        idx = 0;
        for(size_t i = 0; i<sys->DoFreal; i++)
        {
            sys->NonDirIds[i] =  sys->InvDoFmapv(sys->NonDirIds[i]);
        }
        std::vector<size_t> newDoFlevel;
        idx = 0;
        newDoFlevel.push_back(idx);
        //idx = sys->WavePortsNum;
        for(size_t irg=1; irg< sys->DoFlevel.size(); irg++)
        {
            for(size_t rgid=sys->DoFlevel[irg-1]; rgid<sys->DoFlevel[irg]; rgid++)
            {
                if(DoFtoLeave[rgid] == true)
                {
                    idx++;
                }
            }
            newDoFlevel.push_back(idx);
        }
        sys->DoFlevel = newDoFlevel;
        newDoFlevel.clear();
    }
    else if(opt->einc)
    {
        arma::vec kEinc(3), polEinc(3);
        kEinc(0) = opt->k[0];
        kEinc(1) = opt->k[1];
        kEinc(2) = opt->k[2];
        polEinc(0) = opt->E[0];
        polEinc(1) = opt->E[1];
        polEinc(2) = opt->E[2];
        kEinc *= k0;
        polEinc /= std::sqrt(2.0); // Vrms
        sys->DoFreal = DoFnum;
        if(opt->verbose)
        {
            std::cout << "\nSYS DoF = " << sys->DoFreal << "\n";
        }
        logFile << "\tSYS DoF = " << sys->DoFreal << ", ";
        gmm::resize(sys->B, DoFnum, 1);
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
                    cDoF.v = DomDoFmap(msh->tetDom(adjTet(0))).elem(cDoF.v);
                    #pragma omp critical
                    for(int i=0; i<cDoF.v.n_rows; i++)
                    {
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
    logFile << " " << lt.toc() << " s\n";
    logFile << "+" << tt.toc() << " s\n";
}

AssLinDD::~AssLinDD()
{
    //dtor
}
