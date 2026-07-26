#include "assembler.h"
#include <stdexcept>

#include <armadillo>
#include "constants.h"
#include "memory.h"
#include "equation_system.h"
#include "degree_of_freedom.h"
#include "element_matrix.h"
#include "eigen_solver.h"
#include "mesh.h"
#include "option.h"

#include <cfloat>
#include <unordered_map>
#include <complex>

void assembler_em_e_fd_schur::assemble(std::ofstream& logFile, eq_sys& sys)
{
    mesh* msh = sys.msh;
    option* opt = sys.opt;
    project* prj = sys.prj;
    quad* quadr = sys.quadr;
//    double phase[] = {1.0,1.0,1.0,1.0,-1.0,
//                      1.0,-1.0,1.0,1.0,1.0,
//                      1.0,1.0,-1.0,-1.0,1.0,
//                      1.0,1.0,-1.0,1.0,-1.0,
//                      1.0,1.0,-1.0,1.0,-1.0
//                     };
//    size_t countPort = 0;
    logFile << "% Assembly Symmetric:\n";
    logFile << "In solids: ";
    arma::wall_clock tt, lt;
    tt.tic();
    double k0 = 2.0 * consts::pi * sys.get_freq() / consts::c0;
    double kk = k0*k0;
    sys.set_dofnum(dof(prj).dofnumv);
        std::cout << "RAW FE dof = " << sys.get_dofnum() << "\n";
    std::vector<bool> doftoLeave(sys.get_dofnum(), true);
    /// Computing raw dof mapping
    sys.dofmapv_vec().resize(sys.get_dofnum());
    sys.Invdofmapv_vec().resize(sys.get_dofnum());
    arma::uvec Bnddof;
    //arma::field<arma::uvec> Intdof;
    arma::field<std::vector<bool> > IsIntdof;
    //Intdof.set_size(msh->nDomains);
    IsIntdof.set_size(msh->nDomains);
    for(size_t did = 0; did < msh->nDomains; did++)
    {
        std::vector<bool> dofinternal(sys.get_dofnum(), true);
        arma::uvec bndFaces = msh->domFaces(did);
        for(size_t fif = 0; fif < bndFaces.n_rows; fif++)
        {
            dof cdof(prj, 2, bndFaces(fif));
            Bnddof = arma::unique(arma::join_cols(Bnddof, cdof.v));
            for(size_t id=0; id<cdof.v.n_rows; id++)
            {
                dofinternal[cdof.v(id)] = false;
            }
        }
        IsIntdof(did) = dofinternal;
    }
    size_t gidx = 0;
    sys.doflevel_vec().push_back(gidx);
    for(size_t bib = 0; bib < Bnddof.n_rows; bib++)
    {
        sys.dofmapv_vec()(Bnddof(bib)) = gidx++;
    }
    sys.doflevel_vec().push_back(gidx);
    //std::cout << "Bnds: 0-" << gidx-1 << "\n";
    for(size_t did = 0; did < msh->nDomains; did++)
    {
        arma::uvec intTetras = msh->domTetras(did);
        std::vector<bool> dofinternal = IsIntdof(did);
        //std::cout << "Domain " << did << ": " << gidx << "-";
        for(size_t tit = 0; tit < intTetras.n_rows; tit++)
        {
            dof cdof(prj, 3, intTetras(tit));
            //Intdof(did) = arma::unique(arma::join_cols(Intdof(did), cdof.v));
            for(size_t id=0; id<cdof.v.n_rows; id++)
            {
                if(dofinternal[cdof.v(id)])
                {
                    sys.dofmapv_vec()(cdof.v(id)) = gidx++;
                    dofinternal[cdof.v(id)] = false;
                }
            }
        }
        //std::cout << gidx-1 << "\n";
        sys.doflevel_vec().push_back(gidx);
    }
    for(size_t dofid=0; dofid<sys.get_dofnum(); dofid++)
    {
        sys.Invdofmapv_vec()(sys.dofmapv_vec()(dofid)) = dofid;
    }
        std::cout << "FE dof = " << sys.get_dofnum() << " ";
    sys.set_symm_flag(0);
    sys.A_mat().zeros();
    sys.B_mat().zeros();
    sys.Sol_mat().clear();
    sys.Sp_mat().clear();
    sys.A_mat().set_size(sys.get_dofnum(), sys.get_dofnum());
    lt.tic();
    // prebuild AFF
    sys.AFF_vec().resize(msh->nDomains);
    for(size_t did = 0; did < msh->nDomains; did++)
    {
        sys.AFF_vec()[did].set_size(sys.doflevel_vec()[1], sys.doflevel_vec()[1]);
        //std::cout << sys.AFF_vec()[did] << "\n";
    }
    for(size_t id = 0; id < msh->nTetras; id++)
    {
        mtrl* cmtrl = &(msh->tetmtrl[msh->tetLab(id)]);
        ele_mat lMat(opt->p_ord, 3, msh->tet_geo(id), quadr, cmtrl, shape::hcurl);
        dof cdof(prj, 3, id);
        cdof.v = sys.dofmapv_vec().elem(cdof.v);
        for(int i=0; i<cdof.v.n_rows; i++)
        {
            for(int j=0; j<cdof.v.n_rows; j++)
            {
                if(cdof.v(i)<=cdof.v(j))
                {
                    sys.A_mat()(cdof.v(i),cdof.v(j)) += lMat.S(i,j) + k0*lMat.Z(i,j) - kk*lMat.T(i,j);
                    if((cdof.v(i)<sys.doflevel_vec()[1]) && (cdof.v(j) < sys.doflevel_vec()[1]))
                    {
                        sys.AFF_vec()[msh->tetDom(id)](cdof.v(i),cdof.v(j)) += lMat.S(i,j) + k0*lMat.Z(i,j) - kk*lMat.T(i,j);
                    }
                }
            }
        }
    }
//    eq_sys::mat_row_type Test(sys.doflevel_vec()[1], sys.doflevel_vec()[1]);
//    for(size_t did = 0; did < msh->nDomains; did++) {
//        gmm::add(sys.AFF_vec()[did], Test);
//        //std::cout << sys.AFF_vec()[did] << "\n";
//    }
//    std::cout << Test;
//    gmm::add((gmm::sub_interval(0, sys.doflevel_vec()[1]) * (gmm::sub_matrix(sys.A_mat())),-1.0), Test);
//    std::cout << Test; exit(0);
//    for(size_t did = 0; did < msh->nDomains; did++) {
//        //sys.AFF_vec()[did].set_size(ys->doflevel[1],  sys.doflevel_vec()[1]);
//        std::cout << sys.AFF_vec()[did] << "\n";
//    }
//    for(size_t did = 0; did < msh->nDomains; did++) {
//        arma::uvec ddFaces = msh->domFaces(did);
//        std::cout << "Domain " << did << ": " << ddFaces.n_rows << "-\n";
//        for(size_t dif = 0; dif < ddFaces.n_rows; dif++) {
//            std::cout << ddFaces(dif) << "\n";
//        // get
////            dof cdof(prj, 3, intTetras(tit));
////            //Intdof(did) = arma::unique(arma::join_cols(Intdof(did), cdof.v));
////            for(size_t id=0; id<cdof.v.n_rows; id++) {
////                if(dofinternal[cdof.v(id)]) {
////                    sys.dofmapv_vec()(cdof.v(id)) = gidx++;
////                    dofinternal[cdof.v(id)] = false;
////                }
////            }
//        }
//        //std::cout << gidx-1 << "\n";
//    }
//
    logFile << lt.toc() << " s\n";
        std::cout << lt.toc() << " s\n";
    mem_stat::print(logFile);
        mem_stat::print(std::cout);
    logFile << "On boundaries:\n";
    for(size_t bcid = 0; bcid < msh->facbc.size(); bcid++)
    {
        bc* bc = &(msh->facbc[bcid]);
        switch(bc->type)
        {
        case bc::perfect_e :
            lt.tic();
                std::cout << bc->name;
            logFile << "\t" << bc->name << ": ";
            for(size_t fid = 0; fid < bc->Faces.size(); fid++)
            {
                dof cdof(prj, 2, bc->Faces(fid));
                cdof.v = sys.dofmapv_vec().elem(cdof.v);
                {
                    sys.Dirdofs_vec() = arma::join_cols(sys.Dirdofs_vec(), cdof.s);
                    sys.Dirdofv_vec() = arma::join_cols(sys.Dirdofv_vec(), cdof.v);
                    for(size_t dofid=0; dofid < cdof.v.n_rows; dofid++)
                    {
                        doftoLeave[cdof.v(dofid)] = false;
                    }
                }
            }
            sys.Dirdofs_vec() = arma::unique(sys.Dirdofs_vec());
            sys.Dirdofv_vec() = arma::unique(sys.Dirdofv_vec());
            logFile << lt.toc() << " s\n";
                std::cout << " ";
            break;
        case bc::wave_port:
            break;
        case bc::radiation:
            lt.tic();
                std::cout << bc->name;
            logFile << "\t" << bc->name << ": ";
            for(size_t fid = 0; fid < bc->Faces.size(); fid++)
            {
                arma::uvec adjTet = msh->facAdjTet(bc->Faces(fid));
                mtrl* cmtrl = &(msh->tetmtrl[msh->tetLab(adjTet(0))]);
                std::complex<double> epsr(cmtrl->epsr, cmtrl->calc_epsr2(sys.get_freq()));
                double mur = cmtrl->mur;
                ele_mat lMat(opt->p_ord, 2, msh->fac_geo(bc->Faces(fid)), quadr,
                            cmtrl, msh->int_node(bc->Faces(fid)));
                dof cdof(prj, 2, bc->Faces(fid));
                cdof.v = sys.dofmapv_vec().elem(cdof.v);
                for(int i=0; i<cdof.v.n_rows; i++)
                {
                    for(int j=0; j<cdof.v.n_rows; j++)
                    {
                        if(cdof.v(i)<=cdof.v(j))
                        {
                            sys.A_mat()(cdof.v(i),cdof.v(j)) += std::complex<double>(0.0,k0)*lMat.Tt(i,j)*std::sqrt(epsr/mur);
                            if((cdof.v(i)<sys.doflevel_vec()[1]) && (cdof.v(j) < sys.doflevel_vec()[1]))
                            {
                                sys.AFF_vec()[msh->tetDom(adjTet(0))](cdof.v(i),cdof.v(j)) +=
                                    std::complex<double>(0.0,k0)*lMat.Tt(i,j)*std::sqrt(epsr/mur);
                            }
                        }
                    }
                }
            }
            logFile << lt.toc() << " s\n";
                std::cout << " ";
            break;
        case bc::perfect_h:
            lt.tic();
                std::cout << bc->name;
            logFile << "\t" << bc->name << ": ";
            logFile << lt.toc() << " s\n";
                std::cout << " ";
            break;
        default:
            throw std::runtime_error("Wrong boundary type");
        }
    }
    sys.set_wave_ports_num(0);
    sys.set_wave_ports_dofnum(0);
    sys.wave_portIds_vec().reset();
    for(size_t bcid = 0; bcid < msh->facbc.size(); bcid++)
    {
        bc* bc = &(msh->facbc[bcid]);
        if(bc->type == bc::wave_port)
        {
            lt.tic();
                std::cout << bc->name;
            logFile << "\t" << bc->name << ": ";
            arma::uvec tmpEdges, tmpdofv, tmpDirdofv;
            arma::uvec tmpNodes, tmpdofs, tmpDirdofs;
            arma::uvec tmp;
            tmpdofs = arma::zeros<arma::uvec>(dof(prj).dofnums);
            tmpdofv = arma::zeros<arma::uvec>(dof(prj).dofnumv);
            for(size_t fid = 0; fid < bc->Faces.size(); fid++)
            {
                dof cdof(prj, 2, bc->Faces(fid));
                cdof.v = sys.dofmapv_vec().elem(cdof.v);
                tmpEdges = arma::join_cols(tmpEdges, cdof.v);
                tmpNodes = arma::join_cols(tmpNodes, cdof.s);
            }
            tmpNodes = arma::unique(tmpNodes);
            tmpEdges = arma::unique(tmpEdges);
            tmp.reset();
            tmp = arma::uvec(tmpEdges.n_rows);
            tmp.fill(0);
            for(size_t i = 0; i < tmpEdges.n_rows; i++)
            {
                tmpdofv(tmpEdges(i)) = i;
                arma::uvec pD = arma::find(sys.Dirdofv_vec() == tmpEdges(i));
                if(pD.n_rows == 0)
                {
                    tmp(i) = 1;
                }
            }
            tmpDirdofv = arma::find(tmp > 0);
            tmp.reset();
            tmp = arma::uvec(tmpNodes.n_rows);
            tmp.fill(0);
            for(size_t i = 0; i < tmpNodes.n_rows; i++)
            {
                tmpdofs(tmpNodes(i)) = i;
                arma::uvec pD = arma::find(sys.Dirdofs_vec() == tmpNodes(i));
                if(pD.n_rows == 0)
                {
                    tmp(i) = 1;
                }
            }
            tmpDirdofs = arma::find(tmp > 0);
            {
                for(size_t dofid=0; dofid < tmpEdges.n_rows; dofid++)
                {
                    doftoLeave[tmpEdges(dofid)] = false;
                }
            }
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
                mtrl* cmtrl = &(msh->tetmtrl[msh->tetLab(adjTet(0))]);
                std::complex<double> epsr(cmtrl->epsr, cmtrl->calc_epsr2(sys.get_freq()));
                double mur = cmtrl->mur;
                maxepsr = std::max(maxepsr, std::real(epsr));
                maxmur = std::max(maxmur, mur);
                ele_mat lMat(opt->p_ord, 2, msh->fac_geo(bc->Faces(fid)), quadr,
                            cmtrl, msh->int_node(bc->Faces(fid)));
                dof cdof(prj, 2, bc->Faces(fid));
                cdof.v = sys.dofmapv_vec().elem(cdof.v);
                for(int i=0; i<cdof.v.n_rows; i++)
                {
                    for(int j=0; j<cdof.v.n_rows; j++)
                    {
                        tmpSt(tmpdofv(cdof.v(i)),tmpdofv(cdof.v(j))) += lMat.St(i,j)/mur;
                        tmpTt(tmpdofv(cdof.v(i)),tmpdofv(cdof.v(j))) += lMat.Tt(i,j)*epsr;
                        tmpTt2(tmpdofv(cdof.v(i)),tmpdofv(cdof.v(j))) += lMat.Tt(i,j)/mur;
                    }
                }
                for(int i=0; i<cdof.s.n_rows; i++)
                {
                    for(int j=0; j<cdof.s.n_rows; j++)
                    {
                        tmpSz(tmpdofs(cdof.s(i)),tmpdofs(cdof.s(j))) += lMat.Sz(i,j)/mur;
                        tmpTz(tmpdofs(cdof.s(i)),tmpdofs(cdof.s(j))) += lMat.Tz(i,j)*epsr;
                    }
                }
                for(int i=0; i<cdof.v.n_rows; i++)
                {
                    for(int j=0; j<cdof.s.n_rows; j++)
                    {
                        tmpG(tmpdofv(cdof.v(i)),tmpdofs(cdof.s(j))) += lMat.G(i,j)/mur;
                    }
                }
            }
            tmpSt = tmpSt(tmpDirdofv,tmpDirdofv);
            tmpTt = tmpTt(tmpDirdofv,tmpDirdofv);
            tmpTt2 = tmpTt2(tmpDirdofv,tmpDirdofv);
            tmpSz = tmpSz(tmpDirdofs,tmpDirdofs);
            tmpTz = tmpTz(tmpDirdofs,tmpDirdofs);
            tmpG = tmpG(tmpDirdofv,tmpDirdofs);
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
                eigen ceigen(tmpA, tmpB, numt, numz, shift, bc->num_modes);
                bc->mode_beta = ceigen.mode_beta;
                    for(int i=0; i<bc->num_modes; i++)
                    {
                        std::cout << bc->mode_beta(i);
                    }
                bc->mode_vec = ceigen.mode_vec.rows(0,numt-1)*arma::inv(arma::sqrt(ceigen.mode_vec.st()*tmpB*ceigen.mode_vec));
                arma::cx_mat coeff(bc->mode_beta.size(),bc->mode_beta.size());
                coeff.diag() = arma::sqrt(std::complex<double>(0.0, k0*consts::z0)/bc->mode_beta);
                //std::cout << coeff;
                bc->mode_vecf = (tmpB * (ceigen.mode_vec * coeff *
                                        arma::inv(arma::sqrt(ceigen.mode_vec.st()*tmpB*ceigen.mode_vec))));
                bc->mode_vecf = bc->mode_vecf.rows(0,numt-1);
                bc->mode_vecdof = tmpEdges.elem(tmpDirdofv);
            }
            tmpA.clear();
            tmpB.clear();
            sys.set_wave_ports_num(sys.get_wave_ports_num() + bc->num_modes);
            sys.set_wave_ports_dofnum(sys.get_wave_ports_dofnum() + bc->mode_vec.n_rows);
            sys.wave_portIds_vec() = arma::join_cols(sys.wave_portIds_vec(),bc->mode_vecdof);
                std::cout << " ";
            logFile << lt.toc() << " s\n";
        }
    }
    mem_stat::print(logFile);
    logFile << "Finishing:\n";
    lt.tic();
    if(sys.get_wave_ports_num() > 0)
    {
        {
            arma::cx_mat Meig, MAcoeff;
            Meig.set_size(sys.get_wave_ports_num(), sys.get_wave_ports_dofnum());
            MAcoeff.set_size(sys.get_wave_ports_num(), sys.get_wave_ports_num());
            sys.B_mat().set_size(sys.get_wave_ports_num(), sys.get_wave_ports_num());
            arma::cx_mat B_exc(sys.get_wave_ports_num(), sys.get_wave_ports_num(), arma::fill::zeros);
            size_t idx = 0;
            size_t jdx = 0;
            for(size_t bcid = 0; bcid < msh->facbc.size(); bcid++)
            {
                bc* bc = &(msh->facbc[bcid]);
                if(bc->type == bc::wave_port)
                {
                    for(size_t i=0; i<bc->num_modes; i++)
                    {
                        std::complex<double> sqrtBeta(std::sqrt(std::complex<double>(0.0, k0*consts::z0*opt->power)/bc->mode_beta(i)));
                        MAcoeff(idx,idx) = std::complex<double>(0.0, k0*consts::z0*opt->power);
                        // j 2 k0 z0
                        sys.B_mat()(idx,idx) = std::complex<double>(0.0, 2*k0*consts::z0*opt->power);
                        B_exc(idx,idx) = std::complex<double>(0.0, 2*k0*consts::z0*opt->power);
                        for(size_t j=0; j< bc->mode_vec.n_rows; j++)
                        {
                            Meig(idx,jdx+j) = sqrtBeta * bc->mode_vec(j,i);
                        }
                        idx++;
                    }
                    jdx += bc->mode_vec.n_rows;
                    for(size_t i=0; i<bc->mode_vecdof.size(); i++)
                    {
                        bc->mode_vecdof(i) = sys.Invdofmapv_vec()(bc->mode_vecdof(i));
                    }
                }
            }
            sys.Nonwave_portIds_vec().resize(sys.get_dofnum()-(sys.wave_portIds_vec().n_rows+sys.Dirdofv_vec().n_rows));
            idx = 0;
            for(size_t i = 0; i<sys.get_dofnum(); i++)
            {
                if(doftoLeave[i] == true)
                {
                    sys.Nonwave_portIds_vec()(idx++) = i;
                }
            }
            std::vector<size_t> idToKeep;
            for(size_t rgid=sys.doflevel_vec()[0]; rgid < sys.doflevel_vec()[1]; rgid++)
            {
                if(doftoLeave[rgid] == true)
                {
                    idToKeep.push_back(rgid);
                }
            }
            std::vector<size_t> newdoflevel;
            idx = 0;
            newdoflevel.push_back(idx);
            idx = sys.get_wave_ports_num();
            for(size_t irg=1; irg< sys.doflevel_vec().size(); irg++)
            {
                for(size_t rgid=sys.doflevel_vec()[irg-1]; rgid<sys.doflevel_vec()[irg]; rgid++)
                {
                    if(doftoLeave[rgid] == true)
                    {
                        idx++;
                    }
                }
                newdoflevel.push_back(idx);
            }
            sys.doflevel_vec() = newdoflevel;
            newdoflevel.clear();
            for(size_t did = 0; did < msh->nDomains; did++)
            {
                eq_sys::mat_row_type tmpMat(idToKeep.size(),idToKeep.size());
                { std::unordered_map<size_t,size_t> ikm; for(size_t k=0;k<idToKeep.size();k++) ikm[idToKeep[k]]=k; for(auto it=sys.AFF_vec()[did].begin();it!=sys.AFF_vec()[did].end();++it){auto ri=ikm.find(it.row());if(ri==ikm.end())continue;auto ci=ikm.find(it.col());if(ci==ikm.end())continue;tmpMat(ri->second,ci->second)=*it;} }
                sys.AFF_vec()[did]= tmpMat /* swap */;
            }
            sys.set_dofreal(sys.Nonwave_portIds_vec().n_rows + sys.get_wave_ports_num());
                std::cout << "\nSYS dof = " << sys.get_dofreal() << "\n";
            logFile << "\tSYS dof = " << sys.get_dofreal() << ", ";
            sys.B_mat().set_size(sys.get_dofreal(), sys.get_wave_ports_num());
            for(size_t bi=0; bi<sys.get_wave_ports_num(); bi++)
                for(size_t bj=0; bj<sys.get_wave_ports_num(); bj++)
                    if(std::abs(B_exc(bi,bj)) > 0) sys.B_mat()(bi,bj) = B_exc(bi,bj);
            lt.tic(); /// there
            std::vector<size_t> nnWPids(sys.Nonwave_portIds_vec().n_rows);
            std::vector<size_t> WPids(sys.wave_portIds_vec().n_rows);
            for(size_t i=0; i<sys.Nonwave_portIds_vec().n_rows; i++)
            {
                nnWPids[i] = sys.Nonwave_portIds_vec()(i);
                sys.Nonwave_portIds_vec()(i) = sys.Invdofmapv_vec()(sys.Nonwave_portIds_vec()(i));
            }
            for(size_t i=0; i<sys.wave_portIds_vec().n_rows; i++)
            {
                WPids[i] = sys.wave_portIds_vec()(i);
            }
            sys.non_dir_ids_vec().resize(sys.get_dofreal());
            idx = 0;
            for(size_t i = 0; i<sys.get_dofnum(); i++)
            {
                if(doftoLeave[i] == true)
                {
                    sys.non_dir_ids_vec()[idx++] = i;
                }
            }
            for(size_t i = 0; i<sys.get_dofreal(); i++)
            {
                sys.non_dir_ids_vec()[i] =  sys.Invdofmapv_vec()(sys.non_dir_ids_vec()[i]);
            }
            eq_sys::mat_row_type Anew(sys.get_dofreal(), sys.get_dofreal()), Adiag(sys.get_dofnum(), sys.get_dofnum());
            // Build index maps
            std::unordered_map<size_t, size_t> nnWPids_map, WPids_map;
            for(size_t k=0; k<nnWPids.size(); k++) nnWPids_map[nnWPids[k]] = k;
            for(size_t k=0; k<WPids.size(); k++) WPids_map[WPids[k]] = k;
            size_t wnum = WPids.size(), nnum = nnWPids.size();
            // Aii
            for(auto it = sys.A_mat().begin(); it != sys.A_mat().end(); ++it) {
                auto ri = nnWPids_map.find(it.row()); if(ri == nnWPids_map.end()) continue;
                auto ci = nnWPids_map.find(it.col()); if(ci == nnWPids_map.end()) continue;
                Anew(sys.get_wave_ports_num() + ri->second, wnum + ci->second) = *it;
            }
            for(size_t i=0; i<sys.get_dofnum(); i++)
            {
                Adiag(i,i) = sys.A_mat()(i,i);
                sys.A_mat()(i,i) *= 0.0;
            }
            // TFE modal reduction: gmm::sub_index selects BOTH rows and columns
            {
                arma::cx_mat A_wp(wnum, wnum, arma::fill::zeros);
                arma::cx_mat Ad_wp(wnum, wnum, arma::fill::zeros);
                arma::cx_mat A_wp_nn(wnum, nnum, arma::fill::zeros);
                arma::cx_mat A_nn_wp(nnum, wnum, arma::fill::zeros);
                arma::cx_mat Ad_wp_nn(wnum, nnum, arma::fill::zeros);
                for(auto it = sys.A_mat().begin(); it != sys.A_mat().end(); ++it) {
                    size_t r = it.row(), c = it.col();
                    auto wi = WPids_map.find(r);
                    if(wi != WPids_map.end()) {
                        auto wj = WPids_map.find(c);
                        if(wj != WPids_map.end())
                            A_wp(wi->second, wj->second) += std::complex<double>(*it);
                        auto ni = nnWPids_map.find(c);
                        if(ni != nnWPids_map.end())
                            A_wp_nn(wi->second, ni->second) += std::complex<double>(*it);
                    }
                    auto ni = nnWPids_map.find(r);
                    if(ni != nnWPids_map.end()) {
                        auto wj = WPids_map.find(c);
                        if(wj != WPids_map.end())
                            A_nn_wp(ni->second, wj->second) += std::complex<double>(*it);
                    }
                }
                for(size_t i=0; i<sys.get_dofnum(); i++) {
                    auto wi = WPids_map.find(i);
                    if(wi != WPids_map.end()) {
                        Ad_wp(wi->second, wi->second) = Adiag(i,i);
                        auto ni = nnWPids_map.find(i);
                        if(ni != nnWPids_map.end()) Ad_wp_nn(wi->second, ni->second) = Adiag(i,i);
                    }
                }
                arma::cx_mat Attred = Meig * A_wp + Meig * A_wp.t() + Meig * Ad_wp;
                arma::cx_mat top_left = Attred * Meig.t() + MAcoeff;
                for(size_t i=0; i < size_t(sys.get_wave_ports_num()); i++)
                    for(size_t j=0; j <= i; j++)
                        Anew(j, i) = top_left(j, i);
                arma::cx_mat Atired = Meig * A_wp_nn + Meig * A_nn_wp.t() + Meig * Ad_wp_nn;
                for(size_t i=0; i < size_t(sys.get_wave_ports_num()); i++)
                    for(size_t j=0; j < nnum; j++) {
                        Anew(i, wnum + j) = Atired(i, j);
                        Anew(sys.get_wave_ports_num() + j, i) = Atired(i, j);
                    }
            }
            Adiag.zeros();
            std::swap(sys.A_mat(), Anew);
            Anew.zeros();
            Meig.zeros();
            MAcoeff.zeros();
            nnWPids.clear();
            WPids.clear();
        }
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
        sys.set_dofreal(sys.get_dofnum());
            std::cout << "\nSYS dof = " << sys.get_dofreal() << "\n";
        logFile << "\tSYS dof = " << sys.get_dofreal() << ", ";
        sys.Sol_mat().clear();
        sys.Sol_mat().resize(sys.get_dofreal(),1);
        sys.Sol_mat().fill(0);
        sys.B_mat().set_size(sys.get_dofnum(), 1);
        for(size_t bcid = 0; bcid < msh->facbc.size(); bcid++)
        {
            bc* bc = &(msh->facbc[bcid]);
            if(bc->type == bc::radiation)
            {
                for(size_t fid = 0; fid < bc->Faces.size(); fid++)
                {
                    arma::uvec adjTet = msh->facAdjTet(bc->Faces(fid));
                    double epsr = msh->tetmtrl[msh->tetLab(adjTet(0))].epsr;
                    double mur = msh->tetmtrl[msh->tetLab(adjTet(0))].mur;
                    ele_mat lMat(opt->p_ord, 2, msh->fac_geo(bc->Faces(fid)), quadr,
                                &(msh->tetmtrl[msh->tetLab(adjTet(0))]),
                                msh->int_node(bc->Faces(fid)), kEinc, polEinc);
                    dof cdof(prj, 2, bc->Faces(fid));
                    for(int i=0; i<cdof.v.n_rows; i++)
                    {
                        sys.B_mat()(cdof.v(i),0) += lMat.f(i) * std::complex<double>(0.0,k0);
                    }
                }
            }
        }
        std::vector<size_t> DirIds(sys.Dirdofv_vec().size());
        for(size_t i=0; i<sys.Dirdofv_vec().size(); i++)
        {
            DirIds[i] = sys.Dirdofv_vec()(i);
        }
        for(size_t dirid : DirIds) {
            for(auto it = sys.A_mat().begin_row(dirid); it != sys.A_mat().end_row(dirid); )
            { size_t col = it.col(); ++it; sys.A_mat()(dirid, col) = std::complex<double>(0,0); }
            for(auto it = sys.A_mat().begin_col(dirid); it != sys.A_mat().end_col(dirid); )
            { size_t row = it.row(); ++it; sys.A_mat()(row, dirid) = std::complex<double>(0,0); }
        }
        for(size_t i=0; i<DirIds.size(); i++)
        {
            sys.A_mat()(DirIds[i],DirIds[i]) = 1.0;
        }
    }
    sys.set_symm_flag(2);
    logFile << " " << lt.toc() << " s\n";
    logFile << "+" << tt.toc() << " s\n";
}
