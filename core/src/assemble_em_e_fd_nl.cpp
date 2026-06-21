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

#include <cfloat> // for DBL_MIN
#include <unordered_map>
#include <complex>

void assembler_em_e_fd_nl::assemble(std::ofstream& logFile, eq_sys& sys)
{
    mesh* msh = sys.msh;
    option* opt = sys.opt;
    project* prj = sys.prj;
    quad* quadr = sys.quadr;
    double hCoeff[] = {1,3,5,7,9,11,13};
    logFile << "% Assembly:\n";
    logFile << "In solids: ";
    arma::wall_clock tt, lt;
    tt.tic();
    double k0 = 2.0 * consts::pi * sys.get_freq() / consts::c0;
    double kk = k0*k0;
    sys.set_dofnum(dof(prj).dofnumv);
    logFile << "% Assembly nonlinear:\n";
    dof* cdof = new dof(prj, 3, 0);
    size_t num3 = cdof->v.n_rows;
    cdof = new dof(prj, 2, 0);
    size_t num2 = cdof->v.n_rows;
    delete cdof;
    std::cout << "FE dof = " << sys.get_dofnum()* opt->n_harm << " ";
    sys.set_symm_flag(0);
    sys.A_mat().zeros();
    sys.B_mat().zeros();
    sys.PR_mat().zeros();
    sys.A_mat().set_size(opt->n_harm*sys.get_dofnum(), opt->n_harm*sys.get_dofnum());
    if(sys.sol_prev_mat().n_rows==0)
    {
        sys.sol_prev_mat().resize(opt->n_harm*sys.get_dofnum(), 1);
        sys.sol_prev_mat().fill(0);
    }
    sys.Sol_mat().clear();
    sys.Sp_mat().clear();
    lt.tic();
    arma::vec k0Coeff(opt->n_harm*num3);
    arma::vec kkCoeff(opt->n_harm*num3);
    k0Coeff.fill(k0);
    kkCoeff.fill(kk);
    for(size_t ih=1; ih < opt->n_harm; ih++)
    {
        for(size_t i=0; i < num3; i++)
        {
            k0Coeff(ih*num3+i) = hCoeff[ih]*k0;
            kkCoeff(ih*num3+i) = std::pow(hCoeff[ih],2)*kk;
        }
    }
    for(size_t id = 0; id < msh->nTetras; id++)
    {
        dof cdof(prj, 3, id);
        cdof.v.resize(opt->n_harm*num3);
        for(size_t ih=1; ih < opt->n_harm; ih++)
        {
            for(size_t i=0; i < num3; i++)
            {
                cdof.v(ih*num3+i) = ih*sys.get_dofnum() + cdof.v(i);
            }
        }
        mtrl* cmtrl = &(msh->tetmtrl[msh->tetLab(id)]);
        ele_mat lMat(opt->p_ord, 3, msh->tet_geo(id), quadr, cmtrl, &cdof, sys.sol_prev_mat().col(0), opt->n_harm, sys.get_dofnum(), sys.get_freq());
        for(size_t i=0; i<cdof.v.n_rows; i++)
        {
            for(size_t j=0; j<cdof.v.n_rows; j++)
            {
                sys.A_mat()(cdof.v(i),cdof.v(j)) += lMat.S(i,j) + k0Coeff(i)*lMat.Z(i,j) - kkCoeff(i)*lMat.T(i,j);
            }
        }
    }
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
                cdof.v.resize(opt->n_harm*num2);
                for(size_t ih=1; ih < opt->n_harm; ih++)
                {
                    for(size_t i=0; i < num2; i++)
                    {
                        cdof.v(ih*num2+i) = ih*sys.get_dofnum() + cdof.v(i);
                    }
                }
                sys.Dirdofs_vec() = arma::join_cols(sys.Dirdofs_vec(), cdof.s);
                sys.Dirdofv_vec() = arma::join_cols(sys.Dirdofv_vec(), cdof.v);
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
            for(size_t ih=0; ih < opt->n_harm; ih++)
            {
                arma::mat centroid(1,3);
                centroid.fill(0.25);
                for(size_t fid = 0; fid < bc->Faces.size(); fid++)
                {
                    arma::uvec adjTet = msh->facAdjTet(bc->Faces(fid));
                    mtrl* cmtrl = &(msh->tetmtrl[msh->tetLab(adjTet(0))]);
                    cmtrl->updmtrl(sys.get_freq());
                    std::complex<double> epsr(cmtrl->epsr, cmtrl->epsr2);
                    double mur = cmtrl->mur;
                    if(cmtrl->kerr != 0.0)
                    {
                        dof cdof3(prj, 3, adjTet(0));
                        jacobian cJac(3, msh->tet_geo(adjTet(0)));
                        shape cShp(opt->p_ord, 3, shape::hcurl, centroid.row(0), &cJac);
                        arma::uvec tdofs = cdof3.v + (size_t)(sys.get_dofnum()*ih);
                        arma::cx_vec tSol = arma::cx_vec(sys.sol_prev_mat().col(0)).elem(tdofs);
                        double normE = arma::norm(cShp.Nv*tSol,2);
                        epsr *= (1 + cmtrl->kerr*std::pow(normE,2));
                    }
                    ele_mat lMat(opt->p_ord, 2, msh->fac_geo(bc->Faces(fid)), quadr,
                                cmtrl, msh->int_node(bc->Faces(fid)));
                    dof cdof(prj, 2, bc->Faces(fid));
                    for(int i=0; i<cdof.v.n_rows; i++)
                    {
                        for(int j=0; j<cdof.v.n_rows; j++)
                        {
                            sys.A_mat()(ih*sys.get_dofnum() + cdof.v(i),ih*sys.get_dofnum() + cdof.v(j)) +=
                                std::complex<double>(0.0,(hCoeff[ih])*k0)*lMat.Tt(i,j)*std::sqrt(epsr/mur);
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
            size_t numt, numz, numtot;
            tmpdofs = arma::zeros<arma::uvec>(dof(prj).dofnums);
            tmpdofv = arma::zeros<arma::uvec>(dof(prj).dofnumv);
            for(size_t fid = 0; fid < bc->Faces.size(); fid++)
            {
                dof cdof(prj, 2, bc->Faces(fid));
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
            arma::mat centroid(1,3);
            centroid.fill(0.25);
            for(size_t ih=0; ih < opt->n_harm; ih++)
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
                    mtrl* cmtrl = &(msh->tetmtrl[msh->tetLab(adjTet(0))]);
                    double cFreq = (hCoeff[ih])*sys.get_freq();
                    cmtrl->updmtrl(cFreq);
                    std::complex<double> epsr(cmtrl->epsr, cmtrl->epsr2);
                    double mur = cmtrl->mur;
                    if(cmtrl->kerr != 0.0)
                    {
                        dof cdof3(prj, 3, adjTet(0));
                        jacobian cJac(3, msh->tet_geo(adjTet(0)));
                        shape cShp(opt->p_ord, 3, shape::hcurl, centroid.row(0), &cJac);
                        arma::uvec tdofs = cdof3.v + (size_t)(sys.get_dofnum()*ih);
                        arma::cx_vec tSol = arma::cx_vec(sys.sol_prev_mat().col(0)).elem(tdofs);
                        double normE = arma::norm(cShp.Nv*tSol,2);
                        epsr *=  std::complex<double>(1 + cmtrl->kerr*std::pow(normE,2),0.0);
                    }
                    maxepsr = std::max(maxepsr, std::real(epsr));
                    maxmur = std::max(maxmur, mur);
                    ele_mat lMat(opt->p_ord, 2, msh->fac_geo(bc->Faces(fid)), quadr,
                                cmtrl, msh->int_node(bc->Faces(fid)));
                    dof cdof(prj, 2, bc->Faces(fid));
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
                if(ih == 0)
                {
                    numt = tmpSt.n_rows;
                    numz = tmpSz.n_rows;
                    numtot = numt + numz;
                    bc->mode_beta.resize(opt->n_harm*bc->num_modes);
                    bc->mode_vec.resize(numt, opt->n_harm*bc->num_modes);
                    bc->mode_vecdof.resize(numt, opt->n_harm);
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
                    eigen ceigen(tmpA, tmpB, numt, numz, shift, bc->num_modes);
                    bc->mode_beta.rows(ih*bc->num_modes,(ih+1)*bc->num_modes-1) = ceigen.mode_beta;
                        for(int i=0; i<bc->num_modes; i++)
                        {
                            std::cout << bc->mode_beta(ih*bc->num_modes+i);
                        }
                    bc->mode_vec.cols(ih*bc->num_modes,(ih+1)*bc->num_modes-1) = ceigen.mode_vec.rows(0,numt-1)*
                            arma::inv(arma::sqrt(ceigen.mode_vec.st()*tmpB*ceigen.mode_vec));
                    bc->mode_vecdof.col(ih) = (sys.get_dofnum()*ih) + tmpEdges.elem(tmpDirdofv);
                }
                tmpA.clear();
                tmpB.clear();
                sys.set_wave_ports_num(sys.get_wave_ports_num() + bc->num_modes);
                sys.set_wave_ports_dofnum(sys.get_wave_ports_dofnum() + bc->mode_vec.n_rows);
                sys.wave_portIds_vec() = arma::join_cols(sys.wave_portIds_vec(),bc->mode_vecdof.col(ih));
            }
                std::cout << " ";
            logFile << lt.toc() << " s\n";
        }
    }
    mem_stat::print(logFile);
    logFile << "Finishing:\n";
    lt.tic();
    if(sys.get_wave_ports_num() > 0)
    {
        arma::cx_mat Meig, MAcoeff;
        Meig.set_size(sys.get_wave_ports_num(), sys.get_wave_ports_dofnum());
        MAcoeff.set_size(sys.get_wave_ports_num(), sys.get_wave_ports_num());
        sys.B_mat().set_size(sys.get_wave_ports_num(), sys.get_wave_ports_num());
        size_t idx = 0;
        size_t jdx = 0;
        for(size_t bcid = 0; bcid < msh->facbc.size(); bcid++)
        {
            bc* bc = &(msh->facbc[bcid]);
            if(bc->type == bc::wave_port)
            {
                for(size_t ih = 0; ih < opt->n_harm; ih++)
                {
                    for(size_t i = 0; i < bc->num_modes; i++)
                    {
                        std::complex<double> sqrtBeta(std::sqrt(std::complex<double>(0.0, k0*consts::z0*opt->power)/bc->mode_beta(ih*bc->num_modes+i)));
                        MAcoeff(idx,idx) = std::complex<double>(0.0, hCoeff[ih]*k0*consts::z0*opt->power);
                        // j 2 k0 z0
                        sys.B_mat()(idx,idx) = std::complex<double>(0.0, 2*hCoeff[ih]*k0*consts::z0*opt->power);
                        for(size_t j=0; j< bc->mode_vec.n_rows; j++)
                        {
                            Meig(idx,jdx+j) = sqrtBeta * bc->mode_vec(j,ih*bc->num_modes+i);
                        }
                        idx++;
                    }
                    jdx += bc->mode_vec.n_rows;
                }
            }
        }
        sys.Nonwave_portIds_vec().resize(sys.get_dofnum()*opt->n_harm-(sys.wave_portIds_vec().n_rows+sys.Dirdofv_vec().n_rows));
        idx = 0;
        arma::uvec tmp;
        for(size_t i = 0; i<sys.get_dofnum()*opt->n_harm; i++)
        {
            tmp = arma::find(sys.wave_portIds_vec() == i);
            tmp = arma::join_cols(tmp, arma::find(sys.Dirdofv_vec() == i));
            if(tmp.n_rows<1)
            {
                sys.Nonwave_portIds_vec()(idx++) = i;
            }
        }
        tmp.clear();
        sys.set_dofreal(sys.Nonwave_portIds_vec().n_rows + sys.get_wave_ports_num());
        std::cout << "\nSYS dof = " << sys.get_dofreal() << "\n";
        logFile << "\tSYS dof = " << sys.get_dofreal() << ", ";
        sys.B_mat().set_size(sys.get_dofreal(), 1);
        lt.tic();
        std::vector<size_t> nnWPids(sys.Nonwave_portIds_vec().n_rows);
        std::vector<size_t> WPids(sys.wave_portIds_vec().n_rows);
        for(size_t i=0; i<sys.Nonwave_portIds_vec().n_rows; i++)
        {
            nnWPids[i] = sys.Nonwave_portIds_vec()(i);
        }
        for(size_t i=0; i<sys.wave_portIds_vec().n_rows; i++)
        {
            WPids[i] = sys.wave_portIds_vec()(i);
        }
        eq_sys::mat_row_type Anew(sys.get_dofreal(), sys.get_dofreal());
        // Build index maps
        std::unordered_map<size_t, size_t> nnWPids_map, WPids_map;
        for(size_t k=0; k<nnWPids.size(); k++) nnWPids_map[nnWPids[k]] = k;
        for(size_t k=0; k<WPids.size(); k++) WPids_map[WPids[k]] = k;
        size_t wnum = WPids.size(), nnum = nnWPids.size();
        // TFE modal reduction: gmm::sub_index selects BOTH rows and columns
        {
            arma::cx_mat A_wp(wnum, wnum, arma::fill::zeros);
            arma::cx_mat A_wp_nn(wnum, nnum, arma::fill::zeros);
            for(auto it = sys.A_mat().begin(); it != sys.A_mat().end(); ++it) {
                size_t r = it.row(), c = it.col();
                auto wi = WPids_map.find(r);
                if(wi != WPids_map.end()) {
                    auto wj = WPids_map.find(c);
                    if(wj != WPids_map.end()) A_wp(wi->second, wj->second) += std::complex<double>(*it);
                    auto ni = nnWPids_map.find(c);
                    if(ni != nnWPids_map.end()) A_wp_nn(wi->second, ni->second) += std::complex<double>(*it);
                }
            }
            arma::cx_mat top_left = Meig * A_wp * Meig.t() + MAcoeff;
            for(size_t i=0; i < size_t(sys.get_wave_ports_num()); i++)
                for(size_t j=0; j < wnum; j++)
                    Anew(j, i) = top_left(j, i);
            arma::cx_mat Atired_dense = Meig * A_wp_nn;
            for(size_t i=0; i < size_t(sys.get_wave_ports_num()); i++) {
                for(size_t j=0; j < nnum; j++) {
                    Anew(i, sys.get_wave_ports_num() + j) = Atired_dense(i, j);
                    Anew(sys.get_wave_ports_num() + j, i) = Atired_dense(i, j);
                }
            }
            // Aii block: A(nnWPids, nnWPids) → Anew(wnum:, wnum:)
            for(auto it = sys.A_mat().begin(); it != sys.A_mat().end(); ++it) {
                auto ri = nnWPids_map.find(it.row());
                if(ri == nnWPids_map.end()) continue;
                auto ci = nnWPids_map.find(it.col());
                if(ci == nnWPids_map.end()) continue;
                Anew(sys.get_wave_ports_num() + ri->second, wnum + ci->second) = *it;
            }
        }
        std::swap(sys.A_mat(), Anew);
        Anew.zeros();
        Meig.zeros();
        MAcoeff.zeros();
        nnWPids.clear();
        WPids.clear();
    }
    else if(opt->einc)
    {
        sys.Sol_mat().clear();
        sys.Sol_mat().resize(sys.get_dofnum()*opt->n_harm,1);
        sys.Sol_mat().fill(0);
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
        std::cout << "\nSYS dof = " << std::setw(8) << std::right << sys.get_dofreal() << "\n";
        logFile << "\tSYS dof = " << std::setw(8) << std::right << sys.get_dofreal() << ", ";
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
    if(sys.B_mat().n_nonzero == 0)
    {
        throw std::runtime_error("Null Right Hand Side");
    }
    sys.set_symm_flag(0);
    logFile << " " << lt.toc() << " s\n";
    logFile << "+" << tt.toc() << "s\n";
}

