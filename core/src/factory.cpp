#include "assembler.h"
#include "equation_system.h"
#include "option.h"
#include "mesh.h"
#include "boundary_condition.h"
#include "element_matrix.h"
#include "eigen_solver.h"
#include "constants.h"
#include "degree_of_freedom.h"
#include "memory.h"

#include <fstream>
#include <sstream>
#include <string>
#include <complex>
#include <map>
#include <vector>
#include <cfloat>
#include <algorithm>

// ── Factory ──

std::unique_ptr<assembler> assembler::create(Type type)
{
    switch(type) {
    case em_e_fd:        return std::make_unique<assembler_em_e_fd>();
    case em_e_fd_dd:     return std::make_unique<assembler_em_e_fd_dd>();
    case em_e_fd_dd_schur:  return std::make_unique<assembler_em_e_fd_schur>();
    case em_e_fd_nl:     return std::make_unique<assembler_em_e_fd_nl>();
    case em_e_qs:        return std::make_unique<assembler_em_e_qs>();
    case em_ez_fd:       return std::make_unique<assembler_em_ez_fd>();
    case em_e_tl_eig:    return std::make_unique<assembler_em_e_tl_eig>();
    }
    return nullptr;
}

// ── Shared utilities ──

void assembler::read_port_amplitudes(std::ofstream& log, eq_sys& sys)
{
    option* opt = sys.opt;
    double tmpDblr, tmpDbli;
    std::string line;
    std::ifstream fileName(std::string(opt->name + "_Ports.txt").c_str(), std::ios::in | std::ios::binary);
    if(fileName.is_open())
    {
        std::cout << "Provided port amplitudes: ";
        while(getline(fileName,line))
        {
            std::istringstream iss(line);
            iss >> tmpDblr;
            iss >> tmpDbli;
            sys.port_ampl_vec().push_back(std::complex<double>(tmpDblr,tmpDbli));
            std::cout << tmpDblr << "+j" << tmpDbli << "  ";
        }
        std::cout << "\n";
    }
}

eq_sys::mat_row_type assembler::build_sparse(
    const std::map<std::pair<arma::uword, arma::uword>, std::complex<double>>& map,
    size_t n)
{
    std::vector<arma::uword> rows, cols;
    std::vector<std::complex<double>> vals;
    for(auto& kv : map) {
        if(kv.second != std::complex<double>(0,0)) {
            rows.push_back(kv.first.first);
            cols.push_back(kv.first.second);
            vals.push_back(kv.second);
        }
    }
    if(rows.empty())
        return eq_sys::mat_row_type(n, n);

    arma::umat locs(2, rows.size());
    arma::cx_vec cvals(vals.data(), vals.size());
    for(size_t k = 0; k < rows.size(); k++) {
        locs(0,k) = rows[k];
        locs(1,k) = cols[k];
    }
    return eq_sys::mat_row_type(locs, cvals, n, n, true, true);
}

void assembler::rebuild_sparse(eq_sys::mat_row_type& mat, size_t n)
{
    arma::umat locs(2, mat.n_nonzero);
    arma::cx_vec vals(mat.n_nonzero);
    size_t k = 0;
    for(auto it = mat.begin(); it != mat.end(); ++it, ++k) {
        locs(0,k) = it.row();
        locs(1,k) = it.col();
        vals(k) = *it;
    }
    mat = eq_sys::mat_row_type(locs, vals, n, n, true, true);
}

size_t assembler::compute_waveport_modes(
    std::ofstream& logFile, eq_sys& sys,
    mesh* msh, option* opt, project* prj, quad* quadr,
    double k0, double kk,
    std::vector<bool>& doftoLeave,
    arma::wall_clock& lt,
    const arma::field<arma::uvec>* domdofmap,
    const arma::umat* shareddofv)
{
    size_t wave_portsNum_start = sys.get_wave_ports_num();
    for(size_t bcid = 0; bcid < msh->facbc.size(); bcid++)
    {
        bc* bc = &(msh->facbc[bcid]);
        if(bc->type != bc::wave_port && bc->type != bc::lumped_port) continue;

        lt.tic();
        std::cout << bc->name;
        logFile << "\t" << bc->name << ": ";

        arma::uvec tmpEdges, tmpdofv, tmpDirdofv;
        arma::uvec tmpNodes, tmpdofs, tmpDirdofs;
        arma::uvec tmp;
        tmpdofs = arma::zeros<arma::uvec>(dof(prj).dofnums);
        tmpdofv = arma::zeros<arma::uvec>(dof(prj).dofnumv);

        // DD variant: additional shared DOF fixup
        arma::uvec tmpShared, tmpSharedDD, tmpSharedReg;
        arma::umat shareddofv_local;

        for(size_t fid = 0; fid < bc->Faces.size(); fid++)
        {
            dof cdof(prj, 2, bc->Faces(fid));
            if(domdofmap) {
                // DD: track shared DOFs across domains
                tmpShared = arma::join_cols(tmpShared, cdof.v);
                arma::uvec adjTet = msh->facAdjTet(bc->Faces(fid));
                cdof.v = (*domdofmap)(msh->tetDom(adjTet(0))).elem(cdof.v);
                tmpSharedDD = arma::join_cols(tmpSharedDD, cdof.v);
                tmpSharedReg = arma::join_cols(tmpSharedReg, 0*cdof.v + msh->tetDom(adjTet(0)));
            }
            tmpEdges = arma::join_cols(tmpEdges, cdof.v);
            tmpNodes = arma::join_cols(tmpNodes, cdof.s);
        }

        // DD: resolve shared DOFs
        if(domdofmap && tmpShared.size() > 0) {
            arma::uvec sharedRegs = arma::unique(tmpSharedReg);
            for(size_t i = 0; i < tmpShared.size(); i++) {
                arma::uvec dofs = arma::sort(arma::find(tmpShared == tmpShared(i)));
                if(dofs.size() == 2 && tmpSharedReg(dofs(0)) != tmpSharedReg(dofs(1))) {
                    arma::uvec dofv = arma::sort(tmpSharedDD.elem(dofs));
                    if(shareddofv_local.n_rows > 0) {
                        tmp = arma::find(shareddofv_local.col(0) == dofv(0));
                        if(tmp.size() == 0)
                            shareddofv_local = arma::join_cols(shareddofv_local, dofv.st());
                    } else {
                        shareddofv_local = arma::join_cols(shareddofv_local, dofv.st());
                    }
                }
            }
            for(size_t i = 0; i < shareddofv_local.n_rows; i++) {
                tmp = arma::find(tmpEdges == shareddofv_local(i,1));
                if(tmp.size() > 0) {
                    for(size_t j = 0; j < tmp.n_rows; j++)
                        tmpEdges(tmp(j)) = arma::min(shareddofv_local.row(i));
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
            tmpdofv(tmpEdges(i)) = i;
            arma::uvec pD = arma::find(sys.Dirdofv_vec() == tmpEdges(i));
            if(pD.n_rows == 0) tmp(i) = 1;
        }
        tmpDirdofv = arma::find(tmp > 0);
        tmp.reset();
        tmp = arma::uvec(tmpNodes.n_rows);
        tmp.fill(0);
        for(size_t i = 0; i < tmpNodes.n_rows; i++)
        {
            tmpdofs(tmpNodes(i)) = i;
            arma::uvec pD = arma::find(sys.Dirdofs_vec() == tmpNodes(i));
            if(pD.n_rows == 0) tmp(i) = 1;
        }
        tmpDirdofs = arma::find(tmp > 0);
        {
            for(size_t dofid=0; dofid < tmpEdges.n_rows; dofid++)
                doftoLeave[tmpEdges(dofid)] = false;
        }

        arma::cx_mat tmpSt(tmpEdges.n_rows, tmpEdges.n_rows);
        arma::cx_mat tmpTt(tmpEdges.n_rows, tmpEdges.n_rows);
        arma::cx_mat tmpTt2(tmpEdges.n_rows, tmpEdges.n_rows);
        arma::cx_mat tmpSz(tmpNodes.n_rows, tmpNodes.n_rows);
        arma::cx_mat tmpTz(tmpNodes.n_rows, tmpNodes.n_rows);
        arma::cx_mat tmpG(tmpEdges.n_rows, tmpNodes.n_rows);
        tmpSt.fill(0); tmpTt.fill(0); tmpTt2.fill(0);
        tmpSz.fill(0); tmpTz.fill(0); tmpG.fill(0);
        double maxepsr = DBL_MIN, maxmur = DBL_MIN;

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
            for(int i=0; i<cdof.v.n_rows; i++) {
                for(int j=0; j<cdof.v.n_rows; j++) {
                    tmpSt(tmpdofv(cdof.v(i)), tmpdofv(cdof.v(j))) += lMat.St(i,j)/mur;
                    tmpTt(tmpdofv(cdof.v(i)), tmpdofv(cdof.v(j))) += lMat.Tt(i,j)*epsr;
                    tmpTt2(tmpdofv(cdof.v(i)), tmpdofv(cdof.v(j))) += lMat.Tt(i,j)/mur;
                }
            }
            for(int i=0; i<cdof.s.n_rows; i++) {
                for(int j=0; j<cdof.s.n_rows; j++) {
                    tmpSz(tmpdofs(cdof.s(i)), tmpdofs(cdof.s(j))) += lMat.Sz(i,j)/mur;
                    tmpTz(tmpdofs(cdof.s(i)), tmpdofs(cdof.s(j))) += lMat.Tz(i,j)*epsr;
                }
            }
            for(int i=0; i<cdof.v.n_rows; i++) {
                for(int j=0; j<cdof.s.n_rows; j++) {
                    tmpG(tmpdofv(cdof.v(i)), tmpdofs(cdof.s(j))) += lMat.G(i,j)/mur;
                }
            }
        }
        tmpSt = tmpSt(tmpDirdofv, tmpDirdofv);
        tmpTt = tmpTt(tmpDirdofv, tmpDirdofv);
        tmpTt2 = tmpTt2(tmpDirdofv, tmpDirdofv);
        tmpSz = tmpSz(tmpDirdofs, tmpDirdofs);
        tmpTz = tmpTz(tmpDirdofs, tmpDirdofs);
        tmpG = tmpG(tmpDirdofv, tmpDirdofs);
        size_t numt = tmpSt.n_rows, numz = tmpSz.n_rows;
        size_t numtot = numt + numz;
        arma::cx_mat tmpA(numtot, numtot);
        arma::cx_mat tmpB(numtot, numtot);
        tmpA.fill(0); tmpB.fill(0);
        tmpA(arma::span(0,numt-1), arma::span(0,numt-1)) = tmpSt - kk*tmpTt;
        tmpB(arma::span(0,numt-1), arma::span(0,numt-1)) = tmpTt2;
        if(numz > 0) {
            tmpB(arma::span(numt,numtot-1), arma::span(numt,numtot-1)) = tmpSz - kk*tmpTz;
            tmpB(arma::span(0,numt-1), arma::span(numt,numtot-1)) = tmpG;
            tmpB(arma::span(numt,numtot-1), arma::span(0,numt-1)) = tmpG.st();
        }
        tmpSt.clear(); tmpTt.clear(); tmpSz.clear(); tmpTz.clear(); tmpG.clear();

        {
            double shift = -kk * maxepsr * maxmur;
            eigen ceigen(tmpA, tmpB, numt, numz, shift, bc->num_modes);
            bc->mode_beta = ceigen.mode_beta;
            for(int i=0; i<bc->num_modes; i++)
                std::cout << bc->mode_beta(i);

            arma::cx_mat coeff(bc->mode_beta.size(), bc->mode_beta.size());
            coeff.diag() = arma::sqrt(std::complex<double>(0.0, k0*consts::z0) / (bc->mode_beta));
            bc->mode_vec = ceigen.mode_vec.rows(0,numt-1) *
                          arma::inv(arma::sqrt(ceigen.mode_vec.st() * tmpB * ceigen.mode_vec));
            bc->mode_vecf = (tmpB * (ceigen.mode_vec * coeff *
                                      arma::inv(arma::sqrt(ceigen.mode_vec.st() * tmpB * ceigen.mode_vec))));
            bc->mode_vecf = bc->mode_vecf.rows(0,numt-1);
            // Normalize eigenvector signs for determinism
            for(size_t m=0; m<bc->mode_vec.n_cols; m++) {
                size_t best_r = 0;
                double best_v = -1.0;
                for(size_t r=0; r<bc->mode_vec.n_rows; r++) {
                    double av = std::abs(std::real(bc->mode_vec(r,m)));
                    if(av > best_v) { best_v = av; best_r = r; }
                }
                if(std::real(bc->mode_vec(best_r,m)) < 0) {
                    bc->mode_vec.col(m) *= -1.0;
                    bc->mode_vecf.col(m) *= -1.0;
                }
            }
            bc->mode_vecdof = tmpEdges.elem(tmpDirdofv);
        }
        tmpA.clear(); tmpB.clear();
        sys.set_wave_ports_num(sys.get_wave_ports_num() + bc->num_modes);
        sys.set_wave_ports_dofnum(sys.get_wave_ports_dofnum() + bc->mode_vec.n_rows);
        sys.wave_portIds_vec() = arma::join_cols(sys.wave_portIds_vec(), bc->mode_vecdof);
        std::cout << " ";
        logFile << lt.toc() << " s\n";
    }
    return sys.get_wave_ports_num() - wave_portsNum_start;
}
