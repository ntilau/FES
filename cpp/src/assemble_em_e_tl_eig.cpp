#define ARMA_WARN_LEVEL 0

#include "assembler.h"
#include <stdexcept>
#include <algorithm>
#include "equation_system.h"
#include "mesh.h"
#include "option.h"
#include "project.h"
#include "constants.h"
#include "quadrature.h"
#include "shape.h"
#include "degree_of_freedom.h"
#include "element_matrix.h"
#include "eigen_solver.h"
#include "boundary_condition.h"

#include <armadillo>
#include <iostream>
#include <map>
#include <set>

void assembler_em_e_tl_eig::assemble(std::ofstream& logFile, eq_sys& sys)
{
    mesh* msh = sys.msh;
    option* opt = sys.opt;
    project* prj = sys.prj;

    // em_e_tl_eig only works on 2D cross-sections with NO waveport BCs
    if(msh->nTetras > 0)
        throw std::runtime_error("em_e_tl_eig requires a 2D cross-section mesh (use +formula em_e_fd for 3D)");
    for(auto& bc : msh->facbc)
        if(bc.type == bc::wave_port)
            throw std::runtime_error("em_e_tl_eig does not support models with waveport boundaries; "
                              "use +formula em_ez_fd for driven 2D waveport simulation");

    std::cout << "Transmission line cross-section eigenmode analysis\n";
    logFile << "Transmission line cross-section eigenmode analysis\n";
    arma::wall_clock lt;
    lt.tic();

    double k0 = 2.0 * consts::pi * sys.get_freq() / consts::c0;
    double kk = k0 * k0;

    // Build full 2D edge connectivity (interior edges for hcurl DOFs)
    // Clear facEdges first so build_2d_edge_connectivity always runs,
    // even when loading from a .fes file that already has facEdges populated.
    msh->facEdges.reset();
    msh->build_2d_edge_connectivity();

    // DOF counts
    dof gdof(prj);
    size_t nEdges = gdof.dofnumv;
    size_t nNodes  = gdof.dofnums;
    std::cout << "k0 = " << k0 << " dof = " << (nEdges + nNodes)
              << " (" << nEdges << " hcurl + " << nNodes << " hgrad)\n";

    // Allocate global matrices (dense, complex)
    arma::cx_mat St(nEdges, nEdges, arma::fill::zeros);
    arma::cx_mat Tt(nEdges, nEdges, arma::fill::zeros);
    arma::cx_mat Tt2(nEdges, nEdges, arma::fill::zeros);
    arma::cx_mat Sz(nNodes, nNodes, arma::fill::zeros);
    arma::cx_mat Tz(nNodes, nNodes, arma::fill::zeros);
    arma::cx_mat G(nEdges, nNodes, arma::fill::zeros);
    double maxepsr = 1.0, maxmur = 1.0;

    // Material map (face label → mtrl)
    std::map<size_t, const mtrl*> mtrlMap;
    for(const auto& m : msh->tetmtrl) mtrlMap[m.label] = &m;

    // bc type map for Dirichlet detection
    std::map<size_t, bc::bcTYPE> bcTypeMap;
    for(const auto& bc : msh->facbc) bcTypeMap[bc.label] = bc.type;

    // Triangle loop: assemble 2D hcurl/hgrad element matrices
    for(size_t t = 0; t < msh->nFaces; t++)
    {
        double epsr = 1.0, mur = 1.0;
        size_t lab = msh->facLab(t);
        auto mit = mtrlMap.find(lab);
        if(mit != mtrlMap.end()) { epsr = mit->second->epsr; mur = mit->second->mur; }
        maxepsr = std::max(maxepsr, epsr);
        maxmur  = std::max(maxmur,  mur);

        // Build 3x2 triangle geometry from 2D node positions
        arma::mat triGeo2(3, 2);
        for(int k = 0; k < 3; k++) {
            size_t nid = msh->facNodes(t, k);
            triGeo2(k, 0) = msh->nodPos(nid, 0);
            triGeo2(k, 1) = msh->nodPos(nid, 1);
        }
        ele_mat lMat(opt->p_ord, 2, triGeo2, sys.quadr, nullptr, static_cast<shape::s_type>(1));
        dof cdof(prj, 2, t);

        for(int i = 0; i < (int)cdof.v.n_rows; i++) {
            for(int j = 0; j < (int)cdof.v.n_rows; j++) {
                St(cdof.v(i), cdof.v(j)) += lMat.St(i,j) / mur;
                Tt(cdof.v(i), cdof.v(j)) += lMat.Tt(i,j) * epsr;
                Tt2(cdof.v(i), cdof.v(j)) += lMat.Tt(i,j) / mur;
            }
        }
        for(int i = 0; i < (int)cdof.s.n_rows; i++) {
            for(int j = 0; j < (int)cdof.s.n_rows; j++) {
                Sz(cdof.s(i), cdof.s(j)) += lMat.Sz(i,j) / mur;
                Tz(cdof.s(i), cdof.s(j)) += lMat.Tz(i,j) * epsr;
            }
        }
        for(int i = 0; i < (int)cdof.v.n_rows; i++) {
            for(int j = 0; j < (int)cdof.s.n_rows; j++) {
                G(cdof.v(i), cdof.s(j)) += lMat.G(i,j) / mur;
            }
        }
    }

    // PEC Dirichlet edges → zero tangential E on conducting walls
    std::set<size_t> dirDofvSet;
    size_t nMeshEdges = msh->nEdges;
    for(size_t s = 0; s < nMeshEdges; s++) {
        size_t mkr = msh->edgLab(s);
        auto bcIt = bcTypeMap.find(mkr);
        if(bcIt == bcTypeMap.end() || bcIt->second != bc::perfect_e) continue;
        dirDofvSet.insert(s);
        for(size_t lev = 1; lev < opt->p_ord; lev++)
            dirDofvSet.insert(lev * nMeshEdges + s);
    }

    // Free DOF vectors
    arma::uvec freeDofv(nEdges - dirDofvSet.size(), arma::fill::zeros);
    arma::uvec freeDofs(nNodes, arma::fill::zeros);
    {
        size_t idx = 0;
        for(size_t i = 0; i < nEdges; i++)
            if(dirDofvSet.find(i) == dirDofvSet.end())
                freeDofv(idx++) = i;
    }
    for(size_t i = 0; i < nNodes; i++)
        freeDofs(i) = i;

    size_t numt = freeDofv.n_elem;
    size_t numz = freeDofs.n_elem;
    size_t numtot = numt + numz;

    logFile << "Cross-section eigenproblem: " << nEdges << " edge DOFs, "
            << numt << " free, " << nNodes << " node DOFs\n";
    std::cout << "Cross-section eigenproblem: " << nEdges << " edge DOFs, "
              << numt << " free, " << nNodes << " node DOFs\n";

    if(numt == 0) {
        logFile << "0 free edge DOFs — all PEC edges. No tangential modes.\n";
        std::cout << "0 free edge DOFs — all PEC edges. No tangential modes.\n";
    }
    else if(numtot < 2) {
        logFile << "Warning: too few free DOFs for eigenvalue solve (" << numtot << ")\n";
        std::cout << "Warning: too few free DOFs for eigenvalue solve (" << numtot << ")\n";
    }
    else
    {
        // Sub-index matrices to free DOFs
        arma::cx_mat Stf = St(freeDofv, freeDofv);
        arma::cx_mat Ttf = Tt(freeDofv, freeDofv);
        arma::cx_mat Tt2f = Tt2(freeDofv, freeDofv);
        arma::cx_mat Szf = Sz(freeDofs, freeDofs);
        arma::cx_mat Tzf = Tz(freeDofs, freeDofs);
        arma::cx_mat Gf = G(freeDofv, freeDofs);

        // Block eigenproblem: tmpA*x = lambda*tmpB*x, lambda = beta^2
        arma::cx_mat tmpA(numtot, numtot, arma::fill::zeros);
        arma::cx_mat tmpB(numtot, numtot, arma::fill::zeros);

        tmpA(arma::span(0, numt-1), arma::span(0, numt-1)) = Stf - kk * Ttf;
        tmpB(arma::span(0, numt-1), arma::span(0, numt-1)) = Tt2f;

        if(numz > 0) {
            tmpA(arma::span(numt, numtot-1), arma::span(numt, numtot-1)) = Szf - kk * Tzf;
            tmpB(arma::span(0, numt-1), arma::span(numt, numtot-1)) = Gf;
            tmpB(arma::span(numt, numtot-1), arma::span(0, numt-1)) = Gf.st();
        }

        // Number of modes to compute (up to 20, or half the system)
        int num_modes = std::min(5, (int)(numtot/2 + 1));
        if(num_modes < 1) num_modes = 1;

        // Shift-invert eigenvalue solve via MUMPS + ARPACK
        double shift = -kk * maxepsr * maxmur;
        eigen ceigen(tmpA, tmpB, numt, numz, shift, num_modes);

        arma::cx_vec be = ceigen.mode_beta;
        if(be.n_elem > 0) {
            // Sort by |beta| descending
            arma::uvec ord = arma::sort_index(arma::abs(be), "d");
            be = be(ord);

            std::cout << (int)be.n_elem << " modes:\n";
            for(size_t m = 0; m < be.n_elem; m++) {
                // be = sqrt(gamma^2) where gamma = alpha + j*beta
                double alpha =  std::real(be(m));
                double beta  =  std::imag(be(m));
                double kc = std::sqrt(std::abs(k0*k0 - beta*beta + alpha*alpha));
                std::cout << "  " << m+1 << ": beta=" << beta
                          << " alpha=" << alpha << " kc=" << kc;
                if(beta > k0*1e-6) std::cout << " p";
                std::cout << "\n";
                logFile << "\tmode " << m+1
                        << ": beta=" << beta << " alpha=" << alpha
                        << " kc=" << kc << "\n";
            }
        }
    } // end else (enough DOFs)
    logFile << lt.toc() << " s\n";
}
