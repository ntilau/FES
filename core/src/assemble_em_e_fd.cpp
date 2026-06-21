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
#include "boundary_condition.h"
#include "option.h"

#include <cfloat>
#include <unordered_map>
#include <complex>

void assembler_em_e_fd::assemble(std::ofstream& logFile, eq_sys& sys)
{
    mesh* msh = sys.msh;
    option* opt = sys.opt;

    // 3D curl-curl formulation requires tetrahedral mesh
    if(msh->nTetras == 0)
    {
        throw std::runtime_error("2D mesh requires +formula em_ez_fd, not em_e_fd");
    }

        std::cout << "Auto-detected 3D mesh\n";

    project* prj = sys.prj;
    quad* quadr = sys.quadr;
    read_port_amplitudes(logFile, sys);
    logFile << "% Assembly Symmetric:\n";
    logFile << "In solids: ";
    arma::wall_clock tt, lt;
    tt.tic();
    double k0 = 2.0 * consts::pi * sys.get_freq() / consts::c0;
    double kk = k0*k0;
    sys.set_dofnum(dof(prj).dofnumv);
    std::vector<bool> doftoLeave(sys.get_dofnum(), true);
        std::cout << "FE dof = " << sys.get_dofnum() << " ";
    sys.set_symm_flag(1);
    sys.B_mat().zeros();
    sys.Sol_mat().clear();
    sys.Sp_mat().clear();
    lt.tic();

    // Use map-based assembly to avoid Armadillo SpMat cache non-determinism
    std::map<std::pair<arma::uword, arma::uword>, std::complex<double>> A_map;
    for(size_t id = 0; id < msh->nTetras; id++)
    {
        mtrl* cmtrl = &(msh->tetmtrl[msh->tetLab(id)]);
        ele_mat lMat(opt->p_ord, 3, msh->tet_geo(id), quadr, cmtrl, shape::hcurl);
        dof cdof(prj, 3, id);
        for(int i=0; i<cdof.v.n_rows; i++)
        {
            for(int j=0; j<cdof.v.n_rows; j++)
            {
                if(cdof.v(i)<=cdof.v(j))
                {
                    A_map[{cdof.v(i), cdof.v(j)}] += lMat.S(i,j) + k0*lMat.Z(i,j) - kk*lMat.T(i,j);
                }
            }
        }
    }
    // Build SpMat from map
    sys.A_mat() = build_sparse(A_map, sys.get_dofnum());
    A_map.clear();
    // Now that bulk assembly is in A, do boundary assembly via SpMat
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
        case bc::lumped_port:
            // Lumped port: handled via TFE modal reduction like a waveport.
            // The metal connection edges are PEC (already in Dirdofv from perfect_e loop),
            // and the dielectric gap acts as PMC (natural bc on remaining edges).
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
                for(int i=0; i<cdof.v.n_rows; i++)
                {
                    for(int j=0; j<cdof.v.n_rows; j++)
                    {
                        if(cdof.v(i)<=cdof.v(j))
                        {
                            sys.A_mat()(cdof.v(i),cdof.v(j)) += std::complex<double>(0.0,k0)*lMat.Tt(i,j)*std::sqrt(epsr/mur);
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
    sys.A_mat().sync();
    sys.set_wave_ports_num(0);
    sys.set_wave_ports_dofnum(0);
    sys.wave_portIds_vec().reset();
    compute_waveport_modes(logFile, sys, msh, opt, prj, quadr, k0, kk, doftoLeave, lt);
    mem_stat::print(logFile);
    sys.A_mat().sync();
    rebuild_sparse(sys.A_mat(), sys.get_dofnum());
    logFile << "Finishing:\n";
    lt.tic();
    if(sys.get_wave_ports_num() > 0)
    {
        {
            arma::cx_mat Meig(sys.get_wave_ports_num(), sys.get_wave_ports_dofnum(), arma::fill::zeros);
            arma::cx_mat MAcoeff(sys.get_wave_ports_num(), sys.get_wave_ports_num(), arma::fill::zeros);
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
                        sys.B_mat()(idx,idx) = std::complex<double>(0.0, 2*k0*consts::z0*opt->power);
                        B_exc(idx,idx) = std::complex<double>(0.0, 2*k0*consts::z0*opt->power);
                        for(size_t j=0; j< bc->mode_vec.n_rows; j++)
                        {
                            Meig(idx,jdx+j) = sqrtBeta * bc->mode_vec(j,i);
                        }
                        idx++;
                    }
                    jdx += bc->mode_vec.n_rows;
                }
                else if(bc->type == bc::lumped_port)
                {
                    // Lumped port: use the computed mode shape (from 2D eigenproblem)
                    // with the specified port impedance instead of free-space impedance.
                    for(size_t i=0; i<bc->num_modes; i++)
                    {
                        // sqrtBeta uses the port impedance instead of z0:
                        // sqrt(j * k0 * Zport * power / mode_beta)
                        std::complex<double> sqrtBeta(std::sqrt(std::complex<double>(0.0, k0 * bc->impedance * opt->power)/bc->mode_beta(i)));
                        // MAcoeff = j * k0 * Zport * power  (port termination impedance)
                        MAcoeff(idx,idx) = std::complex<double>(0.0, k0 * bc->impedance * opt->power);
                        sys.B_mat()(idx,idx) = std::complex<double>(0.0, 2.0 * k0 * bc->impedance * opt->power);
                        B_exc(idx,idx) = std::complex<double>(0.0, 2.0 * k0 * bc->impedance * opt->power);
                        for(size_t j=0; j< bc->mode_vec.n_rows; j++)
                        {
                            Meig(idx,jdx+j) = sqrtBeta * bc->mode_vec(j,i);
                        }
                        idx++;
                    }
                    jdx += bc->mode_vec.n_rows;
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
            sys.set_dofreal(sys.Nonwave_portIds_vec().n_rows + sys.get_wave_ports_num());
                std::cout << "\nSYS dof = " << sys.get_dofreal() << "\n";
            logFile << "\tSYS dof = " << sys.get_dofreal() << ", ";
            sys.B_mat().set_size(sys.get_dofreal(), sys.get_wave_ports_num());
            for(size_t bi=0; bi<sys.get_wave_ports_num(); bi++)
                for(size_t bj=0; bj<sys.get_wave_ports_num(); bj++)
                    if(std::abs(B_exc(bi,bj)) > 0) sys.B_mat()(bi,bj) = B_exc(bi,bj);
            lt.tic();
            std::vector<size_t> nnWPids(sys.Nonwave_portIds_vec().n_rows);
            std::vector<size_t> WPids(sys.wave_portIds_vec().n_rows);
            for(size_t i=0; i<sys.Nonwave_portIds_vec().n_rows; i++)
                nnWPids[i] = sys.Nonwave_portIds_vec()(i);
            for(size_t i=0; i<sys.wave_portIds_vec().n_rows; i++)
                WPids[i] = sys.wave_portIds_vec()(i);

            // Build index maps
            std::unordered_map<size_t, size_t> nnWPids_map, WPids_map;
            for(size_t k=0; k<nnWPids.size(); k++) nnWPids_map[nnWPids[k]] = k;
            for(size_t k=0; k<WPids.size(); k++) WPids_map[WPids[k]] = k;

            size_t wnum = WPids.size(), nnum = nnWPids.size();

            // Step 1: Extract Aii = A(nnWPids, nnWPids) BEFORE zeroing diagonal
            // Use direct element access to avoid iterator/sync issues
            std::vector<arma::uword> aii_rows, aii_cols;
            std::vector<std::complex<double>> aii_vals;
            for(size_t wi = 0; wi < nnum; wi++) {
                size_t ri = nnWPids[wi];
                for(size_t wj = wi; wj < nnum; wj++) {
                    size_t ci = nnWPids[wj];
                    std::complex<double> val = sys.A_mat()(ri, ci);
                    if(val != std::complex<double>(0,0)) {
                        aii_rows.push_back(sys.get_wave_ports_num() + wi);
                        aii_cols.push_back(sys.get_wave_ports_num() + wj);
                        aii_vals.push_back(val);
                    }
                }
            }

            // Step 2: Save diagonal and zero it in A
            arma::cx_vec Adiag_vals(sys.get_dofnum(), arma::fill::zeros);
            for(size_t i=0; i<sys.get_dofnum(); i++)
            {
                Adiag_vals(i) = sys.A_mat()(i,i);
                sys.A_mat()(i,i) *= 0.0;
            }
            sys.A_mat().sync(); // sync after zeroing: moves zeroed diagonal entries to CSC

            // Step 3: Extract A_wp, A_wp_nn, A_nn_wp from zeroed-diagonal A
            // Use direct element access to avoid iterator/sync issues
            arma::cx_mat A_wp(wnum, wnum, arma::fill::zeros);
            arma::cx_mat A_wp_nn(wnum, nnum, arma::fill::zeros);
            arma::cx_mat A_nn_wp(nnum, wnum, arma::fill::zeros);
            arma::cx_mat Ad_wp(wnum, wnum, arma::fill::zeros);
            // A_wp: upper triangle of waveport block (off-diagonal, diagonal is zeroed)
            for(size_t wi = 0; wi < wnum; wi++) {
                for(size_t wj = wi; wj < wnum; wj++) {
                    A_wp(wi, wj) = sys.A_mat()(WPids[wi], WPids[wj]);
                }
            }
            // A_wp_nn and A_nn_wp: full matrix (all entries)
            for(size_t wi = 0; wi < wnum; wi++) {
                for(size_t nj = 0; nj < nnum; nj++) {
                    std::complex<double> v = sys.A_mat()(WPids[wi], nnWPids[nj]);
                    if(v != std::complex<double>(0,0)) A_wp_nn(wi, nj) = v;
                    v = sys.A_mat()(nnWPids[nj], WPids[wi]);
                    if(v != std::complex<double>(0,0)) A_nn_wp(nj, wi) = v;
                }
            }

            // Step 4: Ad_wp from saved diagonal
            for(size_t i=0; i<sys.get_dofnum(); i++) {
                auto wi = WPids_map.find(i);
                if(wi != WPids_map.end())
                    Ad_wp(wi->second, wi->second) = Adiag_vals(i);
            }

            // TFE modal reduction
            arma::cx_mat Attred = Meig * A_wp + Meig * A_wp.t() + Meig * Ad_wp;
            arma::cx_mat top_left = Attred * Meig.t() + MAcoeff;
            arma::cx_mat Atired = Meig * A_wp_nn + Meig * A_nn_wp.t();

            // Add top-left block: use LOWER triangle (row >= col) convention
            // matching the GMM code which zeroes upper triangle: Anew(i,j) *= (i>=j)
            for(size_t i=0; i < sys.get_wave_ports_num(); i++)
                for(size_t j=0; j <= i; j++) {
                    aii_rows.push_back(i);
                    aii_cols.push_back(j);
                    aii_vals.push_back(top_left(i,j));
                }

            // Add off-diagonal block: upper-right only
            for(size_t i=0; i < sys.get_wave_ports_num(); i++)
                for(size_t j=0; j < nnWPids.size(); j++) {
                    aii_rows.push_back(i);
                    aii_cols.push_back(sys.get_wave_ports_num() + j);
                    aii_vals.push_back(Atired(i,j));
                }

            // Build Anew from triplets
            arma::umat locs(2, aii_rows.size());
            for(size_t k=0; k<aii_rows.size(); k++) {
                locs(0,k) = aii_rows[k];
                locs(1,k) = aii_cols[k];
            }
            arma::cx_vec cvals(aii_vals.data(), aii_vals.size());
            eq_sys::mat_row_type Anew(locs, cvals, sys.get_dofreal(), sys.get_dofreal(), true, true);
            std::swap(sys.A_mat(), Anew);
            Anew.zeros();
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
        polEinc /= std::sqrt(2.0);
        sys.set_dofreal(sys.get_dofnum());
            std::cout << "\nSYS dof = " << sys.get_dofreal() << "\n";
        logFile << "\tSYS dof = " << sys.get_dofreal() << ", ";
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
            // Collect row and column indices before modifying
            std::vector<size_t> colsToClear;
            for(auto it = sys.A_mat().begin_row(dirid); it != sys.A_mat().end_row(dirid); ++it)
                colsToClear.push_back(it.col());
            for(size_t col : colsToClear)
                sys.A_mat()(dirid, col) = std::complex<double>(0,0);
            std::vector<size_t> rowsToClear;
            for(auto it = sys.A_mat().begin_col(dirid); it != sys.A_mat().end_col(dirid); ++it)
                rowsToClear.push_back(it.row());
            for(size_t row : rowsToClear)
                sys.A_mat()(row, dirid) = std::complex<double>(0,0);
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
    logFile << " " << lt.toc() << " s\n";
    logFile << "+" << tt.toc() << " s\n";
}

