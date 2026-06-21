#include "solver.h"
#include <stdexcept>
#include <type_traits>
#include "equation_system.h"
#include "mesh.h"
#include "boundary_condition.h"
#include "option.h"
#include "constants.h"
#include "memory.h"
#include "gmres.h"

#include <smumps_c.h>
#include <dmumps_c.h>
#include <cmumps_c.h>
#include <zmumps_c.h>
#include "mumps_constants.h"

#include <iomanip>
#include <armadillo>
#include <cfloat>

// ── Factory ──

std::unique_ptr<solver> solver::create(const option& opt)
{
    switch(opt.solver) {
    case option::direct: return std::make_unique<mumps_solver>();
    case option::gmres:  return std::make_unique<gmres_solver>();
    default:
        throw std::runtime_error("solver not implemented yet");
    }
}

// ════════════════════════════════════════════════════════════
// ExtractSolution — shared by all solvers
// ════════════════════════════════════════════════════════════

void solver::extract_solution_full(eq_sys& sys, int n, void* rhs, int col, bool isDouble)
{
    mesh* msh = sys.msh;
    option* opt = sys.opt;
    auto read = [&](int idx) -> std::complex<double> {
        if(isDouble) {
            auto* z = static_cast<ZMUMPS_COMPLEX*>(rhs);
            return {z[idx].r, z[idx].i};
        } else {
            auto* c = static_cast<CMUMPS_COMPLEX*>(rhs);
            return {c[idx].r, c[idx].i};
        }
    };

    int shift = col * n;
    if(sys.get_wave_ports_num() > 0)
    {
        {
            for(size_t row = 0; row < sys.get_wave_ports_num(); row++)
                sys.Sp_mat()(row,col) = read(shift + row);
            std::complex<double> jk0z0(0.0, 2.0*consts::pi*sys.get_freq()/consts::c0*consts::z0*opt->power);
            for(size_t i = 0; i < sys.Nonwave_portIds_vec().n_rows; i++)
                if(sys.Nonwave_portIds_vec()(i) < UINT_MAX)
                    sys.Sol_mat()(sys.Nonwave_portIds_vec()(i),col) = read(shift + i + sys.get_wave_ports_num());
            int idx = 0;
            if(opt->nl)
            {
                for(size_t bcid = 0; bcid < msh->facbc.size(); bcid++)
                {
                    bc* bc = &(msh->facbc[bcid]);
                    if(bc->type == bc::wave_port)
                    {
                        for(size_t ih = 0; ih < opt->n_harm; ih++)
                            for(size_t i = 0; i < bc->num_modes; i++)
                                for(size_t j = 0; j < bc->mode_vec.n_rows; j++)
                                    sys.Sol_mat()(bc->mode_vecdof(j,ih),col) += std::sqrt(jk0z0/bc->mode_beta(ih*bc->num_modes+i)) * bc->mode_vec(j,ih*bc->num_modes+i) * sys.Sp_mat()(idx,col);
                        idx++;
                    }
                }
            }
            else
            {
                for(size_t bcid = 0; bcid < msh->facbc.size(); bcid++)
                {
                    bc* bc = &(msh->facbc[bcid]);
                    if(bc->type == bc::wave_port) {
                        for(size_t i = 0; i < bc->num_modes; i++)
                            for(size_t j = 0; j < bc->mode_vec.n_rows; j++)
                                sys.Sol_mat()(bc->mode_vecdof(j),col) += std::sqrt(jk0z0/bc->mode_beta(i)) * bc->mode_vec(j,i) * sys.Sp_mat()(idx,col);
                        idx++;
                    }
                    else if(bc->type == bc::lumped_port) {
                        // Lumped port: use port impedance Zport instead of z0
                        std::complex<double> jk0Zport(0.0, 2.0*consts::pi*sys.get_freq()/consts::c0 * bc->impedance * opt->power);
                        for(size_t i = 0; i < bc->num_modes; i++)
                            for(size_t j = 0; j < bc->mode_vec.n_rows; j++)
                                sys.Sol_mat()(bc->mode_vecdof(j),col) += std::sqrt(jk0Zport/bc->mode_beta(i)) * bc->mode_vec(j,i) * sys.Sp_mat()(idx,col);
                        idx++;
                    }
                }
            }
        }
    }
    else if(opt->einc || opt->assembly == option::em_e_qs || (sys.get_wave_ports_num() == 0 && sys.B_mat().n_nonzero > 0))
    {
        for(size_t row = 0; row < (size_t)n; row++)
            sys.Sol_mat()(row,0) = read(shift + row);
    }
}

// ════════════════════════════════════════════════════════════
// Jacobi-preconditioned gmres (no DD)
// ════════════════════════════════════════════════════════════

namespace {

void Jacobigmres(const arma::SpMat<std::complex<double>>& A,
                        const arma::cx_vec& diag,
                        arma::cx_vec& x,
                        const std::vector<std::complex<double>>& Brhs,
                        int restart, iteration& outer)
{
    size_t n = A.n_rows;
    std::vector<arma::cx_vec> V(restart+1);
    for(auto& v : V) v.set_size(n);
    arma::cx_mat H(restart+1, restart, arma::fill::zeros);
    arma::cx_vec s(restart+1);
    std::vector<std::complex<double>> c_rot(restart+1), g_rot(restart+1);

    auto applyA = [&](const arma::cx_vec& v) -> arma::cx_vec {
        arma::cx_vec w = A * v + A.t() * v;
        for(size_t j = 0; j < n; j++) w(j) -= diag(j) * v(j);
        return w;
    };

    arma::cx_vec r(n, arma::fill::zeros);
    for(size_t i = 0; i < n; i++) r(i) = Brhs[i];
    if(arma::norm(x, 2) > 0) {
        arma::cx_vec Ax = applyA(x);
        for(size_t i = 0; i < n; i++) r(i) -= Ax(i);
    }
    for(size_t i = 0; i < n; i++) r(i) /= diag(i);

    double beta = arma::norm(r, 2);
    outer.set_rhsnorm(beta);
    if(beta == 0) { x.zeros(); return; }

    int itnum = 0;
    while(!outer.finished(beta)) {
        V[0] = r / beta;
        s.zeros(); s(0) = std::complex<double>(beta, 0);
        size_t i = 0;
        do {
            arma::cx_vec w = applyA(V[i]);
            for(size_t j = 0; j < n; j++) w(j) /= diag(j);
            V[i+1] = w;
            for(size_t k = 0; k <= i; k++) {
                std::complex<double> h = arma::cdot(V[k], V[i+1]);
                H(k,i) = h;
                V[i+1] -= h * V[k];
            }
            double a = arma::norm(V[i+1], 2);
            H(i+1,i) = std::complex<double>(a, 0);
            V[i+1] /= a;
            for(size_t k = 0; k < i; k++) {
                ApplyGivensLeft(H(k,i), H(k+1,i), c_rot[k], g_rot[k]);
            }
            GivensRotation(H(i,i), H(i+1,i), c_rot[i], g_rot[i]);
            ApplyGivensLeft(H(i,i), H(i+1,i), c_rot[i], g_rot[i]);
            ApplyGivensLeft(s(i), s(i+1), c_rot[i], g_rot[i]);
            ++outer; ++i;
        } while(!outer.finished(std::abs(s(i))) && i < (size_t)restart);
        for(size_t k = i; k-- > 0; ) {
            s(k) /= H(k,k);
            for(size_t j = 0; j < k; j++) s(j) -= H(j,k) * s(k);
        }
        for(size_t k = 0; k < i; k++) x += s(k) * V[k];
        r.zeros();
        for(size_t j = 0; j < n; j++) r(j) = Brhs[j];
        arma::cx_vec Ax = applyA(x);
        for(size_t j = 0; j < n; j++) r(j) -= Ax(j);
        for(size_t j = 0; j < n; j++) r(j) /= diag(j);
        beta = arma::norm(r, 2);
        itnum++;
    }
}

} // anonymous namespace

// ════════════════════════════════════════════════════════════
// MUMPS solver
// ════════════════════════════════════════════════════════════

void mumps_solver::solve(eq_sys& sys, std::ofstream& log)
{
    if(sys.opt->dbl)
        solveDispatch<ZMUMPS_STRUC_C, ZMUMPS_COMPLEX, ZMUMPS_REAL, zmumps_c>(sys, log, "Double");
    else
        solveDispatch<CMUMPS_STRUC_C, CMUMPS_COMPLEX, CMUMPS_REAL, cmumps_c>(sys, log, "Single");
}

// RAII wrapper for a MUMPS instance — allocaates, initializes, and cleans up
// a MUMPS_STRUC, including its internal arrays (irn, jcn, a, rhs, etc.).
template<typename MUMPS_STRUC, typename MUMPS_COMPLEX, typename MUMPS_REAL, void (*MUMPS_FUNC)(MUMPS_STRUC*)>
struct mumps_handle {
    MUMPS_STRUC* idz;
    bool owned_arrays;

    mumps_handle(int sym_flag, int dofreal)
        : idz(new MUMPS_STRUC), owned_arrays(false)
    {
        idz->job = mumps_job_init;
        idz->par = 1;
        idz->sym = (MUMPS_INT) sym_flag;
        idz->comm_fortran = mumps_comm_world;
        MUMPS_FUNC(idz);
        idz->n = (MUMPS_INT) dofreal;
    }

    ~mumps_handle() {
        if(!idz) return;
        if(owned_arrays) {
            delete[] idz->a;
            delete[] idz->irn;
            delete[] idz->jcn;
            delete[] idz->rhs;
            delete[] idz->rhs_sparse;
            delete[] idz->irhs_sparse;
            delete[] idz->irhs_ptr;
        }
        idz->job = MUMPS_JOB_END;
        MUMPS_FUNC(idz);
        delete idz;
    }

    mumps_handle(const mumps_handle&) = delete;
    mumps_handle& operator=(const mumps_handle&) = delete;

    void allocate_rhs(int nz_rhs_val, int nrhs_val, int lrhs_val) {
        idz->nrhs = (MUMPS_INT) nrhs_val;
        idz->lrhs = (MUMPS_INT) lrhs_val;
        idz->nz_rhs = (MUMPS_INT) nz_rhs_val;
        if(!owned_arrays) {
            idz->irn = new MUMPS_INT [idz->nz];
            idz->jcn = new MUMPS_INT [idz->nz];
            idz->a   = new MUMPS_COMPLEX [idz->nz];
        }
        idz->rhs        = new MUMPS_COMPLEX [idz->lrhs * idz->nrhs];
        idz->rhs_sparse = new MUMPS_COMPLEX [idz->nz_rhs];
        idz->irhs_sparse= new MUMPS_INT [idz->nz_rhs];
        idz->irhs_ptr   = new MUMPS_INT [idz->nrhs + 1];
        owned_arrays = true;
    }
};

template<typename MUMPS_STRUC, typename MUMPS_COMPLEX, typename MUMPS_REAL, void (*MUMPS_FUNC)(MUMPS_STRUC*)>
void mumps_solver::solveDispatch(eq_sys& sys, std::ofstream& log, const char* label)
{
    option* opt = sys.opt;
    log << "% " << label << " precision Complex solver";
        std::cout << label << " precision Complex solver";
    sys.tt_clock().tic();
    {
        switch(sys.get_symm_flag()) {
        case 0: log << " / Non-symmetric\n";  std::cout << " / Non-symmetric\n";  break;
        case 1: log << " / Symmetric\n";      std::cout << " / Symmetric\n";      break;
        case 2: log << " / General symmetric\n"; std::cout << " / General symmetric\n"; break;
        }
    }
    mumps_handle<MUMPS_STRUC, MUMPS_COMPLEX, MUMPS_REAL, MUMPS_FUNC> mh(sys.get_symm_flag(), (int)sys.get_dofreal());
    mem_stat::AvailableMemory(std::cout);
    mh.idz->nz = (MUMPS_INT) sys.A_mat().n_nonzero;
    mh.idz->irn = new MUMPS_INT [mh.idz->nz];
    mh.idz->jcn = new MUMPS_INT [mh.idz->nz];
    mh.idz->a = new MUMPS_COMPLEX [mh.idz->nz];
    mh.owned_arrays = true;
    {
        MUMPS_INT idx = 0;
        for(auto it = sys.A_mat().begin(); it != sys.A_mat().end(); ++it) {
            mh.idz->a[idx].r = (MUMPS_REAL) (*it).real();
            mh.idz->a[idx].i = (MUMPS_REAL) (*it).imag();
            mh.idz->irn[idx] = (MUMPS_INT) it.row() + 1;
            mh.idz->jcn[idx++] = (MUMPS_INT) it.col() + 1;
        }
    }
    mh.idz->icntl[0] = MUMPS_ICNTL_ERRORS;
    mh.idz->icntl[1] = MUMPS_ICNTL_DIAG;
    mh.idz->icntl[2] = MUMPS_ICNTL_GLOBAL;
    mh.idz->icntl[3] = MUMPS_ICNTL_MEMORY;
    mh.idz->icntl[19] = MUMPS_ICNTL_RHS_SPARSE;
    mh.allocate_rhs((int)sys.B_mat().n_nonzero, (int)sys.B_mat().n_cols, (int)sys.B_mat().n_rows);
    {
        MUMPS_INT i = 0, idx = 0;
        for(auto it = sys.B_mat().begin(); it != sys.B_mat().end(); ) {
            MUMPS_INT col = (MUMPS_INT) it.col();
            mh.idz->irhs_ptr[i++] = idx + 1;
            while(it != sys.B_mat().end() && (MUMPS_INT) it.col() == col) {
                mh.idz->irhs_sparse[idx] = (MUMPS_INT) it.row() + 1;
                mh.idz->rhs_sparse[idx].r = (MUMPS_REAL) (*it).real();
                mh.idz->rhs_sparse[idx++].i = (MUMPS_REAL) (*it).imag();
                ++it;
            }
        }
        mh.idz->irhs_ptr[mh.idz->nrhs] = idx + 1;
    }
    {
        eq_sys::mat_row_type Atmp(mh.idz->n, mh.idz->n);
        eq_sys::mat_col_type Btmp(mh.idz->n, mh.idz->nrhs);
        std::swap(sys.A_mat(), Atmp);
        std::swap(sys.B_mat(), Btmp);
    }
    mem_stat::print(std::cout);
    mh.idz->job = MUMPS_JOB_FACTOR_SOLVE;
    std::cout << "Factor and Solve ";
    sys.lt_clock().tic();
    MUMPS_FUNC(mh.idz);
    std::cout << "+ " << mh.idz->info[14] << " MB ";
    log << "\tCommit memory: " << mh.idz->info[14] << " MB\n";
    log << "\tFactor and Solve: " << sys.lt_clock().toc() << " s\n";
    if(opt->nl) {
        sys.Sol_mat().resize(sys.get_dofnum()*opt->n_harm, 1);
        sys.Sp_mat().resize(sys.get_wave_ports_num(), 1);
    } else {
        sys.Sp_mat().resize(mh.idz->nrhs, mh.idz->nrhs);
        sys.Sol_mat().resize(sys.get_dofnum(), mh.idz->nrhs);
    }
    sys.Sol_mat().fill(0);
    sys.Sp_mat().fill(0);
    for(size_t col = 0; col < sys.B_mat().n_cols; col++)
        extract_solution_full(sys, mh.idz->n, (void*)mh.idz->rhs, col, std::is_same<MUMPS_REAL, ZMUMPS_REAL>::value);
    log << "+" << sys.tt_clock().toc() << " s\n";
    std::cout << sys.tt_clock().toc() << " s\n";
}

// Explicit template instantiations
template void mumps_solver::solveDispatch<ZMUMPS_STRUC_C, ZMUMPS_COMPLEX, ZMUMPS_REAL, zmumps_c>(eq_sys&, std::ofstream&, const char*);
template void mumps_solver::solveDispatch<CMUMPS_STRUC_C, CMUMPS_COMPLEX, CMUMPS_REAL, cmumps_c>(eq_sys&, std::ofstream&, const char*);

// ════════════════════════════════════════════════════════════
// gmres solver
// ════════════════════════════════════════════════════════════

void gmres_solver::solve(eq_sys& sys, std::ofstream& log)
{
    option* opt = sys.opt;
    mesh* msh = sys.msh;
    log << "% gmres solution:\n";
    std::cout << "gmres: ";
    sys.tt_clock().tic();
    if(opt->einc || opt->assembly == option::em_e_qs) {
        sys.Sol_mat().resize(sys.get_dofnum(), 1);
        sys.Sol_mat().fill(0);
    } else {
        sys.Sp_mat().resize(sys.get_wave_ports_num(), sys.get_wave_ports_num());
        sys.Sp_mat().fill(0);
        sys.Sol_mat().resize(sys.get_dofnum(), sys.get_wave_ports_num());
        sys.Sol_mat().fill(0);
    }
    if(sys.PR_mat().n_nonzero == 0)
    {
        arma::cx_vec Pdiag(sys.get_dofreal());
        for(size_t i = 0; i < sys.get_dofreal(); i++) Pdiag(i) = sys.A_mat()(i,i);
        std::cout << " " << sys.lt_clock().toc() << " s\n";

        if(sys.get_wave_ports_num() > 0)
        {
            for(size_t col = 0; col < sys.get_wave_ports_num(); col++) {
                arma::cx_vec X(sys.get_dofreal(), arma::fill::zeros);
                std::vector<std::complex<double>> Brhs(sys.get_dofreal());
                for(size_t row = 0; row < sys.get_dofreal(); row++) Brhs[row] = sys.B_mat()(row,col);
                iteration iter(opt->toll, 1000);
                Jacobigmres(sys.A_mat(), Pdiag, X, Brhs, opt->niter, iter);
                std::cout << "Iters " << col << " ~ " << iter.get_iteration() << "\n";
                {
                    for(size_t row = 0; row < sys.get_wave_ports_num(); row++) sys.Sp_mat()(row,col) = X[row];
                    std::complex<double> jk0z0(0.0, 2.0*consts::pi*sys.get_freq()/consts::c0*consts::z0*opt->power);
                    for(size_t i=0; i<sys.Nonwave_portIds_vec().n_rows; i++)
                        if(sys.Nonwave_portIds_vec()(i) < UINT_MAX) sys.Sol_mat()(sys.Nonwave_portIds_vec()(i),col) = X[sys.get_wave_ports_num()+i];
                    size_t idx = 0;
                    if(opt->nl) {
                        for(size_t bcid=0; bcid<msh->facbc.size(); bcid++) {
                            bc* bc=&(msh->facbc[bcid]);
                            if(bc->type==bc::wave_port)
                                for(size_t ih=0; ih<opt->n_harm; ih++)
                                    for(size_t i=0; i<bc->num_modes; i++)
                                        for(size_t j=0; j<bc->mode_vec.n_rows; j++)
                                            sys.Sol_mat()(bc->mode_vecdof(j,ih),col) += std::sqrt(jk0z0/bc->mode_beta(ih*bc->num_modes+i))*bc->mode_vec(j,ih*bc->num_modes+i)*sys.Sp_mat()(idx,col);
                            else if(bc->type==bc::lumped_port)
                                for(size_t j=0; j<bc->mode_vec.n_rows; j++)
                                    sys.Sol_mat()(bc->mode_vecdof(j),col) += std::sqrt(jk0z0/bc->mode_beta(0))*bc->mode_vec(j)*sys.Sp_mat()(idx,col);
                            idx++;
                        }
                    } else {
                        for(size_t bcid=0; bcid<msh->facbc.size(); bcid++) {
                            bc* bc=&(msh->facbc[bcid]);
                            if(bc->type==bc::wave_port)
                                for(size_t i=0; i<bc->num_modes; i++)
                                    for(size_t j=0; j<bc->mode_vec.n_rows; j++)
                                        sys.Sol_mat()(bc->mode_vecdof(j),col) += std::sqrt(jk0z0/bc->mode_beta(i))*bc->mode_vec(j,i)*sys.Sp_mat()(idx,col);
                            else if(bc->type==bc::lumped_port)
                                for(size_t j=0; j<bc->mode_vec.n_rows; j++)
                                    sys.Sol_mat()(bc->mode_vecdof(j),col) += std::sqrt(jk0z0/bc->mode_beta(0))*bc->mode_vec(j)*sys.Sp_mat()(idx,col);
                            idx++;
                        }
                    }
                }
            }
        }
        else if(opt->einc || opt->assembly == option::em_e_qs) {
            std::vector<std::complex<double>> Brhs2(sys.get_dofreal());
            for(size_t row = 0; row < sys.get_dofreal(); row++) Brhs2[row] = sys.B_mat()(row,0);
            iteration iter2(opt->toll, opt->niter);
            arma::cx_vec X2(sys.get_dofreal(), arma::fill::zeros);
            Jacobigmres(sys.A_mat(), Pdiag, X2, Brhs2, opt->niter, iter2);
            std::cout << "Iters  ~ " << iter2.get_iteration() << "\n";
            for(size_t row = 0; row < sys.get_dofreal(); row++) sys.Sol_mat()(row,0) = X2(row);
        }
    }
    else     // DD Preconditioned gmres
    {
        if(sys.get_wave_ports_num() > 0) {
                for(size_t col = 0; col < sys.get_wave_ports_num(); col++) {
                    std::vector<std::complex<double>> X(sys.get_dofreal()), Brhs(sys.get_dofreal());
                    iteration iter(opt->toll, 0, -1, 10);
                    for(size_t row = 0; row < sys.get_dofreal(); row++) Brhs[row] = sys.B_mat()(row,col);
                    gmres(sys.A_mat(), sys.PR_mat(), X, Brhs, sys.doflevel_vec(), opt->niter, iter, log, opt);
                    std::cout << "Iters " << col << " ~ " << iter.get_iteration() << "\n";
                    {
                        for(size_t row = 0; row < sys.get_wave_ports_num(); row++) sys.Sp_mat()(row,col) = X[row];
                        std::complex<double> jk0z0(0.0, 2.0*consts::pi*sys.get_freq()/consts::c0*consts::z0*opt->power);
                        for(size_t i=0; i<sys.Nonwave_portIds_vec().n_rows; i++)
                            if(sys.Nonwave_portIds_vec()(i) < UINT_MAX) sys.Sol_mat()(sys.Nonwave_portIds_vec()(i),col) = X[sys.get_wave_ports_num()+i];
                        size_t idx = 0;
                        if(opt->nl) {
                            for(size_t bcid=0; bcid<msh->facbc.size(); bcid++) {
                                bc* bc=&(msh->facbc[bcid]);
                                if(bc->type==bc::wave_port)
                                    for(size_t ih=0; ih<opt->n_harm; ih++)
                                        for(size_t i=0; i<bc->num_modes; i++)
                                            for(size_t j=0; j<bc->mode_vec.n_rows; j++)
                                                sys.Sol_mat()(bc->mode_vecdof(j,ih),col) += std::sqrt(jk0z0/bc->mode_beta(ih*bc->num_modes+i))*bc->mode_vec(j,ih*bc->num_modes+i)*sys.Sp_mat()(idx,col);
                                else if(bc->type==bc::lumped_port)
                                    for(size_t j=0; j<bc->mode_vec.n_rows; j++)
                                        sys.Sol_mat()(bc->mode_vecdof(j),col) += std::sqrt(jk0z0/bc->mode_beta(0))*bc->mode_vec(j)*sys.Sp_mat()(idx,col);
                                idx++;
                            }
                        } else {
                            for(size_t bcid=0; bcid<msh->facbc.size(); bcid++) {
                                bc* bc=&(msh->facbc[bcid]);
                                if(bc->type==bc::wave_port)
                                    for(size_t i=0; i<bc->num_modes; i++)
                                        for(size_t j=0; j<bc->mode_vec.n_rows; j++)
                                            sys.Sol_mat()(bc->mode_vecdof(j),col) += std::sqrt(jk0z0/bc->mode_beta(i))*bc->mode_vec(j,i)*sys.Sp_mat()(idx,col);
                                else if(bc->type==bc::lumped_port)
                                    for(size_t j=0; j<bc->mode_vec.n_rows; j++)
                                        sys.Sol_mat()(bc->mode_vecdof(j),col) += std::sqrt(jk0z0/bc->mode_beta(0))*bc->mode_vec(j)*sys.Sp_mat()(idx,col);
                                idx++;
                            }
                        }
                    }
                }
            }
        else if(opt->einc || opt->assembly == option::em_e_qs) {
            std::vector<std::complex<double>> X(sys.get_dofreal()), Brhs(sys.get_dofreal());
            iteration iter(opt->toll, 0, -1, 10);
            for(size_t row = 0; row < sys.get_dofreal(); row++) Brhs[row] = sys.B_mat()(row,0);
            gmres(sys.A_mat(), sys.PR_mat(), X, Brhs, sys.doflevel_vec(), opt->niter, iter, log, opt);
            std::cout << "Iters ~ " << iter.get_iteration() << "\n";
            for(size_t row = 0; row < sys.get_dofreal(); row++)
                if(sys.Invdofmapv_vec()(row) < UINT_MAX) sys.Sol_mat()(sys.Invdofmapv_vec()(row),0) = X[row];
        }
    }
    log << "+" << sys.tt_clock().toc() << " s\n";
    std::cout << sys.tt_clock().toc() << " s\n";
}
