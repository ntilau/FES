#define ARMA_WARN_LEVEL 0

#include "eigen_solver.h"
#include <stdexcept>
#include "mumps_constants.h"
#include <zmumps_c.h>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <string>

eigen::eigen(arma::cx_mat& SS, arma::cx_mat& TT, size_t& numt, size_t& numz, double& kk, int& num_modes)
{
    int n = SS.n_rows;
    if(n < 5)
    {
        throw std::runtime_error("Not enough dofs on wave_ports");
    }
    int i, j, nmodes=1;
    for(i=1; i<num_modes; i++)
    {
        nmodes++;
    }
    // Build sparse A = SS/kk - TT in MUMPS coordinate format
    mumpsIdz = new ZMUMPS_STRUC_C;
    {
        // Count nonzeros for A
        int nnz = 0;
        for(i=0; i<n; i++) {
            for(j=0; j<n; j++) {
                std::complex<double> val = SS(i,j)/kk - TT(i,j);
                if(val.real() != 0.0 || val.imag() != 0.0) nnz++;
            }
        }
        // Allocate and fill coordinate arrays
        int* irn = new int[nnz];
        int* jcn = new int[nnz];
        mumps_double_complex* a = new mumps_double_complex[nnz];
        int idx = 0;
        for(i=0; i<n; i++) {
            for(j=0; j<n; j++) {
                std::complex<double> val = SS(i,j)/kk - TT(i,j);
                if(val.real() != 0.0 || val.imag() != 0.0) {
                    irn[idx] = i + 1;  // 1-based
                    jcn[idx] = j + 1;
                    a[idx].r = val.real();
                    a[idx].i = val.imag();
                    idx++;
                }
            }
        }
        // Initialize MUMPS instance
        mumpsIdz->job = mumps_job_init;
        mumpsIdz->par = 1;
        mumpsIdz->sym = 0;  // unsymmetric
        mumpsIdz->comm_fortran = mumps_comm_world;
        zmumps_c(mumpsIdz);
        // Set matrix data
        mumpsIdz->n = n;
        mumpsIdz->nz = nnz;
        mumpsIdz->irn = irn;
        mumpsIdz->jcn = jcn;
        mumpsIdz->a = a;
        mumpsIdz->icntl[0] = MUMPS_ICNTL_ERRORS;
        mumpsIdz->icntl[1] = MUMPS_ICNTL_DIAG;
        mumpsIdz->icntl[2] = MUMPS_ICNTL_GLOBAL;
        mumpsIdz->icntl[3] = MUMPS_ICNTL_MEMORY;
        // Analyze + Factorize
        mumpsIdz->job = MUMPS_JOB_ANALYZE_FACTOR;
        zmumps_c(mumpsIdz);
        // Free matrix arrays (MUMPS has internal copies)
        delete[] irn;
        delete[] jcn;
        delete[] a;
        // Pre-allocate RHS workspace for op() calls
        mumpsIdz->nrhs = 1;
        mumpsIdz->lrhs = n;
        mumpsIdz->rhs = new mumps_double_complex[n];
    }
    // Store TT and kk for RHS computation in op()
    mumpsTT = TT;
    mumpsKk = kk;
    // Orthogonalization (Zhu Cangellaris p 265, eq 9.74)
    {
        int numtot = numt + numz;
        arma::cx_mat OO(numtot, numtot);
        OO.fill(0);
        OO(arma::span(0,numt-1),arma::span(0,numt-1)) = arma::eye<arma::cx_mat>(numt,numt);
        if(numz > 0) {
            OO(arma::span(numt,numtot-1),arma::span(0,numt-1)) = arma::solve(
                -TT(arma::span(numt,numtot-1),arma::span(numt,numtot-1)),
                 TT(arma::span(numt,numtot-1),arma::span(0,numt-1)));
        }
        mumpsOO = OO;
    }
    // ARPACK eigenvalue solve (calls op() which uses MUMPS for solve)
    // Use Armadillo column-major matrix for Evecs — pass memptr() to znaupd
    arma::cx_vec Evals_arma(nmodes);
    arma::cx_mat Evecs_arma(n, nmodes);
    znaupd(n, nmodes, Evals_arma.memptr(), Evecs_arma.memptr(), kk);
    mode_beta.resize(nmodes);
    for(i=0; i<nmodes; i++)
    {
        mode_beta(i) = Evals_arma(nmodes-1-i);
    }
    mode_vec.resize(n,nmodes);
    // ARPACK returns vectors in Fortran column-major order, one column per eigenvector.
    // Evecs_arma data pointer points to v[0] which is column-major, so just copy columns.
    for(j=0; j<nmodes; j++)
    {
        for(i=0; i<n; i++)
        {
            mode_vec(i,j) = Evecs_arma(i, nmodes-1-j);
        }
    }
    // ── Debug diagnostics ──
    {
        // kk = shift ≈ -k₀² (negative! shift = -k₀²·εr·μr from factory.cpp).
        // ARPACK mode 3 (bmat="I", σ=kk) finds eigenvalues of A⁻¹·(TT/kk) near σ.
        // The operator does NOT apply the shift, so effective σ=0.
        // Physical β = sqrt(k₀² − γ) where γ is eigenvalue of (SS − k₀²·TT)·v = γ·B·v.
        // Expected TE10 at 10 GHz for WR90: β≈158 (propagating), γ = k_c²−k₀² ≈ −24955.
        double k0sq = -kk;  // kk is negative, k₀² = -kk for vacuum
        std::cout << "\n[eig dbg] n=" << n << " modes=" << nmodes
                  << " shift=" << kk << " k0²=" << k0sq
                  << " |SS|=" << arma::norm(SS, "fro")
                  << " |TT|=" << arma::norm(TT, "fro") << "\n";
        for(i=0; i<nmodes; i++) {
            std::complex<double> d = Evals_arma(i);
            // For operator A⁻¹·(TT/kk): γ = kk/d (since A⁻¹·(TT/kk)·v = θ·v gives γ·TT·v ≈ γ/d·v)
            std::complex<double> gamma_est = kk / d;
            std::complex<double> beta_phys = std::sqrt(std::complex<double>(k0sq,0) - gamma_est);
            double mode_energy = std::real(arma::as_scalar(mode_vec.col(i).t() * TT * mode_vec.col(i)));
            std::cout << "[eig dbg] mode" << i
                      << " θ=(" << d.real() << "," << d.imag() << ")"
                      << " γ≈(" << gamma_est.real() << "," << gamma_est.imag() << ")"
                      << " β_solver=(" << mode_beta(i).real() << "," << mode_beta(i).imag() << ")"
                      << " β_phys=(" << beta_phys.real() << "," << beta_phys.imag() << ")"
                      << " |mode|²_TT=" << mode_energy
                      << " (expect TE10 β≈" << std::sqrt(k0sq - 18883.0) << ")\n";
        }
    }
}


eigen::~eigen()
{
    mode_vec.clear();
    mumpsTT.clear();
    mumpsOO.clear();
    if(mumpsIdz) {
        mumpsIdz->job = MUMPS_JOB_END;
        zmumps_c(mumpsIdz);
        delete[] mumpsIdz->rhs;
        delete mumpsIdz;
        mumpsIdz = nullptr;
    }
}

void eigen::znaupd(int n, int nev, std::complex<double>* Evals, std::complex<double>* Evecs, double sigma)
{
    double gamtol = 0.0; // gamma tolerance
    int ido = 0;
    char bmat[2] = "I";
    char which[3] = "LM";
    char all[] = "A";
    double tol = 0.0;
    std::vector<std::complex<double>> resid(n);
    // Use a deterministic non-trivial initial vector for ARPACK
    for(int i=0; i<n; i++) resid[i] = std::complex<double>(double(i+1), double(n-i));
    int ncv = 4*nev;
    if(ncv > n) ncv = n;
    int ldv = n;
    std::vector<std::complex<double>> v_buf(ldv * ncv);
    std::complex<double>* v = v_buf.data();
    std::vector<int> iparam_v(11);
    int* iparam = iparam_v.data();
    iparam[0] = 1;
    iparam[2] = 100;
    iparam[3] = 1;
    iparam[6] = 3;
    std::vector<int> ipntr_v(14);
    int* ipntr = ipntr_v.data();
    std::vector<std::complex<double>> workd_buf(3*n);
    std::complex<double>* workd = workd_buf.data();
    int lworkl = 3*ncv*ncv + 5*ncv;
    std::vector<std::complex<double>> workl_buf(lworkl);
    std::complex<double>* workl = workl_buf.data();
    std::vector<double> rwork_buf(ncv);
    double* rwork = rwork_buf.data();
    int info = 1;  // use provided initial vector (deterministic)
    int rvec = 1;
    std::vector<int> select_buf(ncv);
    int* select = select_buf.data();
    std::vector<std::complex<double>> d_buf(ncv);
    std::complex<double>* d = d_buf.data();
    std::vector<std::complex<double>> workev_buf(3*ncv);
    std::complex<double>* workev = workev_buf.data();
    int ierr;
    int iter = 0;
    do
    {
        znaupd_(&ido, bmat, &n, which, &nev, &tol, resid.data(),
                &ncv, v, &ldv, iparam, ipntr, workd, workl,
                &lworkl, rwork, &info);
        if((ido==1)||(ido==-1))
        {
            op(n, workd+ipntr[0]-1, workd+ipntr[1]-1);
        }
        iter++;
    }
    while((ido==1)||(ido==-1));
    std::cout <<  "(" << iter << ")";
    if(info<0)
    {
        throw std::runtime_error("znaupd error = " + std::to_string(info));
    }
    else
    {
        zneupd_(&rvec, all, select, d, v, &ldv, &sigma, workev,
                bmat, &n, which, &nev, &tol, resid.data(), &ncv, v, &ldv,
                iparam, ipntr, workd, workl, &lworkl, rwork, &ierr);
        if(ierr!=0)
        {
            std::cout << info;
        }
        int i, j;
        for(i=0; i<nev; i++)
        {
            Evals[i] = std::sqrt(d[i]);
            double real = std::abs(std::real(Evals[i]));
            double imag = std::abs(std::imag(Evals[i]));
            Evals[i] = std::complex<double>(real < gamtol ? 0.0 : real, imag < gamtol ? 0.0 : imag);
        }
        // ARPACK returns v as Fortran column-major (ncv columns, each of length n).
        // Evecs is also column-major (from arma::cx_mat.memptr()), so copy directly.
        // v[j * n + i] = Evecs(i, j) for column j, row i
        for(j=0; j<nev; j++)
        {
            std::copy(v + j*n, v + (j+1)*n, Evecs + j*n);
        }
        // Sort eigenvalues and vectors by real part, then imaginary part
        std::complex<double> temp;
        for(i=0; i<nev; i++)
        {
            for(j=i; j<nev; j++)
            {
                if(Evals[j].real() < Evals[i].real())
                {
                    std::swap(Evals[i], Evals[j]);
                    for(int k=0; k<n; k++)
                        std::swap(Evecs[i*n + k], Evecs[j*n + k]);
                }
            }
        }
        for(i=0; i<nev; i++)
        {
            for(j=i; j<nev; j++)
            {
                if(Evals[j].imag() < Evals[i].imag())
                {
                    std::swap(Evals[i], Evals[j]);
                    for(int k=0; k<n; k++)
                        std::swap(Evecs[i*n + k], Evecs[j*n + k]);
                }
            }
        }
    }
}

inline void eigen::op(int n, std::complex<double>* in, std::complex<double>* out)
{
    // Compute RHS: b = (TT/kk) * in
    arma::cx_vec v(n);
    for(int i=0; i<n; i++) v(i) = in[i];
    arma::cx_vec b = mumpsTT * v;
    b /= mumpsKk;
    // Set MUMPS RHS
    for(int i=0; i<n; i++) {
        mumpsIdz->rhs[i].r = b(i).real();
        mumpsIdz->rhs[i].i = b(i).imag();
    }
    // Solve A * x = b using existing MUMPS factor
    mumpsIdz->nrhs = 1;
    mumpsIdz->lrhs = n;
    mumpsIdz->job = MUMPS_JOB_SOLVE;
    zmumps_c(mumpsIdz);
    // Extract solution
    arma::cx_vec x(n);
    for(int i=0; i<n; i++) x(i) = std::complex<double>(mumpsIdz->rhs[i].r, mumpsIdz->rhs[i].i);
    // Apply orthogonalization: out = OO * x
    arma::cx_vec y = mumpsOO * x;
    for(int i=0; i<n; i++) out[i] = y(i);
}

