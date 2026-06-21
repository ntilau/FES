#ifndef gmres_H
#define gmres_H

#include <armadillo>
#include <cmumps_c.h>
#include "mumps_constants.h"
#include <zmumps_c.h>
#include "option.h"

#include <complex>
#include <vector>
#include <cmath>
#include <limits>
#include <iostream>

// Replacement for gmm::iteration
struct iteration {
    double tol;
    int maxiter;
    int iter;
    double res, rhsnorm;
    int noisy;

    iteration(double t, int mi = 0) : tol(t), maxiter(mi), iter(0), res(0), rhsnorm(1.0), noisy(0) {}
    iteration(double t, int mi, int, int n) : tol(t), maxiter(mi), iter(0), res(0), rhsnorm(1.0), noisy(n) {}

    bool finished(double r) {
        res = std::abs(r);
        return (iter >= maxiter) || (res <= tol * rhsnorm);
    }
    void set_rhsnorm(double n)      { rhsnorm = n; }
    double get_rhsnorm() const      { return rhsnorm; }
    double get_res() const          { return res; }
    int    get_iteration() const    { return iter; }
    void   init()                   { iter = 0; }
    void   operator++(int)          { iter++; }
    iteration& operator++()         { iter++; return *this; }
    void   set_maxiter(int m)       { maxiter = m; }
    void   reduce_noisy()           { noisy = 0; }
    int    get_noisy() const        { return noisy; }
    void   set_name(const char*)    { }
    void   set_noisy(int n)         { noisy = n; }
};

// Givens rotation helpers for complex gmres
// Computes c (real) and s (complex) such that:
// [ c     s ] [a] = [r]
// [-conj(s) c] [b]   [0]
// where r = sqrt(|a|^2 + |b|^2), c = |a|/r, s = (a/|a|)*(b/r)
template<typename T>
inline void GivensRotation(T a, T b, T& c, T& s) {
    using R = typename T::value_type;
    R zero = R(0);
    if(std::abs(b) == zero) {
        c = T(1.0, zero); s = T(zero, zero);
        return;
    }
    R abs_a = std::abs(a);
    R abs_b = std::abs(b);
    R r = std::sqrt(abs_a*abs_a + abs_b*abs_b);
    c = T(abs_a / r, zero);
    if(abs_a == zero) {
        s = T(abs_b / r, zero);
    } else {
        s = (a / T(abs_a, zero)) * T(abs_b / r, zero);
    }
}

template<typename T>
inline void ApplyGivensLeft(T& x, T& y, const T& c, const T& s) {
    T temp = c * x + s * y;
    y = -std::conj(s) * x + c * y;
    x = temp;
}

template<typename Mat, typename Vec>
inline void UpperTriSolve(const Mat& H, Vec& s, size_t n, bool) {
    for(size_t k = n; k-- > 0; ) {
        s(k) /= H(k, k);
        for(size_t j = 0; j < k; j++) {
            s(j) -= H(j, k) * s(k);
        }
    }
}

inline void Combine(const std::vector<arma::cx_vec>& KS, const arma::cx_vec& s,
                    arma::cx_vec& x, size_t n) {
    for(size_t k = 0; k <= n; k++) {
        x += s(k) * KS[k];
    }
}

// Modified Gram-Schmidt orthogonalization
inline void Orthogonalize(std::vector<arma::cx_vec>& KS,
                          arma::cx_vec& Hcol, size_t n) {
    for(size_t k = 0; k <= n; k++) {
        std::complex<double> dot = arma::cdot(KS[k], KS[n+1]);
        Hcol(k) = dot;
        KS[n+1] -= dot * KS[k];
    }
}

// ============== DD Preconditioners ==============
// Single block solve factorized by MUMPS type parameters.

template <typename Matrix, typename VecI, typename V1, typename V2,
          typename MUMPS_STRUC, typename MUMPS_COMPLEX, typename MUMPS_REAL,
          void (*MUMPS_FUNC)(MUMPS_STRUC*)>
inline void dd_block_solve(const Matrix& A, const Matrix* PR,
                            const VecI& Blocks, const V1& v1, V2& v2,
                            size_t bid, bool fwd)
{
    MUMPS_STRUC* idz = new MUMPS_STRUC;
    idz->job = mumps_job_init;
    idz->par = 1;
    idz->sym = (MUMPS_INT) 1;
    idz->comm_fortran = mumps_comm_world;
    MUMPS_FUNC(idz);
    size_t b0 = Blocks[bid], b1 = Blocks[bid+1];
    idz->n = (MUMPS_INT)(b1 - b0);

    idz->nz = 0;
    for(size_t i = b0; i < b1; i++)
        for(auto it = A.begin_row(i); it != A.end_row(i); ++it)
            if(it.col() >= b0 && it.col() < b1 && it.col() >= i) idz->nz++;

    idz->irn = new MUMPS_INT [idz->nz];
    idz->jcn = new MUMPS_INT [idz->nz];
    idz->a   = new MUMPS_COMPLEX [idz->nz];
    {
        MUMPS_INT idx = 0;
        for(size_t i = b0; i < b1; i++)
            for(auto it = A.begin_row(i); it != A.end_row(i); ++it) {
                size_t col = it.col();
                if(col >= b0 && col < b1 && col >= i) {
                    idz->a[idx].r = (MUMPS_REAL) std::real(*it);
                    idz->a[idx].i = (MUMPS_REAL) std::imag(*it);
                    idz->irn[idx] = (MUMPS_INT)(i - b0) + 1;
                    idz->jcn[idx++] = (MUMPS_INT)(col - b0) + 1;
                }
            }
    }
    idz->icntl[0] = MUMPS_ICNTL_ERRORS;
    idz->icntl[1] = MUMPS_ICNTL_DIAG;
    idz->icntl[2] = MUMPS_ICNTL_GLOBAL;
    idz->icntl[3] = MUMPS_ICNTL_MEMORY;

    // Gauss-Seidel / Schur: subtract block coupling via PR
    if(PR) {
        arma::cx_vec tmp(idz->n, arma::fill::zeros);
        size_t nblocks = Blocks.size() - 1;
        for(size_t lil = 0; lil < nblocks; lil++) {
            if(lil == bid) continue;
            if((fwd && lil < bid) || (!fwd && lil > bid)) {
                size_t c0 = Blocks[lil], c1 = Blocks[lil+1];
                for(auto it = PR->begin(); it != PR->end(); ++it)
                    if(it.row() >= b0 && it.row() < b1 && it.col() >= c0 && it.col() < c1)
                        tmp[it.row() - b0] += (*it) * v2[it.col()];
            }
        }
        for(size_t i = 0; i < size_t(idz->n); i++) v2[b0+i] -= tmp(i);
    }

    idz->rhs = new MUMPS_COMPLEX [idz->n];
    for(size_t row = 0; row < (size_t) idz->n; row++) {
        idz->rhs[row].r = (MUMPS_REAL) std::real(v2[b0+row]);
        idz->rhs[row].i = (MUMPS_REAL) std::imag(v2[b0+row]);
    }
    idz->job = MUMPS_JOB_FACTOR_SOLVE;
    MUMPS_FUNC(idz);
    for(size_t row = 0; row < (size_t) idz->n; row++)
        v2[b0+row] = std::complex<double>(idz->rhs[row].r, idz->rhs[row].i);
    idz->job = MUMPS_JOB_END;
    MUMPS_FUNC(idz);
    delete[] idz->a; delete[] idz->rhs; delete[] idz->irn; delete[] idz->jcn;
    delete idz;
}

// Unified preconditioner dispatching on MUMPS type via tag dispatch.
template <typename Matrix, typename VecI, typename V1, typename V2>
inline void block_preconditioner(const Matrix& A, const Matrix* PR,
                                  const VecI& Blocks, const V1& v1, V2& v2,
                                  bool dbl, bool jor_gs)
{
    v2 = v1;
    bool fwd = !jor_gs;
    for(size_t did = 0; did < Blocks.size()-1; did++) {
        if(dbl)
            dd_block_solve<Matrix, VecI, V1, V2, ZMUMPS_STRUC_C, ZMUMPS_COMPLEX, ZMUMPS_REAL, zmumps_c>(
                A, PR, Blocks, v1, v2, did, fwd);
        else
            dd_block_solve<Matrix, VecI, V1, V2, CMUMPS_STRUC_C, CMUMPS_COMPLEX, CMUMPS_REAL, cmumps_c>(
                A, PR, Blocks, v1, v2, did, fwd);
    }
}


// ============== Main gmres solver ==============

template <typename Mat, typename Vec, typename VecB, typename VecI, typename Log, typename Opt>
void gmres(const Mat& A, const Mat& PR, Vec& x, const VecB& b, const VecI& Blocks,
           int restart, iteration& outer, Log& logFile, Opt& opt)
{
    if(opt->n_jor_gs)
    {
        if(opt->dbl)
        {
            logFile << "Gauss-Seidel double prec. preconditioner\n";
            std::cout << "Gauss-Seidel double prec. preconditioner\n";
        }
        else
        {
            logFile << "Gauss-Seidel single prec. preconditioner\n";
            std::cout << "Gauss-Seidel single prec. preconditioner\n";
        }
    }
    else
    {
        if(opt->dbl)
        {
            logFile << "Jacobi double prec. preconditioner\n";
            std::cout << "Jacobi double prec. preconditioner\n";
        }
        else
        {
            logFile << "Jacobi single prec. preconditioner\n";
            std::cout << "Jacobi single prec. preconditioner\n";
        }
    }

    using T = std::complex<double>;
    using R = double;

    // Select the correct preconditioner based on precision and type
    auto apply_precond = [&](const auto& vec_in, auto& vec_out) {
        block_preconditioner(A, opt->n_jor_gs ? &PR : nullptr,
                             Blocks, vec_in, vec_out, opt->dbl, opt->n_jor_gs);
    };

    size_t n = x.size();
    // Convert external vectors to arma::cx_vec for internal use
    arma::cx_vec xv(n), bv(n);
    for(size_t i=0; i<n; i++) { xv(i) = x[i]; bv(i) = b[i]; }
    std::vector<arma::cx_vec> KS(restart+1);
    for(auto& v : KS) v.set_size(n);
    arma::cx_vec w(n), r(n), u(n), tmp(n);
    std::vector<T> c_rot(restart+1), s_rot(restart+1);
    arma::cx_vec sv(restart+1);
    arma::cx_mat H(restart+1, restart, arma::fill::zeros);

    Mat Adiag(n, n);
    for(size_t id=0; id < n; id++)
    {
        Adiag(id,id) = -A(id,id);
    }

    // Apply preconditioner to RHS
    apply_precond(b, r);

    outer.set_rhsnorm(arma::norm(r, 2));
    if(outer.get_rhsnorm() == 0.0)
    {
        xv.zeros(); for(size_t _i=0; _i<n; _i++) x[_i] = 0;
        return;
    }

    // Compute initial residual: w = bv - (A + A^T + Adiag + PR) * xv
    w = bv - A * xv;
    w -= A.t() * xv;
    w -= Adiag * xv;
    w -= PR * xv;

    apply_precond(w, r);
    R beta = arma::norm(r, 2), beta_old = beta;
    int blocked = 0;
    iteration inner = outer;
    inner.reduce_noisy();
    inner.set_maxiter(restart);
    inner.set_name("GMRes inner");
    int itnum = 0;

    while(! outer.finished(beta))
    {
        KS[0] = r / beta;
        sv.zeros();
        sv(0) = T(beta, 0);
        size_t i = 0;
        inner.init();

        do
        {
            logFile << itnum << " " << (inner.get_res()/inner.get_rhsnorm() > 0 ?
                                        inner.get_res()/inner.get_rhsnorm() :
                                        outer.get_res()/outer.get_rhsnorm()) << "\n";
            std::cout << itnum++ << ": " << (inner.get_res()/inner.get_rhsnorm() > 0 ?
                                             inner.get_res()/inner.get_rhsnorm() :
                                             outer.get_res()/outer.get_rhsnorm()) << "\n";

            // Apply system operator: u = (A + A^T + Adiag + PR) * KS[i]
            u = A * KS[i];
            u += A.t() * KS[i];
            u += Adiag * KS[i];
            u += PR * KS[i];

            apply_precond(u, tmp);
            KS[i+1] = tmp;
            { arma::cx_vec Hcol = H.col(i); Orthogonalize(KS, Hcol, i); H.col(i) = Hcol; }
            R a = arma::norm(KS[i+1], 2);
            H(i+1, i) = T(a, 0);
            KS[i+1] /= a;

            for(size_t k = 0; k < i; k++)
            {
                ApplyGivensLeft(H(k,i), H(k+1,i), c_rot[k], s_rot[k]);
            }
            GivensRotation(H(i,i), H(i+1,i), c_rot[i], s_rot[i]);
            ApplyGivensLeft(H(i,i), H(i+1,i), c_rot[i], s_rot[i]);
            ApplyGivensLeft(sv(i), sv(i+1), c_rot[i], s_rot[i]);

            ++inner, ++outer, ++i;
        }
        while(! inner.finished(std::abs(sv(i))));

        UpperTriSolve(H, sv, i, false);
        Combine(KS, sv, xv, i);

        // Compute new residual: w = bv - (A + A^T + Adiag + PR) * xv
        w = bv - A * xv;
        w -= A.t() * xv;
        w -= Adiag * xv;
        w -= PR * xv;

        apply_precond(w, r);

        beta_old = std::min(beta, beta_old);
        beta = arma::norm(r, 2);
        if(int(inner.get_iteration()) < restart -1 || beta_old <= beta)
        {
            ++blocked;
        }
        else
        {
            blocked = 0;
        }
        if(blocked > 10)
        {
            if(outer.get_noisy())
            {
                std::cout << "Gmres is blocked, exiting\n";
            }
            break;
        }
    }
    // Copy solution back
    for(size_t _i=0; _i<n; _i++) x[_i] = xv(_i);
}

#endif // gmres_H
