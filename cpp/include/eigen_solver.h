#ifndef EIG_H
#define EIG_H

#include <math.h>
#include <armadillo>

#include <zmumps_c.h>

extern "C" void
znaupd_(int* ido, char* bmat, int* n, char* which,
        int* nev, double* tol, std::complex<double>* resid,
        int* ncv, std::complex<double>* v, int* ldv,
        int* iparam, int* ipntr, std::complex<double>* workd,
        std::complex<double>* workl, int* lworkl,
        double* rwork, int* info);

extern "C" void
zneupd_(int* rvec, char* All, int* select,
        std::complex<double>* d, std::complex<double>* v, int* ldv,
        double* sigma, std::complex<double>* workev, char* bmat,
        int* n, char* which, int* nev, double* tol,
        std::complex<double>* resid, int* ncv,
        std::complex<double>* v1, int* ldv1, int* iparam,
        int* ipntr, std::complex<double>* workd,
        std::complex<double>* workl, int* lworkl,
        double* rwork, int* ierr);

class eigen
{
public:
    eigen(arma::cx_mat& SS, arma::cx_mat& TT, size_t& numt, size_t& numz, double& kk, int& num_modes);
    virtual ~eigen();
    arma::cx_mat mode_vec;
    arma::cx_vec mode_beta;
protected:
    void op(int n, std::complex<double>* in, std::complex<double>* out);
    void znaupd(int n, int nev, std::complex<double>* Evals, std::complex<double>* Evecs, double sigma);
private:
    ZMUMPS_STRUC_C* mumpsIdz;   // MUMPS instance for shift-invert solve
    arma::cx_mat    mumpsTT;    // RHS matrix (TT) for op() = (TT/kk) * in
    arma::cx_mat    mumpsOO;    // orthogonalization matrix
    double          mumpsKk;    // shift parameter kk
};

#endif // EIG_H
