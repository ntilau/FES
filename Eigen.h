#ifndef EIG_H
#define EIG_H

#include <math.h>
#include <armadillo>

extern "C"  void
dnaupd_(int* ido, char* problem, int* n, char* whichArr, int* nev, double* tol,
        double* residArr, int* ncv, double* vMatrix, int* ldv, int* iparamArr,
        int* ipntrArr, double* workdArr, double* worklArr, int* lworkl, int* info);

extern "C"  void
dneupd_(int* rvec, char* stringArr, int* selectArr, double* dArrReal, double* dArrImag,
        double* vMatrix, int* ldv, double* sigmaReal, double* sigmaImag, double* workev,
        char* bmat, int* n,
        char* whichArr, int* nev, double* tol, double* residArr, int* ncv,
        double* vMatrix1, int* ldv1, int* iparamArr, int* ipntrArr, double* workdArr,
        double* worklArr, int* lworkl, int* ierr);

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

class Eigen
{
public:
    Eigen(arma::cx_mat& SS, arma::cx_mat& TT, size_t& numt, size_t& numz, double& kk, int& numModes);
    virtual ~Eigen();
    arma::cx_mat modeVec;
    arma::cx_vec modeBeta;
protected:
    void op(int n, double* in, double* out);
    void b(int n, double* in, double* out);
    void op(int n, std::complex<double>* in, std::complex<double>* out);
    void dnaupd(int n, int nev, double* Evals, double** Evecs, double sigma);
    void znaupd(int n, int nev, std::complex<double>* Evals, std::complex<double>** Evecs, double sigma);
private:
    arma::cx_mat OPc;
    arma::mat OPr;
};

#endif // EIG_H
