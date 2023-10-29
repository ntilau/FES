#include "Eigen.h"
#include <iomanip>

/// not accurate for more than one mode and complex materials !!! (different from HFSS)

Eigen::Eigen(arma::cx_mat& SS, arma::cx_mat& TT, size_t& numt, size_t& numz, double& kk, int& numModes)
{
    int n = SS.n_rows;
    if(n < 5)
    {
        throw std::string("Not enough DoFs on WavePorts");
    }
    int i, j, nmodes=1;
    for(i=1; i<numModes; i++)
    {
        nmodes++;
    }
    OPc.resize(n,n);
    /// full matrices solver takes too long for large matrices -> use sparse solver
    arma::solve(OPc, SS/kk-TT, TT/kk, "standard");
    // orthogonalization
    // Zhu Cangellaris p 265 (9.74)
    int numtot = numt + numz;
    arma::cx_mat OO(numtot,numtot);
    OO.fill(0);
    OO(arma::span(0,numt-1),arma::span(0,numt-1)) = arma::eye<arma::cx_mat>(numt,numt); // q_t <-- q_t
    OO(arma::span(numt,numtot-1),arma::span(0,numt-1)) = arma::solve( // q_z <-- -D^-1 C^T q_t
                -TT(arma::span(numt,numtot-1),arma::span(numt,numtot-1)), // -D^-1
                TT(arma::span(numt,numtot-1),arma::span(0,numt-1)), "standard"); // C^T
    OPc = OO*OPc;
    std::complex<double>** Evecs, *Evals;
    Evals = new std::complex<double>[nmodes];
    Evecs = new std::complex<double>* [n];
    for(i=0; i<n; i++)
    {
        Evecs[i] = new std::complex<double>[nmodes];
    }
    znaupd(n, nmodes, Evals, Evecs, kk);
    modeBeta.resize(nmodes);
    for(i=0; i<nmodes; i++)
    {
        modeBeta(i) = Evals[nmodes-1-i];
        //std::cout << modeBeta(i);
    }
    modeVec.resize(n,nmodes);
    for(i=0; i<n; i++)
    {
        for(j=0; j<nmodes; j++)
        {
            modeVec(i,j) = Evecs[i][nmodes-1-j];
        }
    }
    delete Evecs;
    delete Evals;
}


Eigen::~Eigen()
{
    modeVec.clear();
    OPr.clear();
    OPc.clear();
}

void Eigen::znaupd(int n, int nev, std::complex<double>* Evals, std::complex<double>** Evecs, double sigma)
{
    double gamtol = 0.0; // gamma tolerance
    int ido = 0;
    char bmat[2] = "I";
    char which[3] = "LM";
    char all[] = "A";
    double tol = 0.0;
    std::complex<double>* resid;
    resid = new std::complex<double>[n];
    int ncv = 4*nev;
    if(ncv>n)
    {
        ncv = n;
    }
    std::complex<double>* v;
    int ldv = n;
    v = new std::complex<double>[ldv*ncv];
    int* iparam;
    iparam = new int[11];
    iparam[0] = 1;
    iparam[2] = 100;
    iparam[3] = 1;
    iparam[6] = 3;
    int* ipntr;
    ipntr = new int[14];
    std::complex<double>* workd;
    workd = new std::complex<double>[3*n];
    int lworkl = 3*ncv*ncv + 5*ncv;
    std::complex<double>* workl;
    workl = new std::complex<double>[lworkl];
    double* rwork;
    rwork = new double[ncv];
    int info = 0;
    int rvec = 1;  // Changed from above
    int* select;
    select = new int[ncv];
    std::complex<double>* d;
    d = new std::complex<double>[ncv];
    //double sigma;// = 100;
    std::complex<double>* workev;
    workev = new std::complex<double>[3*ncv];
    int ierr;
    int iter = 0;
    do
    {
        znaupd_(&ido, bmat, &n, which, &nev, &tol, resid,
                &ncv, v, &ldv, iparam, ipntr, workd, workl,
                &lworkl, rwork, &info);
        if((ido==1)||(ido==-1))
        {
            op(n, workd+ipntr[0]-1, workd+ipntr[1]-1);
        }
        iter++;
        //std::cout << "+";
    }
    while((ido==1)||(ido==-1));
    std::cout <<  "(" << iter << ")";
    if(info<0)
    {
        throw std::string("znaupd error = " + info);
    }
    else
    {
        zneupd_(&rvec, all, select, d, v, &ldv, &sigma, workev,
                bmat, &n, which, &nev, &tol, resid, &ncv, v, &ldv,
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
        for(i=0; i<nev; i++) for(j=0; j<n; j++)
            {
                Evecs[j][i] = v[i*n+j];
            }
        std::complex<double> temp;
        for(int i=0; i<nev; i++)
        {
            for(int j=i; j<nev; j++)
            {
                if(Evals[j].real() < Evals[i].real())
                {
                    temp = Evals[j];
                    Evals[j] = Evals[i];
                    Evals[i] = temp;
                    for(int k=0; k<n; k++)
                    {
                        temp = Evecs[k][i];
                        Evecs[k][i] = Evecs[k][j];
                        Evecs[k][j] = temp;
                    }
                }
            }
        }
        for(int i=0; i<nev; i++)
        {
            for(int j=i; j<nev; j++)
            {
                if(Evals[j].imag() < Evals[i].imag())
                {
                    temp = Evals[j];
                    Evals[j] = Evals[i];
                    Evals[i] = temp;
                    for(int k=0; k<n; k++)
                    {
                        temp = Evecs[k][i];
                        Evecs[k][i] = Evecs[k][j];
                        Evecs[k][j] = temp;
                    }
                }
            }
        }
        delete resid;
        delete v;
        delete iparam;
        delete ipntr;
        delete workd;
        delete workl;
        delete select;
        delete d;
    }
}

inline void Eigen::op(int n, std::complex<double>* in, std::complex<double>* out)
{
    int i, j;
    for(i=0; i<n; i++)
    {
        out[i] = 0.0;
        for(j=0; j<n; j++)
        {
            out[i] += OPc(i,j)*in[j];
        }
    }
}

void Eigen::dnaupd(int n, int nev, double* Evals, double** Evecs, double sigmaReal)
{
    int ido = 0;
    char bmat[2] = "I";
    char which[3] = "LR";
    char all[] = "A";
    double tol = 0.0;
    double* resid;
    resid = new double[n];
    double* v;
    int ldv = n;
    int lancz = 2*nev+1;
    v = new double[ldv*lancz];
    int* iparam;
    iparam = new int[11];
    iparam[0] = 1;
    iparam[2] = 100;
    iparam[3] = 1;
    iparam[6] = 3;
    int* ipntr;
    ipntr = new int[14];
    double* workd;
    workd = new double[3*n];
    double* workl;
    int lworkl = 3*lancz*(lancz+2);
    workl = new double[lworkl];
    double* workev;
    workev = new double[3*lancz];
    int info = 0;
    int rvec = 1;  // return vectors
    int* select;
    select = new int[lancz];
    double* dR, *dI;
    dR = new double[nev];
    dI = new double[nev];
    double sigmaImag = 0.0;
    int ierr;
    do
    {
        dnaupd_(&ido, bmat, &n, which, &nev, &tol, resid,
                &lancz, v, &n, iparam, ipntr, workd, workl,
                &lworkl, &info);
        if(ido==-1 || ido==1)
        {
            op(n, &(workd[ipntr[0]-1]), &(workd[ipntr[1]-1]));
        }
    }
    while((ido==1)||(ido==-1)||(ido==2));
    if(info<0)
    {
        throw std::string("dnaupd error = " + info);
    }
    else
    {
        dneupd_(&rvec, all, select, dR, dI, v, &ldv, &sigmaReal, &sigmaImag,
                workev, bmat, &n, which, &nev, &tol, resid, &lancz, v, &ldv,
                iparam, ipntr, workd, workl, &lworkl, &ierr);
        if(ierr!=0)
        {
            std::cout << info;
        }
        int i, j;
        for(i=0; i<nev; i++)
        {
            std::complex<double> beta = std::sqrt(std::complex<double>(dR[i],dI[i]));
            std::cout << beta << " ";
            Evals[i] =  beta.real() + beta.imag();
        }
        for(i=0; i<nev; i++) for(j=0; j<n; j++)
            {
                Evecs[j][i] = v[i*n+j];
            }
        delete resid;
        delete v;
        delete iparam;
        delete ipntr;
        delete workd;
        delete workl;
        delete workev;
        delete select;
        delete dR;
        delete dI;
    }
}


inline void Eigen::op(int n, double* in, double* out)
{
    int i, j;
    for(i=0; i<n; i++)
    {
        out[i] = 0.0;
        for(j=0; j<n; j++)
        {
            out[i] += OPr(i,j)*in[j];
        }
    }
}
