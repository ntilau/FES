#ifndef EQSYS_H
#define EQSYS_H

#include <gmm.h>
#include <armadillo>

class Mesh;
class Option;
class Project;
class Quad;

class EqSys
{
public:
    typedef gmm::row_matrix<gmm::rsvector<std::complex<double> > > MatRowType;
    typedef gmm::col_matrix<gmm::rsvector<std::complex<double> > > MatColType;
    typedef gmm::rsvector<std::complex<double> > VecType;
    EqSys(std::ofstream&, Project*);
    virtual ~EqSys();
    void CheckSystemType(); // check if symmetric positive definite matrices are available
    void SolveSingleComplex(std::ofstream&);
    void SolveDoubleComplex(std::ofstream&);
    void SolveGmRes(std::ofstream&);
    void SaveSystem(std::ofstream&);
    void SaveData(std::ofstream&);
    //
    Project* prj;
    Mesh* msh;
    Option* opt;
    Quad* quad;
    //
    arma::uvec DirDoFs, DirDoFv;
    std::vector<size_t> NonDirIds;
    arma::uvec DoFmapv, InvDoFmapv; // usefull for DD mapping or reordering
    std::vector<size_t> DoFlevel; // DoF level for Schur decomposition
    std::vector<MatRowType> AFF; // for each domain internal AFF
    arma::uvec WavePortIds, NonWavePortIds;
    //
    size_t DoFnum, DoFreal, NonZero;
    size_t WavePortsNum;
    size_t WavePortsDoFnum;
    std::vector<std::complex<double> > PortAmpl;
    double freq, k0, kk, error;
    size_t iter;
    MatRowType A, PR, Adiag;
    MatColType B;
    arma::cx_mat Sp, Spprev, Sol, Solprev;
    int SymmFlag; // A matrix symmetry
    arma::wall_clock tt, lt;
};

#endif // EQSYS_H
