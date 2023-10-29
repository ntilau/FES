#ifndef MUMPS_H
#define MUMPS_H

#include <armadillo>
#include <gmm.h>

class MUMPS
{
public:
    typedef gmm::row_matrix<gmm::rsvector<std::complex<double> > > MatRowType;
    typedef gmm::rsvector<std::complex<double> > VecType;

    MUMPS(arma::cx_mat& A, arma::cx_mat& B);
    MUMPS(MatRowType& AII, MatRowType& AFI, MatRowType& S);
    MUMPS(MatRowType& SF, MatRowType& gF);
    MUMPS(MatRowType& A, std::vector<size_t>& RangeAII, size_t& RangeAFF, MatRowType& IO, bool Mode); // mode 0: IO ~ SF, mode 1: IO ~ gF
    MUMPS() {}
    virtual ~MUMPS();
    arma::cx_mat Sol;
    MatRowType A, B, invAIIAFIt, SII;
};

#endif // MUMPS_H
