#ifndef COUPL_H
#define COUPL_H

#include <armadillo>

class Coupl
{
public:
    Coupl(size_t&, std::complex<double>& epsr, std::complex<double>& kerr, arma::vec& normE);
    virtual ~Coupl();
    arma::cx_mat D, N;
};

#endif // COUPL_H
