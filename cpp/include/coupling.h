#ifndef COUPL_H
#define COUPL_H

#include <armadillo>

class coupl
{
public:
    coupl(size_t&, std::complex<double>& epsr, std::complex<double>& kerr, arma::vec& normE);
    virtual ~coupl();
    arma::cx_mat D, N;
};

#endif // COUPL_H
