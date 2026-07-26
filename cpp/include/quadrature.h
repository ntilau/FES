#ifndef QUAD_H
#define QUAD_H

#include <armadillo>

class quad
{
public:
    quad(size_t p);
    virtual ~quad();
    void setquad(size_t p);
    arma::mat xq3;
    arma::mat xq2;
    arma::vec wq3;
    arma::vec wq2;
};

#endif // QUAD_H
