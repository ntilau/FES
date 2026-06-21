#ifndef SHAPE_H
#define SHAPE_H

#include <armadillo>

class jacobian
{
public:
    jacobian(size_t cDim, arma::mat cGeo);
    virtual ~jacobian();
    double detJ;
    arma::mat invJ;
};

class shape
{
public:
    enum s_type {hgrad, hcurl, hdiv};
    shape(size_t p_ord, size_t cDim, s_type sType, arma::rowvec cPos, jacobian* cJac);
    virtual ~shape();
    arma::mat Ns, dNs, cNs, Nv, dNv, cNv, divNv;
};

#endif // SHAPE_H
