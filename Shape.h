#ifndef SHAPE_H
#define SHAPE_H

#include <armadillo>

class Jacobian
{
public:
    Jacobian(size_t cDim, arma::mat cGeo);
    virtual ~Jacobian();
    double detJ;
    arma::mat invJ;
};

class Shape
{
public:
    enum STYPE {Hgrad, Hcurl, Hdiv};
    Shape(size_t pOrd, size_t cDim, STYPE sType, arma::rowvec cPos, Jacobian* cJac);
    virtual ~Shape();
    arma::mat Ns, dNs, cNs, Nv, dNv, cNv;
};

#endif // SHAPE_H
