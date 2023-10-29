#ifndef FEMAT_H
#define FEMAT_H

#include <armadillo>
#include "Quad.h"
#include "Shape.h"
#include "Mesh.h"
#include "DoF.h"

class EleMat
{
public:
    EleMat(size_t p, size_t cDim, arma::mat cGeo, Quad*, Mtrl*, Shape::STYPE);
    EleMat(size_t p, size_t cDim, arma::mat cGeo, Quad*, Mtrl*, arma::vec intNode);
    EleMat(size_t p, size_t cDim, arma::mat cGeo, Quad*, Mtrl*, arma::vec intNode,
           arma::vec kEinc, arma::vec polEinc);
    EleMat(size_t p, size_t cDim, arma::mat cGeo, Quad*, Mtrl*, DoF*,
           arma::cx_vec cSol, size_t nHarm, size_t dofNum, double mFreq);
    virtual ~EleMat();
    Jacobian* cJac;
    arma::cx_mat S, T, Z;
    arma::cx_mat St, Tt, Sz, Tz, G, STt, SSt;
    arma::cx_vec f;
};

#endif // FEMAT_H
