#ifndef FEMAT_H
#define FEMAT_H

#include <armadillo>
#include "quadrature.h"
#include "shape.h"
#include "mesh.h"
#include "degree_of_freedom.h"

class ele_mat
{
public:
    ele_mat(size_t p, size_t cDim, arma::mat cGeo, quad*, mtrl*, shape::s_type);
    ele_mat(size_t p, size_t cDim, arma::mat cGeo, quad*, mtrl*, arma::vec int_node);
    ele_mat(size_t p, size_t cDim, arma::mat cGeo, quad*, mtrl*, arma::vec int_node,
           arma::vec kEinc, arma::vec polEinc);
    ele_mat(size_t p, size_t cDim, arma::mat cGeo, quad*, mtrl*, dof*,
           arma::cx_vec cSol, size_t n_harm, size_t dofNum, double mFreq);
    virtual ~ele_mat();
    jacobian* cJac;
    arma::cx_mat S, T, Z;
    arma::cx_mat St, Tt, Sz, Tz, G, STt, SSt;
    arma::cx_vec f;
};

#endif // FEMAT_H
