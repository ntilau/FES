#ifndef RAD_H
#define RAD_H

#include "project.h"
#include <armadillo>
#include "quadrature.h"
#include "shape.h"
#include "mesh.h"
#include "degree_of_freedom.h"

class rad
{
public:
    rad();
    rad(project* prj, arma::cx_mat& sol, double freq, double& Pacc);
    void Savefield();
    virtual ~rad();
private:
    project* prj;
    double pfreq;
    arma::cx_mat Ef;
    double Pacc, Prad;
    arma::cx_mat Dir;
    size_t n_theta, n_phi;
    bool nDirOrGain;
    quad* quadr;
};

#endif // RAD_H
