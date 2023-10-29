#ifndef RAD_H
#define RAD_H

#include "Project.h"
#include <armadillo>
#include "Project.h"
#include "Quad.h"
#include "Shape.h"
#include "Mesh.h"
#include "DoF.h"

class Rad
{
public:
    Rad();
    Rad(Project* prj, arma::cx_mat& sol, double& freq, double& Pacc);
    void SaveField();
    virtual ~Rad();
private:
    Project* prj;
    double pfreq;
    arma::cx_mat Ef;
    double Pacc, Prad;
    arma::cx_mat Dir;
    size_t nTheta, nPhi;
    bool nDirOrGain;
    Quad* quad;
};

#endif // RAD_H
