#ifndef DOF_H
#define DOF_H

#include "project.h"
#include <armadillo>

class dof
{
public:
    dof(project*, size_t dim, size_t id); // returns current dof id
    dof(project*); // returns RAW dof
    virtual ~dof();
    arma::uvec s, v;
    size_t dofnumv, dofnums;
};

#endif // DOF_H
