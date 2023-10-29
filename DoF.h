#ifndef DOF_H
#define DOF_H

#include "Project.h"
#include <armadillo>

class DoF
{
public:
    DoF(Project*, size_t dim, size_t id); // returns current DoF id
    DoF(Project*); // returns RAW DoF
    virtual ~DoF();
    arma::uvec s, v;
    size_t DoFnumv, DoFnums;
};

#endif // DOF_H
