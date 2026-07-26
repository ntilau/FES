#ifndef FIELD_H
#define FIELD_H

#include "project.h"
#include <armadillo>

class field
{
public:
    field() {}
    field(project* prj, arma::cx_mat& sol, double freq);
    void Dumpmesh();
    void dump_efield();
    void dump_hfield();
    void dump_vpot();
    virtual ~field();
private:
    project* prj;
    arma::mat Nodes;
    arma::umat Cells;
    size_t nCells;
    arma::cx_mat CellVal, NodeVal;
    size_t cnt;
    bool frstPntData;
    double freq;
};

#endif // FIELD_H
