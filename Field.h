#ifndef FIELD_H
#define FIELD_H

#include "Project.h"
#include <armadillo>

class Field
{
public:
    Field() {}
    Field(Project* prj, arma::cx_mat& sol, double& freq);
    void DumpMesh();
    void DumpEfield();
    void DumpHfield();
    void DumpVpot();
    virtual ~Field();
private:
    Project* prj;
    arma::mat Nodes;
    arma::umat Cells;
    size_t nCells;
    arma::cx_mat CellVal, NodeVal;
    size_t cnt;
    bool frstPntData;
    double freq;
};

#endif // FIELD_H
