#ifndef FEMESH_H
#define FEMESH_H

#include <armadillo>

#include "BC.h"
#include "Mtrl.h"

class Mesh
{
public:
    static const size_t maxLab = UINT_MAX;
    Mesh();
    virtual ~Mesh();
    // methods
    void PartitionMesh(int);
    void SaveField(std::string);
    void RefineHomogeneous();
    void Reorder();
    void Clear();
    arma::mat tetGeo(size_t);
    arma::mat facGeo(size_t);
    arma::mat facGeo2(size_t);
    arma::vec intNode(size_t);
    arma::vec intNode(size_t, size_t&);
    // members
    arma::umat tetNodes;
    arma::umat tetEdges;
    arma::umat tetFaces;
    arma::umat facNodes;
    arma::umat facEdges;
    arma::umat edgNodes;
    arma::mat nodPos;
    arma::uvec facLab;
    arma::uvec tetLab;
    arma::uvec tetDom;
    std::vector<bool> facOnRadBnd;
    arma::field<arma::uvec> facAdjTet;
    arma::field<arma::uvec> edgAdjFac;
    arma::field<arma::uvec> domTetras;
    arma::field<arma::uvec> domFaces;
    std::vector<BC> facBC;
    std::vector<Mtrl> tetMtrl;
    size_t nNodes, nEdges, nFaces, nTetras, nDomains;
};

#endif // FEMESH_H
