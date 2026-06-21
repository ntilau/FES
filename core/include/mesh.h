#ifndef FEMESH_H
#define FEMESH_H

#include <armadillo>

#include "boundary_condition.h"
#include "material.h"

class mesh
{
public:
    static const size_t maxLab = UINT_MAX;
    mesh();
    virtual ~mesh();
    // methods
    void Partitionmesh(int);
    void Savefield(std::string);
    void refine_homogeneous();
    void Reorder();
    void Clear();
    size_t check_regular(std::ofstream& log);
    void normalize_order();
    void build_2d_edge_connectivity();
    arma::mat tet_geo(size_t) const;
    arma::mat fac_geo(size_t) const;
    arma::mat fac_geo2(size_t) const;
    arma::vec int_node(size_t) const;
    arma::vec int_node(size_t, size_t&) const;
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
    std::vector<bool> facOnradBnd;
    arma::field<arma::uvec> facAdjTet;
    arma::field<arma::uvec> edgAdjFac;
    arma::field<arma::uvec> domTetras;
    arma::field<arma::uvec> domFaces;
    std::vector<bc> facbc;
    std::vector<mtrl> tetmtrl;
    size_t nNodes, nEdges, nFaces, nTetras, nDomains;
    int mesh_dim;           // 2 or 3 (0 = unset)
    arma::uvec edgLab;       // E×1 edge boundary markers (2D TMz)

    // PLC geometry (for mesh regeneration from .fes)
    bool plc_valid;
    // 3D TetGen PLC
    std::vector<double> plc_points;         // x,y,z triplets
    std::vector<int>    plc_facet_markers;  // boundary markers per facet
    std::vector<int>    plc_poly_vertex_counts;  // flattened polygon vertex counts
    std::vector<int>    plc_poly_vertex_list;    // flattened polygon vertex indices
    std::vector<double> plc_facet_holes;    // (x,y,z) per facet hole
    std::vector<double> plc_volume_holes;   // (x,y,z) volume hole seeds
    std::vector<double> plc_regions;        // (x,y,z,attrib,maxvol) per region
    // 2D Triangle PLC
    std::vector<double> plc_2d_points;      // x,y pairs
    std::vector<int>    plc_segments;       // v1,v2 pairs
    std::vector<int>    plc_seg_markers;    // segment boundary markers
    std::vector<double> plc_2d_holes;       // x,y pairs
    std::vector<double> plc_2d_regions;     // (x,y,attrib,maxarea) per region
    // Populate PLC from external mesh generator input
    void store_plc_from_tetgen(class tetgenio& in);
    void store_plc_from_triangle(struct triangulateio& tri_in);
    void store_plc_from_hfss();  // from existing mesh (HFSS already provides complete mesh)
};

#endif // FEMESH_H
