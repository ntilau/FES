#include "mesh.h"
#include "project.h"
#include <algorithm>
#include <cmath>

namespace {
    constexpr int VtkTetCellType = 10;
    constexpr int VtkTetVertsPerCell = 4;
    constexpr int VtkTetfieldsPerCell = VtkTetVertsPerCell + 1; // count + vertex indices
}
#include <iomanip>
#include <map>
#include <numeric>
#include <set>
#include <unordered_set>
#include <vector>

namespace {

static const int nd_threshold = 40;

// Recursive Coordinate Bisection: sort elements by centroid coordinate
// along alternating axes and split into balanced partitions.
static void rcbRecursive(const arma::mat& centroids,
                         std::vector<int>& elemIds, int start, int end,
                         int partBase, int nparts, int depth,
                         arma::uvec& tetDom)
{
    int count = end - start;
    if (nparts <= 1 || count <= 1)
    {
        for (int i = start; i < end; i++)
            tetDom(elemIds[i]) = partBase;
        return;
    }

    int axis = depth % 3;
    std::sort(elemIds.begin() + start, elemIds.begin() + end,
              [&](int a, int b) { return centroids(a, axis) < centroids(b, axis); });

    int mid = start + count / 2;
    int leftParts = nparts / 2;
    int rightParts = nparts - leftParts;

    rcbRecursive(centroids, elemIds, start, mid, partBase, leftParts, depth + 1, tetDom);
    rcbRecursive(centroids, elemIds, mid, end, partBase + leftParts, rightParts, depth + 1, tetDom);
}

// Build undirected nodal adjacency graph from tetrahedral element connectivity.
// Replaces METIS_meshToNodal.
static void buildNodalGraph(int nn, const arma::umat& tetNodes,
                            std::vector<int>& xadj, std::vector<int>& adjncy)
{
    int ne = (int)tetNodes.n_rows;
    std::vector<std::set<int>> adj(nn);
    for (int e = 0; e < ne; e++)
    {
        int n0 = (int)tetNodes(e, 0);
        int n1 = (int)tetNodes(e, 1);
        int n2 = (int)tetNodes(e, 2);
        int n3 = (int)tetNodes(e, 3);
        adj[n0].insert({n1, n2, n3});
        adj[n1].insert({n0, n2, n3});
        adj[n2].insert({n0, n1, n3});
        adj[n3].insert({n0, n1, n2});
    }
    xadj.resize(nn + 1);
    xadj[0] = 0;
    for (int i = 0; i < nn; i++)
        xadj[i + 1] = xadj[i] + (int)adj[i].size();
    adjncy.resize(xadj[nn]);
    int idx = 0;
    for (int i = 0; i < nn; i++)
        for (int v : adj[i])
            adjncy[idx++] = v;
}

// Geometric nested dissection: recursively bisect by longest axis,
// find separator nodes with edges crossing the cut, and order
// interior nodes before separator nodes to reduce matrix fill-in.
// Replaces METIS_NodeND.
static void ndRecursive(const std::vector<int>& nodes,
                        const arma::mat& nodPos,
                        const std::vector<int>& xadj,
                        const std::vector<int>& adjncy,
                        std::vector<int>& perm)
{
    int n = (int)nodes.size();
    if (n <= nd_threshold)
    {
        for (int v : nodes) perm.push_back(v);
        return;
    }

    // Find longest axis of bounding box
    double minC[3] = {HUGE_VAL, HUGE_VAL, HUGE_VAL};
    double maxC[3] = {-HUGE_VAL, -HUGE_VAL, -HUGE_VAL};
    for (int v : nodes)
    {
        for (int d = 0; d < 3; d++)
        {
            double c = nodPos(v, d);
            if (c < minC[d]) minC[d] = c;
            if (c > maxC[d]) maxC[d] = c;
        }
    }
    int axis = 0;
    if (maxC[1] - minC[1] > maxC[axis] - minC[axis]) axis = 1;
    if (maxC[2] - minC[2] > maxC[axis] - minC[axis]) axis = 2;

    // Sort nodes by coordinate along longest axis
    std::vector<int> sorted = nodes;
    std::sort(sorted.begin(), sorted.end(), [&](int a, int b) {
        return nodPos(a, axis) < nodPos(b, axis);
    });

    int mid = n / 2;
    std::unordered_set<int> leftSet(sorted.begin(), sorted.begin() + mid);

    // Find separator: nodes with edges crossing left/right partition
    std::unordered_set<int> sepSet;
    for (int i = 0; i < mid; i++)
    {
        int v = sorted[i];
        for (int j = xadj[v]; j < xadj[v + 1]; j++)
            if (!leftSet.count(adjncy[j])) { sepSet.insert(v); break; }
    }
    for (int i = mid; i < n; i++)
    {
        int v = sorted[i];
        for (int j = xadj[v]; j < xadj[v + 1]; j++)
            if (leftSet.count(adjncy[j])) { sepSet.insert(v); break; }
    }

    // Build interior sets (non-separator nodes in each half)
    std::vector<int> leftInt, rightInt;
    for (int i = 0; i < mid; i++)
        if (!sepSet.count(sorted[i])) leftInt.push_back(sorted[i]);
    for (int i = mid; i < n; i++)
        if (!sepSet.count(sorted[i])) rightInt.push_back(sorted[i]);

    // Nested dissection order: interiors first, separator last
    ndRecursive(leftInt, nodPos, xadj, adjncy, perm);
    ndRecursive(rightInt, nodPos, xadj, adjncy, perm);
    for (int v : sorted)
        if (sepSet.count(v)) perm.push_back(v);
}

} // anonymous namespace

mesh::mesh()
{
}

mesh::~mesh()
{
    Clear();
}

void mesh::Clear()
{
    tetNodes.clear();
    tetEdges.clear();
    tetFaces.clear();
    facNodes.clear();
    facEdges.clear();
    edgNodes.clear();
    nodPos.clear();
    facLab.clear();
    tetLab.clear();
    tetDom.clear();
    facOnradBnd.clear();
    facAdjTet.clear();
    edgAdjFac.clear();
    domTetras.clear();
    domFaces.clear();
    facbc.clear();
}

void mesh::normalize_order()
{
    auto safeRowVec = [&](size_t idx) -> arma::vec {
        if(idx >= (size_t)nNodes || idx >= nodPos.n_rows)
            return arma::zeros<arma::vec>(3);
        return nodPos.row(idx).st();
    };

    // Edge: two-node entities — increasing order is always safe
    for(size_t i = 0; i < nEdges; i++)
        edgNodes.row(i) = arma::sort(edgNodes.row(i));

    // Face: sort nodes, then reorient to maintain positive signed area
    for(size_t i = 0; i < nFaces; i++) {
        arma::urowvec n = arma::sort(facNodes.row(i));
        if(nTetras == 0) {
            // 2D: triangles are elements with a consistent orientation
            arma::vec v0 = safeRowVec(n(0));
            arma::vec v1 = safeRowVec(n(1));
            arma::vec v2 = safeRowVec(n(2));
            double cx = (v1(0)-v0(0))*(v2(1)-v0(1)) - (v1(1)-v0(1))*(v2(0)-v0(0));
            if(cx < 0) std::swap(n(1), n(2));
        }
        facNodes.row(i) = n;
    }

    // Tetrahedron: sort nodes, then reorient to maintain positive signed volume
    for(size_t i = 0; i < nTetras; i++) {
        arma::urowvec n = arma::sort(tetNodes.row(i));
        arma::vec v0 = safeRowVec(n(0));
        arma::vec v1 = safeRowVec(n(1));
        arma::vec v2 = safeRowVec(n(2));
        arma::vec v3 = safeRowVec(n(3));
        double vol = arma::dot(arma::cross(v1-v0, v2-v0), v3-v0) / 6.0;
        if(vol < 0) std::swap(n(2), n(3));
        tetNodes.row(i) = n;
    }
}

void mesh::build_2d_edge_connectivity()
{
    // Build full edge connectivity for 2D hcurl DOF mapping.
    // Triangle only stores boundary edges; we need ALL triangle edges as DOFs.
    if(!facEdges.is_empty() || nFaces == 0) return;

    // Build complete edge map from all triangles
    std::map<std::pair<size_t,size_t>, size_t> emap;
    for(size_t ei = 0; ei < nEdges; ei++) {
        size_t a = edgNodes(ei,0), b = edgNodes(ei,1);
        if(a > b) std::swap(a,b);
        emap[std::make_pair(a,b)] = ei;
    }
    size_t nextEdge = nEdges;
    for(size_t fi = 0; fi < nFaces; fi++) {
        for(int ei = 0; ei < 3; ei++) {
            size_t a = facNodes(fi, (ei+1)%3);
            size_t b = facNodes(fi, (ei+2)%3);
            if(a > b) std::swap(a,b);
            auto key = std::make_pair(a,b);
            if(emap.find(key) == emap.end())
                emap[key] = nextEdge++;
        }
    }
    size_t totalEdges = nextEdge;
    size_t origEdges = nEdges;
    nEdges = totalEdges;
    edgNodes.set_size(totalEdges, 2);
    // Preserve original edge labels, then extend with maxLab for new edges
    arma::uvec newEdgLab(totalEdges, arma::fill::zeros);
    if(origEdges > 0)
        newEdgLab.rows(0, origEdges-1) = edgLab.rows(0, origEdges-1);
    edgLab = newEdgLab;
    for(auto& kv : emap) {
        edgNodes(kv.second, 0) = kv.first.first;
        edgNodes(kv.second, 1) = kv.first.second;
        if(kv.second >= origEdges)
            edgLab(kv.second) = maxLab;
    }
    facEdges.set_size(nFaces, 3);
    for(size_t fi = 0; fi < nFaces; fi++) {
        size_t n0 = facNodes(fi,0), n1 = facNodes(fi,1), n2 = facNodes(fi,2);
        if(n0 > n1) std::swap(n0,n1);
        if(n0 > n2) std::swap(n0,n2);
        if(n1 > n2) std::swap(n1,n2);
        facEdges(fi,0) = emap[std::make_pair(n0,n1)];
        facEdges(fi,1) = emap[std::make_pair(n0,n2)];
        facEdges(fi,2) = emap[std::make_pair(n1,n2)];
    }
}

void mesh::refine_homogeneous()
{
    if(nTetras > 0) {
        // 3D refinement not yet implemented via this path
        return;
    }

    // 2D uniform refinement: subdivide each triangle into 4
    // by inserting mid-edge nodes, preserving boundary markers.

    // Map each edge (sorted node pair) → mid-node index
    std::map<std::pair<size_t,size_t>, size_t> midMap;
    for(size_t f = 0; f < nFaces; f++) {
        for(int ei = 0; ei < 3; ei++) {
            size_t n0 = facNodes(f, ei);
            size_t n1 = facNodes(f, (ei+1)%3);
            if(n0 > n1) std::swap(n0, n1);
            auto key = std::make_pair(n0, n1);
            if(midMap.find(key) == midMap.end())
                midMap[key] = nNodes + midMap.size();
        }
    }

    size_t newNodeCount = nNodes + midMap.size();
    size_t newFaceCount = nFaces * 4;
    size_t newEdgeCount = nEdges * 2;

    // Create new node positions at edge midpoints
    arma::mat newNodPos(newNodeCount, 3, arma::fill::zeros);
    newNodPos.rows(0, nNodes-1) = nodPos;
    for(auto& kv : midMap) {
        size_t n0 = kv.first.first, n1 = kv.first.second;
        newNodPos.row(kv.second) = (nodPos.row(n0) + nodPos.row(n1)) * 0.5;
    }

    // Create subtriangles (4 per original)
    arma::umat newFacNodes(newFaceCount, 3);
    arma::uvec newFacLab(newFaceCount);
    for(size_t f = 0; f < nFaces; f++) {
        size_t n0 = facNodes(f,0), n1 = facNodes(f,1), n2 = facNodes(f,2);
        auto key01 = std::make_pair(std::min(n0,n1), std::max(n0,n1));
        auto key12 = std::make_pair(std::min(n1,n2), std::max(n1,n2));
        auto key20 = std::make_pair(std::min(n2,n0), std::max(n2,n0));
        size_t m01 = midMap[key01], m12 = midMap[key12], m20 = midMap[key20];
        size_t f4 = f * 4;
        newFacNodes.row(f4)   = arma::urowvec{n0, m01, m20};
        newFacNodes.row(f4+1) = arma::urowvec{n1, m12, m01};
        newFacNodes.row(f4+2) = arma::urowvec{n2, m20, m12};
        newFacNodes.row(f4+3) = arma::urowvec{m01, m12, m20};
        newFacLab(f4) = newFacLab(f4+1) = newFacLab(f4+2) = newFacLab(f4+3) = facLab(f);
    }

    // Create subsegments (2 per original boundary edge)
    arma::umat newEdgNodes(newEdgeCount, 2);
    arma::uvec newEdgLab(newEdgeCount);
    for(size_t e = 0; e < nEdges; e++) {
        size_t n0 = edgNodes(e,0), n1 = edgNodes(e,1);
        auto key = std::make_pair(std::min(n0,n1), std::max(n0,n1));
        size_t mid = midMap[key];
        size_t e2 = e * 2;
        newEdgNodes.row(e2)   = arma::urowvec{n0, mid};
        newEdgNodes.row(e2+1) = arma::urowvec{mid, n1};
        newEdgLab(e2) = newEdgLab(e2+1) = edgLab(e);
    }

    // Replace mesh data
    nodPos = newNodPos;
    facNodes = newFacNodes;
    facLab = newFacLab;
    edgNodes = newEdgNodes;
    edgLab = newEdgLab;
    nNodes = newNodeCount;
    nFaces = newFaceCount;
    nEdges = newEdgeCount;
    facEdges.reset();
    edgAdjFac.clear();
    facAdjTet.set_size(nFaces);
    facOnradBnd.assign(nFaces, false);
}

size_t mesh::check_regular(std::ofstream& log)
{
    size_t bad = 0;
    auto safeRow = [&](size_t idx) -> arma::vec {
        if(idx >= (size_t)nNodes || idx >= nodPos.n_rows) {
            static arma::vec zero = arma::zeros<arma::vec>(3);
            return zero;
        }
        return nodPos.row(idx).st();
    };
    if(nTetras > 0) {
        // 3D: signed volume of each tetrahedron must be positive
        for(size_t t = 0; t < nTetras; t++) {
            arma::vec v0 = safeRow(tetNodes(t,0));
            arma::vec v1 = safeRow(tetNodes(t,1));
            arma::vec v2 = safeRow(tetNodes(t,2));
            arma::vec v3 = safeRow(tetNodes(t,3));
            double vol = arma::dot(arma::cross(v1-v0, v2-v0), v3-v0) / 6.0;
            if(vol <= 0) bad++;
        }
    } else {
        // 2D: signed area of each triangle must be positive
        for(size_t t = 0; t < nFaces; t++) {
            arma::vec v0 = safeRow(facNodes(t,0));
            arma::vec v1 = safeRow(facNodes(t,1));
            arma::vec v2 = safeRow(facNodes(t,2));
            double cx = (v1(0)-v0(0))*(v2(1)-v0(1)) - (v1(1)-v0(1))*(v2(0)-v0(0));
            if(cx <= 0) bad++;
        }
    }
    if(bad > 0)
        log << "Warning: " << bad << " element(s) with non-positive signed volume/area "
            << "— mesh regularity issue or inverted elements.\n";
    else
        log << "mesh regularity: all elements have positive signed volume/area.\n";
    return bad;
}

void mesh::Savefield(std::string fieldName)
{
    std::ofstream outfield(std::string(fieldName + ".vtk").data());
    outfield << "# vtk DataFile Version 2.0\n";
    outfield << "mesh data\n";
    outfield << "ASCII\n";
    outfield << "DATASET UNSTRUCTURED_GRID\n";
    outfield << "POINTS " << nNodes << " double \n";
    for(size_t i = 0; i < nNodes; i++)
    {
        outfield << std::setprecision(16) << nodPos(i,0) << " ";
        outfield << std::setprecision(16) << nodPos(i,1) << " ";
        outfield << std::setprecision(16) << nodPos(i,2) << "\n";
    }
    outfield << "CELLS " << nTetras << " " << VtkTetfieldsPerCell * nTetras << "\n";
    for(size_t i = 0; i < nTetras; i++)
    {
        outfield << VtkTetVertsPerCell << " ";
        outfield << tetNodes(i,0) << " ";
        outfield << tetNodes(i,1) << " ";
        outfield << tetNodes(i,2) << " ";
        outfield << tetNodes(i,3) << "\n";
    }
    outfield << "CELL_TYPES " << nTetras << "\n";
    for(size_t i = 0; i < nTetras; i++)
    {
        outfield << VtkTetCellType << "\n";
    }
    outfield.close();
}

void mesh::Partitionmesh(int nparts)
{
    std::cout << "mesh partitioning";
    arma::wall_clock mt;
    mt.tic();
    nDomains = nparts;

    // Compute element centroids
    arma::mat centroids(nTetras, 3);
    for(size_t i = 0; i < nTetras; i++)
    {
        centroids(i, 0) = (nodPos(tetNodes(i,0), 0) + nodPos(tetNodes(i,1), 0) +
                           nodPos(tetNodes(i,2), 0) + nodPos(tetNodes(i,3), 0)) * 0.25;
        centroids(i, 1) = (nodPos(tetNodes(i,0), 1) + nodPos(tetNodes(i,1), 1) +
                           nodPos(tetNodes(i,2), 1) + nodPos(tetNodes(i,3), 1)) * 0.25;
        centroids(i, 2) = (nodPos(tetNodes(i,0), 2) + nodPos(tetNodes(i,1), 2) +
                           nodPos(tetNodes(i,2), 2) + nodPos(tetNodes(i,3), 2)) * 0.25;
    }

    // Recursive Coordinate Bisection partitioning
    tetDom.resize(nTetras);
    if(nparts > 1)
    {
        std::vector<int> elemIds(nTetras);
        std::iota(elemIds.begin(), elemIds.end(), 0);
        rcbRecursive(centroids, elemIds, 0, (int)nTetras, 0, nparts, 0, tetDom);
    }
    else
    {
        tetDom.zeros();
    }

    std::cout << ".";
    domTetras.set_size(nparts);
    domFaces.set_size(nparts);
    std::vector<size_t> domcnt(nparts,0);
    for(size_t tit = 0; tit < nTetras; tit++)
    {
        {
            ++domcnt[tetDom(tit)];
        }
    }
    for(size_t did = 0; did < nparts; did++)
    {
        domTetras(did).resize(domcnt[did]);
        //std::cout << domcnt[did] << " ";
        domcnt[did] = 0;
    }
    for(size_t tit = 0; tit < nTetras; tit++)
    {
        size_t dom = tetDom(tit);
        {
            domTetras(dom)(domcnt[dom]++) = tit;
        }
    }
    std::cout << ".";
    for(size_t fif = 0; fif < nFaces; fif++)
    {
        if(facAdjTet(fif).n_rows > 1)
        {
            arma::uvec fid(1);
            fid(0) = fif;
            arma::uvec adjTet = facAdjTet(fif);
            if(tetDom(adjTet(0)) != tetDom(adjTet(1)))
            {
                {
                    domFaces(tetDom(adjTet(0))) = arma::join_cols(domFaces(tetDom(adjTet(0))), fid);
                    domFaces(tetDom(adjTet(1))) = arma::join_cols(domFaces(tetDom(adjTet(1))), fid);
                }
            }
        }
    }
    std::cout << ".";
    std::cout << " " << mt.toc() << " s, ";
    std::ofstream outfield(std::string("Partitions.vtk").data());
    outfield << "# vtk DataFile Version 2.0\n";
    outfield << "mesh data\n";
    outfield << "ASCII\n";
    outfield << "DATASET UNSTRUCTURED_GRID\n";
    outfield << "POINTS " << nNodes << " double \n";
    for(size_t i = 0; i < nNodes; i++)
    {
        outfield << std::setprecision(16) << nodPos(i,0) << " ";
        outfield << std::setprecision(16) << nodPos(i,1) << " ";
        outfield << std::setprecision(16) << nodPos(i,2) << "\n";
    }
    outfield << "CELLS " << nTetras << " " << VtkTetfieldsPerCell * nTetras << "\n";
    for(size_t i = 0; i < nTetras; i++)
    {
        outfield << VtkTetVertsPerCell << " ";
        outfield << tetNodes(i,0) << " ";
        outfield << tetNodes(i,1) << " ";
        outfield << tetNodes(i,2) << " ";
        outfield << tetNodes(i,3) << "\n";
    }
    outfield << "CELL_TYPES " << nTetras << "\n";
    for(size_t i = 0; i < nTetras; i++)
    {
        outfield << VtkTetCellType << "\n";
    }
    outfield << "CELL_DATA " << nTetras  << "\n";
    outfield << "SCALARS Domain float" << "\n";
    outfield << "LOOKUP_TABLE default" << "\n";
    for(size_t i = 0; i < nTetras; i++)
    {
        outfield << (float) tetDom(i) << "\n";
    }
    outfield.close();
}

void mesh::Reorder()
{
    std::cout << "Nested dissection reordering\n";
    int nn = (int) nNodes;

    // Build nodal graph from element connectivity
    std::vector<int> xadj, adjncy;
    buildNodalGraph(nn, tetNodes, xadj, adjncy);

    // Geometric nested dissection ordering
    std::vector<int> allNodes(nn);
    std::iota(allNodes.begin(), allNodes.end(), 0);
    std::vector<int> permVec;
    ndRecursive(allNodes, nodPos, xadj, adjncy, permVec);

    // Compute inverse permutation
    std::vector<int> ipermVec(nn);
    for (int i = 0; i < nn; i++)
        ipermVec[permVec[i]] = i;

    /// reconstruct mesh
    arma::mat newNodPos(nNodes,3);
    for(size_t nid = 0; nid < nNodes; nid++)
    {
        newNodPos.row(nid) = nodPos.row(permVec[nid]);
    }
    nodPos = newNodPos;
    arma::umat newTetNodes(nTetras,4);
    for(size_t tit = 0; tit < nTetras; tit++)
    {
        for(size_t nid=0; nid< 4; nid++)
        {
            newTetNodes(tit,nid) = (size_t)ipermVec[tetNodes(tit,nid)];
        }
        newTetNodes.row(tit) = arma::sort(newTetNodes.row(tit));
    }
    tetNodes = newTetNodes;
    arma::umat newFacNodes(nFaces,3);
    for(size_t tit = 0; tit < nFaces; tit++)
    {
        for(size_t nid=0; nid< 3; nid++)
        {
            newFacNodes(tit,nid) = (size_t)ipermVec[facNodes(tit,nid)];
        }
        newFacNodes.row(tit) = arma::sort(newFacNodes.row(tit));
    }
    facNodes = newFacNodes;
    arma::umat newEdgNodes(nEdges,2);
    for(size_t tit = 0; tit < nEdges; tit++)
    {
        for(size_t nid=0; nid< 2; nid++)
        {
            newEdgNodes(tit,nid) = (size_t)ipermVec[edgNodes(tit,nid)];
        }
        newEdgNodes.row(tit) = arma::sort(newEdgNodes.row(tit));
    }
    edgNodes = newEdgNodes;
}


arma::mat mesh::tet_geo(size_t id) const
{
    arma::mat cGeo(4,3);
    cGeo.row(0) = nodPos.row(tetNodes(id,0));
    cGeo.row(1) = nodPos.row(tetNodes(id,1));
    cGeo.row(2) = nodPos.row(tetNodes(id,2));
    cGeo.row(3) = nodPos.row(tetNodes(id,3));
    return cGeo;
}

arma::mat mesh::fac_geo(size_t id) const
{
    arma::mat cGeo(3,3);
    cGeo.row(0) = nodPos.row(facNodes(id,0));
    cGeo.row(1) = nodPos.row(facNodes(id,1));
    cGeo.row(2) = nodPos.row(facNodes(id,2));
    return cGeo;
}

arma::mat mesh::fac_geo2(size_t id) const
{
    arma::mat cGeo(3,3);
    cGeo.row(0) = nodPos.row(facNodes(id,0));
    cGeo.row(1) = nodPos.row(facNodes(id,1));
    cGeo.row(2) = nodPos.row(facNodes(id,2));
    arma::vec v0 = cGeo(0,arma::span::all).st();
    arma::vec v1 = cGeo(1,arma::span::all).st();
    arma::vec v2 = cGeo(2,arma::span::all).st();
    v1 -= v0;
    v2 -= v0;
    arma::vec u = v1 / arma::norm(v1,2);
    arma::vec n = arma::cross(v1,v2);
    n /= arma::norm(n,2);
    arma::vec v = arma::cross(n,u);
    arma::mat cGeo2(3,2);
    cGeo2.fill(0);
    cGeo2(1,0) = arma::dot((cGeo.row(1)-cGeo.row(0)).st(), u);
    cGeo2(2,0) = arma::dot((cGeo.row(2)-cGeo.row(0)).st(), u);
    cGeo2(2,1) = arma::dot((cGeo.row(2)-cGeo.row(0)).st(), v);
    return cGeo2;
}

arma::vec mesh::int_node(size_t id) const
{
    arma::vec nod(3);
    arma::uvec nfac = facNodes.row(id).st();
    arma::uvec ntet = tetNodes.row(facAdjTet(id)(0)).st();
    size_t intid;
    for(size_t i = 0; i<4; i++)
    {
        bool found = true;
        intid = ntet(i);
        for(size_t j = 0; j<3; j++)
            if(ntet(i) == nfac(j))
            {
                found = false;
            }
        if(found)
        {
            break;
        }
    }
    nod = nodPos.row(intid).st();
    return nod;
}

arma::vec mesh::int_node(size_t id, size_t& RefFace) const
{
    arma::vec nod(3);
    arma::uvec nfac = facNodes.row(id).st();
    arma::uvec ntet = tetNodes.row(facAdjTet(id)(RefFace)).st();
    size_t intid = 0;
    for(size_t i = 0; i<4; i++)
    {
        bool found = true;
        intid = ntet(i);
        for(size_t j = 0; j<3; j++)
            if(ntet(i) == nfac(j))
            {
                found = false;
            }
        if(found)
        {
            RefFace = i;
            break;
        }
    }
    nod = nodPos.row(intid).st();
    return nod;
}

// ── Store PLC from existing mesh (HFSS import path) ──

void mesh::store_plc_from_hfss()
{
    plc_valid = true;
    mesh_dim = 3;

    int np = (int)nNodes;
    plc_points.resize(np * 3);
    for(int i = 0; i < np; i++) {
        plc_points[i*3]     = nodPos(i, 0);
        plc_points[i*3 + 1] = nodPos(i, 1);
        plc_points[i*3 + 2] = nodPos(i, 2);
    }

    int nf = (int)nFaces;
    plc_facet_markers.resize(nf);
    plc_poly_vertex_counts.resize(nf);
    plc_poly_vertex_list.resize(nf * 3);
    for(int i = 0; i < nf; i++) {
        // Preserve boundary markers: use facLab label if set (non-sentinel)
        if((size_t)i < facLab.n_elem && facLab(i) != maxLab)
            plc_facet_markers[i] = (int)facLab(i);
        else
            plc_facet_markers[i] = 0;
        plc_poly_vertex_counts[i] = 3;
        plc_poly_vertex_list[i*3]     = (int)facNodes(i, 0) + 1;
        plc_poly_vertex_list[i*3 + 1] = (int)facNodes(i, 1) + 1;
        plc_poly_vertex_list[i*3 + 2] = (int)facNodes(i, 2) + 1;
    }

    // Region seeds from tet material centroids
    std::map<size_t, arma::rowvec> region_centroids;
    if(nTetras > 0) {
        for(size_t i = 0; i < nTetras; i++) {
            size_t lab = tetLab(i);
            arma::rowvec c(3, arma::fill::zeros);
            for(int j = 0; j < 4; j++)
                c += nodPos.row(tetNodes(i, j));
            c /= 4.0;
            if(region_centroids.find(lab) == region_centroids.end())
                region_centroids[lab] = c;
        }
        int nr = (int)region_centroids.size();
        plc_regions.resize(nr * 5);
        int idx = 0;
        for(auto& kv : region_centroids) {
            plc_regions[idx*5]     = kv.second(0);
            plc_regions[idx*5 + 1] = kv.second(1);
            plc_regions[idx*5 + 2] = kv.second(2);
            plc_regions[idx*5 + 3] = (double)kv.first;
            plc_regions[idx*5 + 4] = 0.0;
            idx++;
        }
    }
}
