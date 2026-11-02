// C++ oracle harness for fes.core.shape / shape3d / element_matrix / dof.
//
// Builds the *real* cFES classes (Shape.cpp, Quad.cpp, EleMat.cpp, DoF.cpp)
// and dumps every array the Python port has to reproduce, with 17 significant
// digits.  tools/parity/build_and_dump.py compiles and runs this file and
// writes tests/fixtures/cpp_oracle.json.
//
// The harness never writes into the cFES tree and does not import a mesh; the
// small meshes used by tests/test_dof.py and tests/test_dof3d.py are filled
// into the public Mesh members directly.

#include <armadillo>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "BC.h"
#include "DoF.h"
#include "EleMat.h"
#include "Mesh.h"
#include "Mtrl.h"
#include "Option.h"
#include "Project.h"
#include "Quad.h"
#include "Shape.h"

namespace
{

typedef std::vector<std::pair<std::string, std::string> > Obj;

std::string obj(const Obj& entries)
{
    std::ostringstream os;
    os << "{";
    for (size_t i = 0; i < entries.size(); ++i)
    {
        if (i) os << ", ";
        os << "\"" << entries[i].first << "\": " << entries[i].second;
    }
    os << "}";
    return os.str();
}

// Row-major flat dump; the Python fixture loader reshapes with order="C".
std::string matjson(const arma::mat& m)
{
    std::ostringstream os;
    os << std::setprecision(17);
    os << "{\"rows\": " << m.n_rows << ", \"cols\": " << m.n_cols
       << ", \"data\": [";
    bool first = true;
    for (size_t i = 0; i < m.n_rows; ++i)
    {
        for (size_t j = 0; j < m.n_cols; ++j)
        {
            if (!first) os << ", ";
            first = false;
            os << m(i, j);
        }
    }
    os << "]}";
    return os.str();
}

std::string cxmatjson(const arma::cx_mat& m)
{
    std::ostringstream os;
    os << std::setprecision(17);
    os << "{\"rows\": " << m.n_rows << ", \"cols\": " << m.n_cols
       << ", \"data\": [";
    bool first = true;
    for (size_t i = 0; i < m.n_rows; ++i)
    {
        for (size_t j = 0; j < m.n_cols; ++j)
        {
            if (!first) os << ", ";
            first = false;
            std::complex<double> z = m(i, j);
            os << "[" << z.real() << ", " << z.imag() << "]";
        }
    }
    os << "]}";
    return os.str();
}

std::string vecjson(const arma::vec& v)
{
    arma::mat m(v.n_elem, 1);
    m.col(0) = v;
    return matjson(m);
}

std::string cxvecjson(const arma::cx_vec& v)
{
    arma::cx_mat m(v.n_elem, 1);
    m.col(0) = v;
    return cxmatjson(m);
}

std::string uvecjson(const arma::uvec& v)
{
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < v.n_elem; ++i)
    {
        if (i) os << ", ";
        os << v(i);
    }
    os << "]";
    return os.str();
}

std::string rowsjson(const std::vector<std::vector<double> >& rows)
{
    std::ostringstream os;
    os << std::setprecision(17);
    os << "[";
    for (size_t i = 0; i < rows.size(); ++i)
    {
        if (i) os << ", ";
        os << "[";
        for (size_t j = 0; j < rows[i].size(); ++j)
        {
            if (j) os << ", ";
            os << rows[i][j];
        }
        os << "]";
    }
    os << "]";
    return os.str();
}

arma::mat matFromRows(const std::vector<std::vector<double> >& rows)
{
    arma::mat m(rows.size(), rows.empty() ? 0 : rows[0].size());
    for (size_t i = 0; i < rows.size(); ++i)
        for (size_t j = 0; j < rows[i].size(); ++j)
            m(i, j) = rows[i][j];
    return m;
}

arma::umat umatFromRows(const std::vector<std::vector<int> >& rows)
{
    arma::umat m(rows.size(), rows.empty() ? 0 : rows[0].size());
    for (size_t i = 0; i < rows.size(); ++i)
        for (size_t j = 0; j < rows[i].size(); ++j)
            m(i, j) = static_cast<arma::uword>(rows[i][j]);
    return m;
}

// ── quadrature ──────────────────────────────────────────────────────────────

std::string quad2djson()
{
    Obj entries;
    for (size_t p = 1; p <= 5; ++p)
    {
        Quad q(p);
        std::ostringstream name;
        name << p;
        entries.push_back(std::make_pair(
            name.str(), obj({{"xq", matjson(q.xq2)}, {"wq", vecjson(q.wq2)}})));
    }
    return obj(entries);
}

std::string quad3djson()
{
    Obj entries;
    for (size_t p = 1; p <= 5; ++p)
    {
        Quad q(p);
        std::ostringstream name;
        name << p;
        entries.push_back(std::make_pair(
            name.str(), obj({{"xq", matjson(q.xq3)}, {"wq", vecjson(q.wq3)}})));
    }
    return obj(entries);
}

// ── shape functions ─────────────────────────────────────────────────────────

std::string shape2djson(size_t p, const arma::mat& geo)
{
    Jacobian jac(2, geo);
    Quad q(p + 1);
    const size_t nq = q.wq2.n_rows;

    Shape first(p, 2, Shape::Hcurl, q.xq2.row(0), &jac);
    const size_t nums = first.Ns.n_cols;
    const size_t numv = first.Nv.n_cols;

    arma::mat Ns(nq, nums), dNs0(nq, nums), dNs1(nq, nums);
    arma::mat Nv0(nq, numv), Nv1(nq, numv), dNv(nq, numv);
    for (size_t iq = 0; iq < nq; ++iq)
    {
        Shape shp(p, 2, Shape::Hcurl, q.xq2.row(iq), &jac);
        Ns.row(iq) = shp.Ns;
        dNs0.row(iq) = shp.dNs.row(0);
        dNs1.row(iq) = shp.dNs.row(1);
        Nv0.row(iq) = shp.Nv.row(0);
        Nv1.row(iq) = shp.Nv.row(1);
        dNv.row(iq) = shp.dNv.row(0);
    }
    return obj({{"Ns", matjson(Ns)},
                {"dNs_0", matjson(dNs0)},
                {"dNs_1", matjson(dNs1)},
                {"Nv_0", matjson(Nv0)},
                {"Nv_1", matjson(Nv1)},
                {"dNv", matjson(dNv)}});
}

std::string shape3djson(size_t p, const arma::mat& geo)
{
    Jacobian jac(3, geo);
    Quad q(p + 1);
    const size_t nq = q.wq3.n_rows;

    Shape hg0(p, 3, Shape::Hgrad, q.xq3.row(0), &jac);
    const size_t nums = hg0.Ns.n_cols;
    arma::mat hgNs(nq, nums);
    arma::mat hg0x(nq, nums), hg0y(nq, nums), hg0z(nq, nums);
    for (size_t iq = 0; iq < nq; ++iq)
    {
        Shape shp(p, 3, Shape::Hgrad, q.xq3.row(iq), &jac);
        hgNs.row(iq) = shp.Ns;
        hg0x.row(iq) = shp.dNs.row(0);
        hg0y.row(iq) = shp.dNs.row(1);
        hg0z.row(iq) = shp.dNs.row(2);
    }

    Shape hc0(p, 3, Shape::Hcurl, q.xq3.row(0), &jac);
    const size_t numv = hc0.Nv.n_cols;
    arma::mat hc0x(nq, numv), hc0y(nq, numv), hc0z(nq, numv);
    arma::mat hd0x(nq, numv), hd0y(nq, numv), hd0z(nq, numv);
    for (size_t iq = 0; iq < nq; ++iq)
    {
        Shape shp(p, 3, Shape::Hcurl, q.xq3.row(iq), &jac);
        hc0x.row(iq) = shp.Nv.row(0);
        hc0y.row(iq) = shp.Nv.row(1);
        hc0z.row(iq) = shp.Nv.row(2);
        hd0x.row(iq) = shp.dNv.row(0);
        hd0y.row(iq) = shp.dNv.row(1);
        hd0z.row(iq) = shp.dNv.row(2);
    }

    return obj(
        {{"hgrad",
          obj({{"Ns", matjson(hgNs)},
               {"dNs_0", matjson(hg0x)},
               {"dNs_1", matjson(hg0y)},
               {"dNs_2", matjson(hg0z)}})},
         {"hcurl",
          obj({{"Nv_0", matjson(hc0x)},
               {"Nv_1", matjson(hc0y)},
               {"Nv_2", matjson(hc0z)},
               {"dNv_0", matjson(hd0x)},
               {"dNv_1", matjson(hd0y)},
               {"dNv_2", matjson(hd0z)}})}});
}

std::string shape2dAll(const std::map<std::string, arma::mat>& geos)
{
    Obj geoEntries;
    for (std::map<std::string, arma::mat>::const_iterator it = geos.begin();
         it != geos.end(); ++it)
    {
        Obj orders;
        for (size_t p = 1; p <= 4; ++p)
        {
            std::ostringstream name;
            name << p;
            orders.push_back(
                std::make_pair(name.str(), shape2djson(p, it->second)));
        }
        geoEntries.push_back(std::make_pair(it->first, obj(orders)));
    }
    return obj(geoEntries);
}

std::string shape3dAll(const std::map<std::string, arma::mat>& geos)
{
    Obj geoEntries;
    for (std::map<std::string, arma::mat>::const_iterator it = geos.begin();
         it != geos.end(); ++it)
    {
        Obj orders;
        for (size_t p = 1; p <= 4; ++p)
        {
            std::ostringstream name;
            name << p;
            orders.push_back(
                std::make_pair(name.str(), shape3djson(p, it->second)));
        }
        geoEntries.push_back(std::make_pair(it->first, obj(orders)));
    }
    return obj(geoEntries);
}

// ── element matrices ────────────────────────────────────────────────────────

std::string elemat3djson(size_t p, const arma::mat& geo, Mtrl& m)
{
    Quad q(p + 1);
    EleMat hcurl(p, 3, geo, &q, &m, Shape::Hcurl);
    EleMat hgrad(p, 3, geo, &q, &m, Shape::Hgrad);
    return obj({{"S", cxmatjson(hcurl.S)},
                {"T", cxmatjson(hcurl.T)},
                {"Z", cxmatjson(hcurl.Z)},
                {"Shgrad", cxmatjson(hgrad.S)}});
}

std::string elemat2djson(size_t p, const arma::mat& geo, Mtrl& m)
{
    Quad q(p + 1);
    // The two-dimensional bulk assembly lives in the STYPE constructor; the
    // element type is unused when cDim == 2 (EleMat.cpp:64-115).
    EleMat em(p, 2, geo, &q, &m, Shape::Hcurl);
    return obj({{"St", cxmatjson(em.St)},
                {"Tt", cxmatjson(em.Tt)},
                {"Sz", cxmatjson(em.Sz)},
                {"Tz", cxmatjson(em.Tz)},
                {"G", cxmatjson(em.G)}});
}

std::string elemfacejson(size_t p, const arma::mat& geo, const arma::vec& intNode,
                         Mtrl& m)
{
    Quad q(p + 1);
    EleMat em(p, 2, geo, &q, &m, intNode);
    return obj({{"St", cxmatjson(em.St)},
                {"Tt", cxmatjson(em.Tt)},
                {"Sz", cxmatjson(em.Sz)},
                {"Tz", cxmatjson(em.Tz)},
                {"G", cxmatjson(em.G)},
                {"STt", cxmatjson(em.STt)}});
}

std::string eincjson(size_t p, const arma::mat& geo, const arma::vec& intNode,
                     Mtrl& m, const arma::vec& kEinc, const arma::vec& polEinc)
{
    Quad q(p + 1);
    EleMat em(p, 2, geo, &q, &m, intNode, kEinc, polEinc);
    return obj({{"f", cxvecjson(em.f)}});
}

template <typename Fn>
std::string byOrder(Fn fn)
{
    Obj entries;
    for (size_t p = 1; p <= 4; ++p)
    {
        std::ostringstream name;
        name << p;
        entries.push_back(std::make_pair(name.str(), fn(p)));
    }
    return obj(entries);
}

// ── degrees of freedom ──────────────────────────────────────────────────────

struct MeshSpec
{
    arma::umat tetNodes, tetEdges, tetFaces;
    arma::umat facNodes, facEdges;
    size_t nNodes, nEdges, nFaces, nTetras;
};

void loadMesh(Mesh* msh, const MeshSpec& spec)
{
    msh->nNodes = spec.nNodes;
    msh->nEdges = spec.nEdges;
    msh->nFaces = spec.nFaces;
    msh->nTetras = spec.nTetras;
    msh->tetNodes = spec.tetNodes;
    msh->tetEdges = spec.tetEdges;
    msh->tetFaces = spec.tetFaces;
    msh->facNodes = spec.facNodes;
    msh->facEdges = spec.facEdges;
}

std::string dofElementJson(Project& prj, size_t dim, size_t id, size_t p)
{
    prj.opt->pOrd = p;
    DoF dof(&prj, dim, id);
    return obj({{"s", uvecjson(dof.s)}, {"v", uvecjson(dof.v)}});
}

std::string dofAll(Project& prj, size_t dim, size_t nElements)
{
    Obj elements;
    for (size_t id = 0; id < nElements; ++id)
    {
        std::ostringstream idName;
        idName << id;
        Obj orders;
        for (size_t p = 1; p <= 4; ++p)
        {
            std::ostringstream pName;
            pName << p;
            orders.push_back(
                std::make_pair(pName.str(), dofElementJson(prj, dim, id, p)));
        }
        elements.push_back(std::make_pair(idName.str(), obj(orders)));
    }
    return obj(elements);
}

std::string dofCounts(Project& prj)
{
    Obj entries;
    for (size_t p = 1; p <= 4; ++p)
    {
        prj.opt->pOrd = p;
        DoF dof(&prj);
        std::ostringstream name;
        name << p;
        std::ostringstream body;
        body << "{\"v\": " << dof.DoFnumv << ", \"s\": " << dof.DoFnums << "}";
        entries.push_back(std::make_pair(name.str(), body.str()));
    }
    return obj(entries);
}

} // namespace

int main()
{
    // ── geometries ──────────────────────────────────────────────────────────
    const arma::mat tri_ref = matFromRows({{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}});
    const arma::mat tri_skew =
        matFromRows({{0.10, -0.20}, {1.20, 0.30}, {0.40, 1.10}});

    const arma::mat tet_ref = matFromRows(
        {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}});
    const arma::mat tet_skew = matFromRows({{0.10, -0.20, 0.05},
                                            {1.20, 0.30, 0.10},
                                            {0.40, 1.10, 0.15},
                                            {0.20, 0.10, 1.00}});

    // First three vertices of tet_skew form the boundary face; the fourth
    // vertex is the interior node used to orient its normal.
    arma::mat face_geo(3, 3);
    for (size_t i = 0; i < 3; ++i)
        face_geo.row(i) = tet_skew.row(i);
    arma::vec face_int = tet_skew.row(3).t();

    std::map<std::string, arma::mat> triGeos;
    triGeos["tri_ref"] = tri_ref;
    triGeos["tri_skew"] = tri_skew;

    std::map<std::string, arma::mat> tetGeos;
    tetGeos["tet_ref"] = tet_ref;
    tetGeos["tet_skew"] = tet_skew;

    // ── material and incident wave ──────────────────────────────────────────
    Mtrl mtrl("solid", "mat", 2.5, 1.3, 0.02, 0.001);
    arma::vec kEinc = matFromRows({{0.0}, {-2.1}, {3.7}}).col(0);
    arma::vec polEinc = matFromRows({{0.3}, {-1.1}, {0.7}}).col(0);

    // ── meshes for the DoF maps (same as tests/test_dof*.py) ────────────────
    // 2D: unit square split into two triangles, nN=4, nE=5, nF=2.
    MeshSpec triMesh;
    triMesh.nNodes = 4;
    triMesh.nEdges = 5;
    triMesh.nFaces = 2;
    triMesh.nTetras = 0;
    triMesh.tetNodes = umatFromRows({});
    triMesh.tetEdges = umatFromRows({});
    triMesh.tetFaces = umatFromRows({});
    triMesh.facNodes = umatFromRows({{0, 1, 2}, {1, 2, 3}});
    triMesh.facEdges = umatFromRows({{0, 3, 4}, {4, 1, 2}});

    // 3D: two tetrahedra sharing (0,1,2), shuffled global edge/face numbering.
    MeshSpec tetMesh;
    tetMesh.nNodes = 5;
    tetMesh.nEdges = 9;
    tetMesh.nFaces = 7;
    tetMesh.nTetras = 2;
    tetMesh.tetNodes = umatFromRows({{0, 1, 2, 3}, {0, 1, 2, 4}});
    tetMesh.tetEdges =
        umatFromRows({{2, 5, 1, 0, 6, 8}, {2, 5, 7, 0, 4, 3}});
    tetMesh.tetFaces = umatFromRows({{0, 5, 4, 6}, {3, 2, 1, 6}});
    const std::vector<std::vector<int> > edges = {
        {1, 2}, {0, 3}, {0, 1}, {2, 4}, {1, 4},
        {0, 2}, {1, 3}, {0, 4}, {2, 3}};
    const std::vector<std::vector<int> > faces = {
        {1, 2, 3}, {0, 1, 4}, {0, 2, 4}, {1, 2, 4},
        {0, 1, 3}, {0, 2, 3}, {0, 1, 2}};
    tetMesh.facNodes = umatFromRows(faces);
    std::map<std::pair<int, int>, int> edgeId;
    for (size_t e = 0; e < edges.size(); ++e)
    {
        int a = edges[e][0], b = edges[e][1];
        if (a > b) std::swap(a, b);
        edgeId[std::make_pair(a, b)] = static_cast<int>(e);
    }
    std::vector<std::vector<int> > faceEdges(faces.size(),
                                             std::vector<int>(3, 0));
    for (size_t f = 0; f < faces.size(); ++f)
    {
        const int n0 = faces[f][0], n1 = faces[f][1], n2 = faces[f][2];
        const int pairs[3][2] = {{n1, n2}, {n0, n2}, {n0, n1}};
        for (size_t c = 0; c < 3; ++c)
        {
            int a = pairs[c][0], b = pairs[c][1];
            if (a > b) std::swap(a, b);
            std::map<std::pair<int, int>, int>::const_iterator it =
                edgeId.find(std::make_pair(a, b));
            faceEdges[f][c] = (it == edgeId.end()) ? 0 : it->second;
        }
    }
    tetMesh.facEdges = umatFromRows(faceEdges);

    std::ofstream nullLog("/dev/null");
    Option opt;
    Project prj2d(nullLog, opt);
    Project prj3d(nullLog, opt);
    loadMesh(prj2d.msh, triMesh);
    loadMesh(prj3d.msh, tetMesh);

    // ── emit ────────────────────────────────────────────────────────────────
    std::cout << std::setprecision(17);
    std::cout << "{\n";
    std::cout << "  \"geo\": "
              << obj({{"tri_ref", rowsjson({{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}})},
                      {"tri_skew",
                       rowsjson({{0.10, -0.20}, {1.20, 0.30}, {0.40, 1.10}})},
                      {"tet_ref", rowsjson({{0.0, 0.0, 0.0},
                                            {1.0, 0.0, 0.0},
                                            {0.0, 1.0, 0.0},
                                            {0.0, 0.0, 1.0}})},
                      {"tet_skew", rowsjson({{0.10, -0.20, 0.05},
                                             {1.20, 0.30, 0.10},
                                             {0.40, 1.10, 0.15},
                                             {0.20, 0.10, 1.00}})},
                      {"face_geo", rowsjson({{0.10, -0.20, 0.05},
                                             {1.20, 0.30, 0.10},
                                             {0.40, 1.10, 0.15}})},
                      {"face_int_node", rowsjson({{0.20, 0.10, 1.00}})},
                      {"material",
                       obj({{"epsr", "2.5"},
                            {"mur", "1.3"},
                            {"sigma", "0.02"},
                            {"tand", "0.001"}})},
                      {"k_einc", rowsjson({{0.0, -2.1, 3.7}})},
                      {"pol_einc", rowsjson({{0.3, -1.1, 0.7}})},
                      {"dof_mesh2d",
                       obj({{"n_nodes", "4"},
                            {"n_edges", "5"},
                            {"n_faces", "2"},
                            {"n_tetras", "0"},
                            {"fac_nodes", "[[0, 1, 2], [1, 2, 3]]"},
                            {"fac_edges", "[[0, 3, 4], [4, 1, 2]]"}})},
                      {"dof_mesh3d",
                       obj({{"n_nodes", "5"},
                            {"n_edges", "9"},
                            {"n_faces", "7"},
                            {"n_tetras", "2"},
                            {"tet_nodes", "[[0, 1, 2, 3], [0, 1, 2, 4]]"},
                            {"fac_nodes",
                             "[[1, 2, 3], [0, 1, 4], [0, 2, 4], [1, 2, 4], "
                             "[0, 1, 3], [0, 2, 3], [0, 1, 2]]"},
                            {"fac_edges", "[" +
                                              [&]() {
                                                  std::ostringstream os;
                                                  for (size_t f = 0;
                                                       f < faceEdges.size();
                                                       ++f)
                                                  {
                                                      if (f) os << ", ";
                                                      os << "["
                                                         << faceEdges[f][0]
                                                         << ", " << faceEdges[f][1]
                                                         << ", " << faceEdges[f][2]
                                                         << "]";
                                                  }
                                                  return os.str();
                                              }() +
                                              "]"}})}})
              << ",\n";

    std::cout << "  \"quad2d\": " << quad2djson() << ",\n";
    std::cout << "  \"quad3d\": " << quad3djson() << ",\n";
    std::cout << "  \"shape2d\": " << shape2dAll(triGeos) << ",\n";
    std::cout << "  \"shape3d\": " << shape3dAll(tetGeos) << ",\n";

    std::cout << "  \"elemat3d\": " << byOrder([&](size_t p) {
        return elemat3djson(p, tet_skew, mtrl);
    }) << ",\n";
    std::cout << "  \"elemat2d\": " << byOrder([&](size_t p) {
        return elemat2djson(p, tri_skew, mtrl);
    }) << ",\n";
    std::cout << "  \"elemface\": " << byOrder([&](size_t p) {
        return elemfacejson(p, face_geo, face_int, mtrl);
    }) << ",\n";
    std::cout << "  \"einc\": " << byOrder([&](size_t p) {
        return eincjson(p, face_geo, face_int, mtrl, kEinc, polEinc);
    }) << ",\n";

    std::cout << "  \"dof2d\": " << dofAll(prj2d, 2, 2) << ",\n";
    std::cout << "  \"dof3d\": " << dofAll(prj3d, 3, 2) << ",\n";
    std::cout << "  \"dof3dface\": " << dofAll(prj3d, 2, 7) << ",\n";
    std::cout << "  \"counts2d\": " << dofCounts(prj2d) << ",\n";
    std::cout << "  \"counts3d\": " << dofCounts(prj3d) << "\n";
    std::cout << "}\n";
    return 0;
}
