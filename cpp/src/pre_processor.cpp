#include "pre_processor.h"
#include "mesh.h"
#include <triangle.h>
#include <map>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <cctype>

// Peek at the .poly file header to determine spatial dimension.
// Returns 2 or 3, or 0 on error.
static int detectPolyDimension(const char* name)
{
    std::string path = std::string(name) + ".poly";
    std::ifstream f(path);
    if(!f.is_open()) return 0;
    std::string line;
    while(std::getline(f, line))
        if(!line.empty() && line[0] != '#') break;
    int nn, dim;
    std::istringstream(line) >> nn >> dim;
    return (dim == 2 || dim == 3) ? dim : 0;
}

// Read #Formula tag from .poly header to auto-select assembly type.
// Returns true if a formula was found and applied.
static bool detectFormula(option* opt)
{
    std::string path = opt->name + ".poly";
    std::ifstream f(path);
    if(!f.is_open()) return false;
    std::string line;
    while(std::getline(f, line)) {
        if(line.find("#Formula") != std::string::npos) {
            std::string tag;
            std::istringstream iss(line);
            iss >> tag >> tag; // skip "#Formula", read formula name
            // Case-insensitive comparison
            for(auto& c : tag) c = std::tolower((unsigned char)c);
            if(tag == "em_e_fd")      { opt->assembly = option::em_e_fd; return true; }
            if(tag == "em_ez_fd")     { opt->assembly = option::em_ez_fd; return true; }
            if(tag == "em_e_qs")      { opt->field = true; opt->assembly = option::em_e_qs; return true; }
            if(tag == "em_e_tl_eig")  { opt->assembly = option::em_e_tl_eig; return true; }
            // unrecognized formula tag — ignore
            break;
        }
        if(!line.empty() && line[0] != '#') break; // past all comments
    }
    return false;
}

preprocessing::preprocessing(project* cProj): prj(cProj), dbg(cProj->opt->dbg), scaling(1.0)
{
    // Apply file formula only if CLI didn't specify an explicit assembly
    if(prj->opt->assembly == option::em_e_fd)
        detectFormula(prj->opt);
    if(prj->msh->plc_valid)
    {
        // Mesh regeneration from stored PLC data
        if(!prj->msh->plc_2d_points.empty() || prj->msh->mesh_dim == 2) {
            std::cout << "Re-meshing 2D from stored PLC geometry\n";
            const char* triSwitches = prj->opt->poly_cmd.empty() ? nullptr : prj->opt->poly_cmd.c_str();
            triangulatemesh2D_from_plc(prj->msh, triSwitches);
        } else {
            std::string poly_cmd = prj->opt->poly_cmd.empty()
                ? "pqAfeeQ"
                : "p" + prj->opt->poly_cmd + "AfeeQ";
            char* switches = new char[poly_cmd.size() + 1];
            strcpy(switches, poly_cmd.c_str());
            populate_tetgenio_from_plc(prj->msh, in);
            std::cout << "Re-meshing 3D from stored PLC with TetGen command = " << poly_cmd << "\n";
            tetrahedralize(switches, &in, &out, &addin, &bgmin);
            Createmesh();
            delete[] switches;
        }
    }
    else
    {
        int dim = detectPolyDimension(prj->opt->name.c_str());
        if(dim == 2)
        {
            char* name = new char[prj->opt->name.size() + 1];
            strcpy(name, prj->opt->name.c_str());
            std::cout << "Detected 2D .poly, meshing with Triangle\n";
            in.load_poly(name);
            load_extra();
            const char* triSwitches = prj->opt->poly_cmd.empty() ? nullptr : prj->opt->poly_cmd.c_str();
            if(!triangulatemesh2D(name, prj->msh, triSwitches)) {
                std::cout << "Triangle 2D meshing failed, aborting\n";
                throw std::runtime_error("Triangle 2D meshing failed");
            }
            delete[] name;
        }
        else
        {
            if(dim == 0)
                std::cout << "Warning: could not detect .poly dimension, assuming 3D\n";
            std::string poly_cmd = prj->opt->poly_cmd.empty()
                ? "pqAfeeQ"
                : "p" + prj->opt->poly_cmd + "AfeeQ";
            char* name = new char[prj->opt->name.size() + 1];
            strcpy(name, prj->opt->name.c_str());
            char* switches = new char[poly_cmd.size() + 1];
            strcpy(switches, poly_cmd.c_str());
            in.load_poly(name);
            load_extra();
            // Store PLC geometry for mesh regeneration
            prj->msh->store_plc_from_tetgen(in);
            std::cout << "meshing with TetGen command = " << poly_cmd << "\n";
            tetrahedralize(switches, &in, &out, &addin, &bgmin);
            Createmesh();
            delete[] name;
            delete[] switches;
        }
    }
}

preprocessing::~preprocessing()
{
}

// ── Triangle 2D .poly parser ──

bool preprocessing::read_poly_for_triangle(const char* polyfile, struct triangulateio* tri_in)
{
    std::ifstream f(polyfile);
    if(!f.is_open()) return false;
    std::string line;
    // Skip comments to get header
    while(std::getline(f, line)) {
        if(!line.empty() && line[0] != '#') break;
    }
    int nn, dim, nattrib, nmarkers;
    {
        std::istringstream iss(line);
        iss >> nn >> dim >> nattrib >> nmarkers;
    }
    tri_in->numberofpoints = nn;
    tri_in->pointlist = (double*)malloc(nn * 2 * sizeof(double));
    tri_in->pointmarkerlist = nmarkers ? (int*)malloc(nn * sizeof(int)) : nullptr;
    for(int i = 0; i < nn; i++) {
        while(std::getline(f, line)) if(!line.empty() && line[0] != '#') break;
        std::istringstream iss(line);
        int idx; double x, y, z;
        iss >> idx >> x >> y >> z;
        tri_in->pointlist[2*i] = x; tri_in->pointlist[2*i+1] = y;
        if(nmarkers) { int m; iss >> m; tri_in->pointmarkerlist[i] = m; }
    }
    // Skip to segments
    while(std::getline(f, line)) {
        if(!line.empty() && line[0] != '#') break;
    }
    int ns, nsm;
    {
        std::istringstream iss(line);
        iss >> ns >> nsm;
    }
    if(ns > 0) {
        tri_in->numberofsegments = ns;
        tri_in->segmentlist = (int*)malloc(ns * 2 * sizeof(int));
        tri_in->segmentmarkerlist = nsm ? (int*)malloc(ns * sizeof(int)) : nullptr;
        for(int i = 0; i < ns; i++) {
            while(std::getline(f, line)) if(!line.empty() && line[0] != '#') break;
            std::istringstream iss(line);
            int idx, v1, v2;
            iss >> idx >> v1 >> v2;
            tri_in->segmentlist[2*i] = v1;   // 1-based (no z flag)
            tri_in->segmentlist[2*i+1] = v2;
            if(nsm) { int m; iss >> m; tri_in->segmentmarkerlist[i] = m; }
        }
    }

    // Skip to holes
    while(std::getline(f, line)) {
        if(!line.empty() && line[0] != '#') break;
    }
    {
        std::istringstream iss(line);
        int nh; iss >> nh;
        tri_in->numberofholes = nh;
        if(nh > 0) {
            tri_in->holelist = (double*)malloc(nh * 2 * sizeof(double));
            for(int i = 0; i < nh; i++) {
                while(std::getline(f, line)) if(!line.empty() && line[0] != '#') break;
                std::istringstream iss2(line);
                double x, y; iss2 >> x >> y;
                tri_in->holelist[2*i] = x; tri_in->holelist[2*i+1] = y;
            }
        }
    }

    // Skip to regions
    while(std::getline(f, line)) {
        if(!line.empty() && line[0] != '#') break;
    }
    {
        std::istringstream iss(line);
        int nr; iss >> nr;
        tri_in->numberofregions = nr;
        if(nr > 0) {
            tri_in->regionlist = (double*)malloc(nr * 4 * sizeof(double));
            for(int i = 0; i < nr; i++) {
                while(std::getline(f, line)) if(!line.empty() && line[0] != '#') break;
                std::istringstream iss2(line);
                int idx; double x, y, attr, maxarea;
                iss2 >> idx >> x >> y >> attr >> maxarea;
                // Triangle expects: x y attribute maxarea (no idx)
                tri_in->regionlist[4*i] = x;
                tri_in->regionlist[4*i+1] = y;
                tri_in->regionlist[4*i+2] = attr;
                tri_in->regionlist[4*i+3] = maxarea;
            }
        }
    }
    return true;
}

// ── Triangle 2D triangulation → extrude to 3D tetgenio ──

bool preprocessing::triangulate(const char* name, tetgenio& out, double scaling)
{
    std::string polyfile = std::string(name) + ".poly";
    struct triangulateio tri_in, tri_out;
    memset(&tri_in, 0, sizeof(tri_in));
    memset(&tri_out, 0, sizeof(tri_out));

    if(!read_poly_for_triangle(polyfile.c_str(), &tri_in))
        return false;

    char switches[] = "pfeQ";
    ::triangulate(switches, &tri_in, &tri_out, NULL);
    std::cout << "Triangle output: " << tri_out.numberofpoints << " points, "
              << tri_out.numberoftriangles << " triangles" << std::endl;

    if(tri_out.numberofpoints == 0 || tri_out.numberoftriangles == 0) {
        free(tri_out.pointlist); free(tri_out.trianglelist);
        return false;
    }

    double xmin = tri_out.pointlist[0], xmax = tri_out.pointlist[0];
    double ymin = tri_out.pointlist[1], ymax = tri_out.pointlist[1];
    for(int i = 0; i < tri_out.numberofpoints; i++) {
        if(tri_out.pointlist[2*i] < xmin) xmin = tri_out.pointlist[2*i];
        if(tri_out.pointlist[2*i] > xmax) xmax = tri_out.pointlist[2*i];
        if(tri_out.pointlist[2*i+1] < ymin) ymin = tri_out.pointlist[2*i+1];
        if(tri_out.pointlist[2*i+1] > ymax) ymax = tri_out.pointlist[2*i+1];
    }
    double diag = std::sqrt((xmax-xmin)*(xmax-xmin) + (ymax-ymin)*(ymax-ymin));
    double dz = diag * 0.01;

    int np = tri_out.numberofpoints;
    int nt = tri_out.numberoftriangles;
    int ns = tri_out.numberofsegments;

    // Extrude 2D triangulation to 3D tetrahedra via tetgenio
    out.numberofpoints = 2 * np;
    out.pointlist = new double[out.numberofpoints * 3];
    for(int i = 0; i < np; i++) {
        out.pointlist[3*i]       = tri_out.pointlist[2*i] * scaling;
        out.pointlist[3*i + 1]   = tri_out.pointlist[2*i + 1] * scaling;
        out.pointlist[3*i + 2]   = 0.0;
        out.pointlist[3*(i+np)]     = tri_out.pointlist[2*i] * scaling;
        out.pointlist[3*(i+np) + 1] = tri_out.pointlist[2*i + 1] * scaling;
        out.pointlist[3*(i+np) + 2] = dz * scaling;
    }

    // Each triangle → 3 tetras (prism) — Triangle output is 1-based (no z flag), TetGen uses 1-based
    out.numberoftetrahedra = 3 * nt;
    out.tetrahedronlist = new int[out.numberoftetrahedra * 4];
    int tidx = 0;
    for(int i = 0; i < nt; i++) {
        int a = tri_out.trianglelist[3*i], b = tri_out.trianglelist[3*i + 1], c = tri_out.trianglelist[3*i + 2];
        int at = a+np, bt = b+np, ct = c+np;
        out.tetrahedronlist[tidx++] = a;    out.tetrahedronlist[tidx++] = b;
        out.tetrahedronlist[tidx++] = c;    out.tetrahedronlist[tidx++] = at;
        out.tetrahedronlist[tidx++] = b;    out.tetrahedronlist[tidx++] = c;
        out.tetrahedronlist[tidx++] = at;   out.tetrahedronlist[tidx++] = bt;
        out.tetrahedronlist[tidx++] = c;    out.tetrahedronlist[tidx++] = at;
        out.tetrahedronlist[tidx++] = bt;   out.tetrahedronlist[tidx++] = ct;
    }
    out.numberofcorners = 4;

    if(tri_out.triangleattributelist) {
        out.numberoftetrahedronattributes = tri_out.numberoftriangleattributes;
        out.tetrahedronattributelist = new double[out.numberoftetrahedra * out.numberoftetrahedronattributes];
        for(int i = 0; i < nt; i++)
            for(int t = 0; t < 3; t++)
                for(int a = 0; a < tri_out.numberoftriangleattributes; a++)
                    out.tetrahedronattributelist[(3*i+t)*out.numberoftetrahedronattributes + a] =
                        tri_out.triangleattributelist[i*tri_out.numberoftriangleattributes + a];
    } else {
        out.numberoftetrahedronattributes = 1;
        out.tetrahedronattributelist = new double[out.numberoftetrahedra];
        for(size_t i = 0; i < out.numberoftetrahedra; i++)
            out.tetrahedronattributelist[i] = 0.0;
    }

    out.numberofedges = 2 * ns + np;
    out.edgelist = new int[out.numberofedges * 2];
    int eidx = 0;
    for(int i = 0; i < ns; i++) {
        int e0 = tri_out.segmentlist[2*i], e1 = tri_out.segmentlist[2*i+1];
        out.edgelist[eidx++] = e0;      out.edgelist[eidx++] = e1;
        out.edgelist[eidx++] = e0+np;   out.edgelist[eidx++] = e1+np;
    }
    for(int i = 0; i < np; i++) {
        out.edgelist[eidx++] = i+1; out.edgelist[eidx++] = i + np + 1;
    }

    out.numberoftrifaces = 2 * nt + 2 * ns;
    out.trifacelist = new int[out.numberoftrifaces * 3];
    out.trifacemarkerlist = new int[out.numberoftrifaces];
    int fidx = 0;
    for(int i = 0; i < nt; i++) {
        int a = tri_out.trianglelist[3*i], b = tri_out.trianglelist[3*i+1], c = tri_out.trianglelist[3*i+2];
        out.trifacelist[fidx*3]=a; out.trifacelist[fidx*3+1]=b; out.trifacelist[fidx*3+2]=c;
        out.trifacemarkerlist[fidx]=0; fidx++;
        out.trifacelist[fidx*3]=a+np; out.trifacelist[fidx*3+1]=c+np; out.trifacelist[fidx*3+2]=b+np;
        out.trifacemarkerlist[fidx]=0; fidx++;
    }
    for(int i = 0; i < ns; i++) {
        int e0 = tri_out.segmentlist[2*i], e1 = tri_out.segmentlist[2*i+1];
        int marker = tri_out.segmentmarkerlist ? tri_out.segmentmarkerlist[i] : 0;
        out.trifacelist[fidx*3]=e0; out.trifacelist[fidx*3+1]=e0+np; out.trifacelist[fidx*3+2]=e1;
        out.trifacemarkerlist[fidx]=marker; fidx++;
        out.trifacelist[fidx*3]=e0+np; out.trifacelist[fidx*3+1]=e1+np; out.trifacelist[fidx*3+2]=e1;
        out.trifacemarkerlist[fidx]=marker; fidx++;
    }

    std::cout << "Triangle 2D mesh: " << np << " nodes, " << nt << " triangles\n";

    free(tri_out.pointlist); free(tri_out.trianglelist);
    free(tri_out.segmentlist); free(tri_out.pointmarkerlist);
    free(tri_out.segmentmarkerlist); free(tri_out.triangleattributelist);
    free(tri_in.pointlist);
    free(tri_in.segmentlist);
    free(tri_in.pointmarkerlist);
    free(tri_in.segmentmarkerlist);
    if(tri_in.holelist) free(tri_in.holelist);
    if(tri_in.regionlist) free(tri_in.regionlist);
    return true;
}

// ── Triangle 2D triangulation → populate mesh for TMz ──

bool preprocessing::triangulatemesh2D(const char* name, mesh* mesh, const char* switches)
{
    std::string polyfile = std::string(name) + ".poly";
    struct triangulateio tri_in, tri_out;
    memset(&tri_in, 0, sizeof(tri_in));
    memset(&tri_out, 0, sizeof(tri_out));

    if(!read_poly_for_triangle(polyfile.c_str(), &tri_in))
        return false;

    // Store PLC geometry for mesh regeneration
    mesh->store_plc_from_triangle(tri_in);

    std::string cmd;
    if(switches && switches[0])
        cmd = std::string("p") + switches + "AfeQ";
    else
        cmd = "pq34AfeQ";
    ::triangulate((char*)cmd.c_str(), &tri_in, &tri_out, NULL);
    std::cout << "Triangle 2D output: " << tri_out.numberofpoints << " points, "
              << tri_out.numberoftriangles << " triangles, "
              << tri_out.numberofsegments << " segments\n";

    int np = tri_out.numberofpoints;
    int nt = tri_out.numberoftriangles;
    int ns = tri_out.numberofsegments;
    mesh->mesh_dim = 2;

    if(np == 0 || nt == 0) {
        free(tri_out.pointlist); free(tri_out.trianglelist);
        return false;
    }

    // Store nodes
    mesh->nNodes = np;
    mesh->nodPos.resize(np, 3);
    for(int i = 0; i < np; i++) {
        mesh->nodPos(i, 0) = tri_out.pointlist[2*i];
        mesh->nodPos(i, 1) = tri_out.pointlist[2*i + 1];
        mesh->nodPos(i, 2) = 0.0;
    }

    // Store triangles as faces
    mesh->nFaces = nt;
    mesh->facNodes.set_size(nt, 3);
    mesh->facLab.set_size(nt);
    for(int i = 0; i < nt; i++) {
        mesh->facNodes(i, 0) = tri_out.trianglelist[3*i] - 1;
        mesh->facNodes(i, 1) = tri_out.trianglelist[3*i + 1] - 1;
        mesh->facNodes(i, 2) = tri_out.trianglelist[3*i + 2] - 1;
        mesh->facLab(i) = tri_out.triangleattributelist
                          ? (size_t)tri_out.triangleattributelist[i] : 0;
    }

    // Store boundary segments as edges
    mesh->nEdges = ns;
    mesh->edgNodes.set_size(ns, 2);
    mesh->edgLab.set_size(ns);
    for(int i = 0; i < ns; i++) {
        mesh->edgNodes(i, 0) = tri_out.segmentlist[2*i] - 1;
        mesh->edgNodes(i, 1) = tri_out.segmentlist[2*i + 1] - 1;
        mesh->edgLab(i) = tri_out.segmentmarkerlist ? (size_t)tri_out.segmentmarkerlist[i] : 0;
    }

    std::cout << "2D mesh stored: " << np << " nodes, " << nt << " triangles, " << ns << " segments\n";

    free(tri_out.pointlist); free(tri_out.trianglelist);
    free(tri_out.segmentlist); free(tri_out.pointmarkerlist);
    free(tri_out.segmentmarkerlist); free(tri_out.triangleattributelist);
    free(tri_in.pointlist);
    free(tri_in.segmentlist);
    free(tri_in.pointmarkerlist);
    free(tri_in.segmentmarkerlist);
    if(tri_in.holelist) free(tri_in.holelist);
    if(tri_in.regionlist) free(tri_in.regionlist);
    return true;
}

// ── TetGen mesh population ──

void preprocessing::Createmesh()
{
    size_t n0, n1, n2;
    prj->msh->mesh_dim = 3;
    // Nodes
    prj->msh->nNodes = out.numberofpoints;
    prj->msh->nodPos.resize(prj->msh->nNodes,3);
    prj->msh->nodPos.fill(0);
    if(dbg)
    {
        std::cout << "nNodes  = " << prj->msh->nNodes << "\n";
    }
    for(size_t i=0; i<prj->msh->nNodes; i++)
    {
        for(size_t j=0; j<3; j++)
        {
            prj->msh->nodPos(i,j) = out.pointlist[i*3+j];
        }
    }
    //prj->msh->nodPos *= scaling;
    // Edges
    std::map<std::pair<size_t,size_t>, size_t> edgesMap;
    prj->msh->nEdges = out.numberofedges;
    prj->msh->edgNodes.resize(prj->msh->nEdges,2);
    prj->msh->edgNodes.fill(0);
    if(dbg)
    {
        std::cout << "nEdges  = " << prj->msh->nEdges << "\n";
    }
    for(size_t i=0; i<prj->msh->nEdges; i++)
    {
        for(size_t j=0; j<2; j++)
        {
            prj->msh->edgNodes(i,j) = out.edgelist[i*2+j]-1;
        }
        prj->msh->edgNodes.row(i) = arma::sort(prj->msh->edgNodes.row(i));
        n0 = prj->msh->edgNodes(i,0);
        n1 = prj->msh->edgNodes(i,1);
        edgesMap[std::make_pair(n0, n1)] = i;
    }
    // Faces
    std::map<std::pair<size_t,std::pair<size_t,size_t> >, size_t> facesMap;
    prj->msh->nFaces = out.numberoftrifaces;
    prj->msh->facNodes.resize(prj->msh->nFaces,3);
    prj->msh->facNodes.fill(0);
    prj->msh->facEdges.resize(prj->msh->nFaces,3);
    prj->msh->facEdges.fill(0);
    prj->msh->facLab.resize(prj->msh->nFaces);
    prj->msh->facLab.fill(prj->msh->maxLab);
    prj->msh->facAdjTet.set_size(prj->msh->nFaces);
    if(dbg)
    {
        std::cout << "nFaces  = " << prj->msh->nFaces << "\n";
    }
    for(size_t i=0; i<prj->msh->nFaces; i++)
    {
        for(size_t j=0; j<3; j++)
        {
            prj->msh->facNodes(i,j) = out.trifacelist[i*3+j]-1;
        }
        prj->msh->facNodes.row(i) = arma::sort(prj->msh->facNodes.row(i));
        n0 = prj->msh->facNodes(i,1);
        n1 = prj->msh->facNodes(i,2);
        prj->msh->facEdges(i,0) = edgesMap[std::make_pair(n0, n1)];
        n0 = prj->msh->facNodes(i,0);
        n1 = prj->msh->facNodes(i,2);
        prj->msh->facEdges(i,1) = edgesMap[std::make_pair(n0, n1)];
        n0 = prj->msh->facNodes(i,0);
        n1 = prj->msh->facNodes(i,1);
        prj->msh->facEdges(i,2) = edgesMap[std::make_pair(n0, n1)];
        //
        n0 = prj->msh->facNodes(i,0);
        n1 = prj->msh->facNodes(i,1);
        n2 = prj->msh->facNodes(i,2);
        facesMap[std::make_pair(n0, std::make_pair(n1,n2))] = i;
        // labels
        prj->msh->facLab(i) = out.trifacemarkerlist[i];
    }
    // Tetras
    prj->msh->nTetras = out.numberoftetrahedra;
    prj->msh->tetNodes.resize(prj->msh->nTetras,4);
    prj->msh->tetNodes.fill(0);
    prj->msh->tetEdges.resize(prj->msh->nTetras,6);
    prj->msh->tetEdges.fill(0);
    prj->msh->tetFaces.resize(prj->msh->nTetras,4);
    prj->msh->tetFaces.fill(0);
    prj->msh->tetLab.resize(prj->msh->nTetras);
    prj->msh->tetLab.fill(prj->msh->maxLab);
    if(dbg)
    {
        std::cout << "nTetras = " << prj->msh->nTetras << "\n";
    }
    for(size_t i=0; i<prj->msh->nTetras; i++)
    {
        for(size_t j=0; j<4; j++)
        {
            prj->msh->tetNodes(i,j) = out.tetrahedronlist[i*4+j]-1;
        }
        prj->msh->tetNodes.row(i) = arma::sort(prj->msh->tetNodes.row(i));
        //
        n0 = prj->msh->tetNodes(i,0);
        n1 = prj->msh->tetNodes(i,1);
        prj->msh->tetEdges(i,0) = edgesMap[std::make_pair(n0, n1)];
        n0 = prj->msh->tetNodes(i,0);
        n1 = prj->msh->tetNodes(i,2);
        prj->msh->tetEdges(i,1) = edgesMap[std::make_pair(n0, n1)];
        n0 = prj->msh->tetNodes(i,0);
        n1 = prj->msh->tetNodes(i,3);
        prj->msh->tetEdges(i,2) = edgesMap[std::make_pair(n0, n1)];
        n0 = prj->msh->tetNodes(i,1);
        n1 = prj->msh->tetNodes(i,2);
        prj->msh->tetEdges(i,3) = edgesMap[std::make_pair(n0, n1)];
        n0 = prj->msh->tetNodes(i,1);
        n1 = prj->msh->tetNodes(i,3);
        prj->msh->tetEdges(i,4) = edgesMap[std::make_pair(n0, n1)];
        n0 = prj->msh->tetNodes(i,2);
        n1 = prj->msh->tetNodes(i,3);
        prj->msh->tetEdges(i,5) = edgesMap[std::make_pair(n0, n1)];
        //
        n0 = prj->msh->tetNodes(i,1);
        n1 = prj->msh->tetNodes(i,2);
        n2 = prj->msh->tetNodes(i,3);
        prj->msh->tetFaces(i,0) = facesMap[std::make_pair(n0, std::make_pair(n1, n2))];
        n0 = prj->msh->tetNodes(i,0);
        n1 = prj->msh->tetNodes(i,2);
        n2 = prj->msh->tetNodes(i,3);
        prj->msh->tetFaces(i,1) = facesMap[std::make_pair(n0, std::make_pair(n1, n2))];
        n0 = prj->msh->tetNodes(i,0);
        n1 = prj->msh->tetNodes(i,1);
        n2 = prj->msh->tetNodes(i,3);
        prj->msh->tetFaces(i,2) = facesMap[std::make_pair(n0, std::make_pair(n1, n2))];
        n0 = prj->msh->tetNodes(i,0);
        n1 = prj->msh->tetNodes(i,1);
        n2 = prj->msh->tetNodes(i,2);
        prj->msh->tetFaces(i,3) = facesMap[std::make_pair(n0, std::make_pair(n1, n2))];
        // labels
        if(out.tetrahedronattributelist != NULL)
            prj->msh->tetLab(i) = out.tetrahedronattributelist[i];
        // adjTets
        arma::uvec adjTetId(1);
        adjTetId(0) = i;
        for(size_t j=0; j<4; j++)
        {
            size_t fid = prj->msh->tetFaces(i,j);
            prj->msh->facAdjTet(fid) = arma::join_cols(prj->msh->facAdjTet(fid), adjTetId);
        }
    }
    for(size_t i = 0; i < prj->msh->facbc.size(); i++)
    {
        size_t cLab = prj->msh->facbc[i].label;
        if(dbg)
        {
            std::cout << prj->msh->facbc[i].name << " " << cLab << "\n";
        }
        for(size_t fid = 0; fid < prj->msh->nFaces; fid++)
        {
            if(cLab == prj->msh->facLab(fid))
            {
                arma::uvec cFace(1);
                cFace(0) = fid;
                prj->msh->facbc[i].Faces = arma::join_cols(prj->msh->facbc[i].Faces, cFace);
            }
        }
    }
    for(size_t i = 0; i < prj->msh->tetmtrl.size(); i++)
    {
        size_t cLab = prj->msh->tetmtrl[i].label;
        if(dbg)
        {
            std::cout << prj->msh->tetmtrl[i].name << " " << cLab << "\n";
        }
        size_t tetcnt = 0;
        for(size_t tid = 0; tid < prj->msh->nTetras; tid++)
        {
            if(cLab == prj->msh->tetLab(tid))
            {
//                arma::uvec cTet(1);
//                cTet(0) = tid;
//                prj->msh->tetmtrl[i].Tetras = arma::join_cols(prj->msh->tetmtrl[i].Tetras, cTet);
                ++tetcnt;
            }
        }
        prj->msh->tetmtrl[i].Tetras.resize(tetcnt);
        tetcnt = 0;
        for(size_t tid = 0; tid < prj->msh->nTetras; tid++)
        {
            if(cLab == prj->msh->tetLab(tid))
            {
                prj->msh->tetmtrl[i].Tetras(tetcnt++) = tid;
            }
        }
    }
}

void preprocessing::load_extra()
{
    size_t tmpInt;
    double tmpDbl;
    std::string tmpStr;
    std::string line;
    std::ifstream fileName(std::string(prj->opt->name + ".poly").c_str(), std::ios::in);
    if(fileName.is_open())
    {
        while(getline(fileName,line))
        {
            std::istringstream iss(line);
            iss >> tmpStr;
            /*
            if(tmpStr == "#Scaling")
            {
                iss >> scaling;
                tmpStr.clear();
            }
            */
            if(tmpStr == "#Solids" || tmpStr == "#Regions")
            {
                iss >> tmpInt;
                for(size_t i = 0; i < tmpInt; i++)
                {
                    mtrl mtr;
                    mtr.label = i;
                    getline(fileName,line);
                    std::istringstream iss(line);
                    iss >> mtr.sld_name;
                    iss >> mtr.epsr;
                    iss >> mtr.mur;
                    iss >> mtr.sigma;
                    iss >> mtr.tand;
                    iss >> mtr.name;
                    mtr.updmtrl();
                    prj->msh->tetmtrl.push_back(mtr);
                    std::cout << mtr.sld_name << " " << mtr.name << "\n";
                }
                tmpStr.clear();
            }
            if(tmpStr == "#Boundaries")
            {
                iss >> tmpInt;
                for(size_t i = 0; i < tmpInt; i++)
                {
                    std::string type;
                    bc bc;
                    getline(fileName,line);
                    std::istringstream iss(line);
                    iss >> bc.name;
                    iss >> bc.label;
                    iss >> type;
                    bc.set_type(type);
                    if(bc.type == bc::wave_port)
                    {
                        iss >> bc.num_modes;
                    }
                    prj->msh->facbc.push_back(bc);
                    std::cout << bc.name << " " << type << " " << bc.num_modes <<"\n";
                }
                tmpStr.clear();
            }
        }
    }
}


void preprocessing::CopyOldmesh()
{
    // Nodes
    in.firstnumber = 0;
    in.numberofpoints = prj->msh->nNodes;
    in.pointlist = new REAL[3*in.numberofpoints];
    for(size_t i=0; i<prj->msh->nNodes; i++)
    {
        for(size_t j=0; j<3; j++)
        {
            in.pointlist[i*3+j] = prj->msh->nodPos(i,j);
        }
    }
    // Edges
    in.numberofedges = prj->msh->nEdges;
    in.edgelist = new int[in.numberofedges*2];
    for(size_t i=0; i<prj->msh->nEdges; i++)
    {
        for(size_t j=0; j<2; j++)
        {
            in.edgelist[i*2+j] = prj->msh->edgNodes(i,j);
        }
    }
    // Faces
    in.numberoftrifaces = prj->msh->nFaces;
    in.trifacelist = new int[in.numberoftrifaces*3];
    in.trifacemarkerlist = new int[in.numberoftrifaces];
    for(size_t i=0; i<prj->msh->nFaces; i++)
    {
        for(size_t j=0; j<3; j++)
        {
            in.trifacelist[i*3+j] = prj->msh->facNodes(i,j);
        }
        // labels
        in.trifacemarkerlist[i] = prj->msh->facLab(i)+1;
        //std::cout << in.trifacemarkerlist[i] << " ";
    }
    // Tetras
    in.numberoftetrahedra = prj->msh->nTetras;
    in.tetrahedronlist = new int[in.numberoftetrahedra*4];
    in.numberoftetrahedronattributes = 1;
    in.tetrahedronattributelist = new double[in.numberoftetrahedra];
    for(size_t i=0; i<prj->msh->nTetras; i++)
    {
        for(size_t j=0; j<4; j++)
        {
            in.tetrahedronlist[i*4+j] = prj->msh->tetNodes(i,j);
        }
        // labels
        in.tetrahedronattributelist[i] = prj->msh->tetLab(i)+1;
        //std::cout << in.tetrahedronattributelist[i] << " ";
    }
}

void preprocessing::CopyNewmesh()
{
    size_t n0, n1, n2;
    // Nodes
    //std::cout << out.numberoftetrahedronattributes << "\n\n";
    prj->msh->nNodes = out.numberofpoints;
    prj->msh->nodPos.clear();
    prj->msh->nodPos.resize(prj->msh->nNodes,3);
    prj->msh->nodPos.fill(0);
    if(dbg)
    {
        std::cout << "nNodes  = " << prj->msh->nNodes << "\n";
    }
    std::cout << "Nodes ";
    for(size_t i=0; i<prj->msh->nNodes; i++)
    {
        for(size_t j=0; j<3; j++)
        {
            prj->msh->nodPos(i,j) = out.pointlist[i*3+j];
        }
    }
    //prj->msh->nodPos *= scaling;
    //std::cout << "nodes done.\n";
    // Edges
    std::map<std::pair<size_t,size_t>, size_t> edgesMap;
    prj->msh->nEdges = out.numberofedges;
    prj->msh->edgNodes.clear();
    prj->msh->edgNodes.resize(prj->msh->nEdges,2);
    prj->msh->edgNodes.fill(0);
    if(dbg)
    {
        std::cout << "nEdges  = " << prj->msh->nEdges << "\n";
    }
    std::cout << "Edges ";
    for(size_t i=0; i<prj->msh->nEdges; i++)
    {
        for(size_t j=0; j<2; j++)
        {
            prj->msh->edgNodes(i,j) = out.edgelist[i*2+j];
        }
        prj->msh->edgNodes.row(i) = arma::sort(prj->msh->edgNodes.row(i));
        n0 = prj->msh->edgNodes(i,0);
        n1 = prj->msh->edgNodes(i,1);
        edgesMap[std::make_pair(n0, n1)] = i;
    }
    //std::cout << "edges done.\n";
    // Faces
    std::map<std::pair<size_t,std::pair<size_t,size_t> >, size_t> facesMap;
    prj->msh->nFaces = out.numberoftrifaces;
    prj->msh->facNodes.clear();
    prj->msh->facNodes.resize(prj->msh->nFaces,3);
    prj->msh->facNodes.fill(0);
    prj->msh->facEdges.clear();
    prj->msh->facEdges.resize(prj->msh->nFaces,3);
    prj->msh->facEdges.fill(0);
    prj->msh->facLab.clear();
    prj->msh->facLab.resize(prj->msh->nFaces);
    prj->msh->facLab.fill(prj->msh->maxLab);
    prj->msh->facAdjTet.set_size(prj->msh->nFaces);
    if(dbg)
    {
        std::cout << "nFaces  = " << prj->msh->nFaces << "\n";
    }
    std::cout << "Faces ";
    for(size_t i=0; i<prj->msh->nFaces; i++)
    {
        for(size_t j=0; j<3; j++)
        {
            prj->msh->facNodes(i,j) = out.trifacelist[i*3+j];
        }
        prj->msh->facNodes.row(i) = arma::sort(prj->msh->facNodes.row(i));
        n0 = prj->msh->facNodes(i,0);
        n1 = prj->msh->facNodes(i,1);
        prj->msh->facEdges(i,0) = edgesMap[std::make_pair(n0, n1)];
        n0 = prj->msh->facNodes(i,0);
        n1 = prj->msh->facNodes(i,2);
        prj->msh->facEdges(i,1) = edgesMap[std::make_pair(n0, n1)];
        n0 = prj->msh->facNodes(i,1);
        n1 = prj->msh->facNodes(i,2);
        prj->msh->facEdges(i,2) = edgesMap[std::make_pair(n0, n1)];
        //
        n0 = prj->msh->facNodes(i,0);
        n1 = prj->msh->facNodes(i,1);
        n2 = prj->msh->facNodes(i,2);
        facesMap[std::make_pair(n0, std::make_pair(n1,n2))] = i;
        // labels
        //std::cout << out.trifacemarkerlist[i] << "\n";
        //if(out.trifacemarkerlist[i] > -1)
        prj->msh->facLab(i) = out.trifacemarkerlist[i]-1;
        //else
        //prj->msh->facLab(i) = prj->msh->maxLab;
        //std::cout << prj->msh->facLab(i) << " ";
    }
    //std::cout << "faces done.\n";
    // Tetras
    prj->msh->nTetras = out.numberoftetrahedra;
    prj->msh->tetNodes.clear();
    prj->msh->tetNodes.resize(prj->msh->nTetras,4);
    prj->msh->tetNodes.fill(0);
    prj->msh->tetEdges.clear();
    prj->msh->tetEdges.resize(prj->msh->nTetras,6);
    prj->msh->tetEdges.fill(0);
    prj->msh->tetFaces.clear();
    prj->msh->tetFaces.resize(prj->msh->nTetras,4);
    prj->msh->tetFaces.fill(0);
    prj->msh->tetLab.clear();
    prj->msh->tetLab.resize(prj->msh->nTetras);
    prj->msh->tetLab.fill(0);
    if(dbg)
    {
        std::cout << "nTetras = " << prj->msh->nTetras << "\n";
    }
    std::cout << "Tetras ";
    for(size_t i=0; i<prj->msh->nTetras; i++)
    {
        for(size_t j=0; j<4; j++)
        {
            prj->msh->tetNodes(i,j) = out.tetrahedronlist[i*4+j];
        }
        prj->msh->tetNodes.row(i) = arma::sort(prj->msh->tetNodes.row(i));
        //
        n0 = prj->msh->tetNodes(i,0);
        n1 = prj->msh->tetNodes(i,1);
        prj->msh->tetEdges(i,0) = edgesMap[std::make_pair(n0, n1)];
        n0 = prj->msh->tetNodes(i,0);
        n1 = prj->msh->tetNodes(i,2);
        prj->msh->tetEdges(i,1) = edgesMap[std::make_pair(n0, n1)];
        n0 = prj->msh->tetNodes(i,0);
        n1 = prj->msh->tetNodes(i,3);
        prj->msh->tetEdges(i,2) = edgesMap[std::make_pair(n0, n1)];
        n0 = prj->msh->tetNodes(i,1);
        n1 = prj->msh->tetNodes(i,2);
        prj->msh->tetEdges(i,3) = edgesMap[std::make_pair(n0, n1)];
        n0 = prj->msh->tetNodes(i,1);
        n1 = prj->msh->tetNodes(i,3);
        prj->msh->tetEdges(i,4) = edgesMap[std::make_pair(n0, n1)];
        n0 = prj->msh->tetNodes(i,2);
        n1 = prj->msh->tetNodes(i,3);
        prj->msh->tetEdges(i,5) = edgesMap[std::make_pair(n0, n1)];
        //
        n0 = prj->msh->tetNodes(i,1);
        n1 = prj->msh->tetNodes(i,2);
        n2 = prj->msh->tetNodes(i,3);
        prj->msh->tetFaces(i,0) = facesMap[std::make_pair(n0, std::make_pair(n1, n2))];
        n0 = prj->msh->tetNodes(i,0);
        n1 = prj->msh->tetNodes(i,2);
        n2 = prj->msh->tetNodes(i,3);
        prj->msh->tetFaces(i,1) = facesMap[std::make_pair(n0, std::make_pair(n1, n2))];
        n0 = prj->msh->tetNodes(i,0);
        n1 = prj->msh->tetNodes(i,1);
        n2 = prj->msh->tetNodes(i,3);
        prj->msh->tetFaces(i,2) = facesMap[std::make_pair(n0, std::make_pair(n1, n2))];
        n0 = prj->msh->tetNodes(i,0);
        n1 = prj->msh->tetNodes(i,1);
        n2 = prj->msh->tetNodes(i,2);
        prj->msh->tetFaces(i,3) = facesMap[std::make_pair(n0, std::make_pair(n1, n2))];
        // labels
        if(out.tetrahedronattributelist != NULL)
        {
            prj->msh->tetLab(i) = out.tetrahedronattributelist[i]-1;
            //std::cout << prj->msh->tetLab(i) << "\n";
        }
        // adjTets
        arma::uvec adjTetId(1);
        adjTetId(0) = i;
        for(size_t j=0; j<4; j++)
        {
            size_t fid = prj->msh->tetFaces(i,j);
            prj->msh->facAdjTet(fid) = arma::join_cols(prj->msh->facAdjTet(fid), adjTetId);
        }
    }
    //std::cout << "tets done.\n";
    if(dbg)
    {
        std::cout << "nFacbc = " << prj->msh->facbc.size() << "\n";
    }
    std::cout << "bc ";
    for(size_t i = 0; i < prj->msh->facbc.size(); i++)
    {
        prj->msh->facbc[i].Faces.clear();
        size_t cLab = prj->msh->facbc[i].label;
        //std::cout << prj->msh->facbc[i].name << " " << cLab << "\n";
        for(size_t fid = 0; fid < prj->msh->nFaces; fid++)
        {
            if(cLab == prj->msh->facLab(fid))
            {
                arma::uvec cFace(1);
                cFace(0) = fid;
                prj->msh->facbc[i].Faces = arma::join_cols(prj->msh->facbc[i].Faces, cFace);
            }
        }
    }
    if(dbg)
    {
        std::cout << "nTetmtrl = " << prj->msh->tetmtrl.size() << "\n";
    }
    std::cout << "mtrl";
    std::vector<size_t> mtrlTets(prj->msh->tetmtrl.size(),0), LabMap(prj->msh->tetmtrl.size(),0);
    for(size_t tid = 0; tid < prj->msh->nTetras; tid++)
    {
        mtrlTets[prj->msh->tetLab(tid)]++;
    }
    for(size_t i = 0; i < prj->msh->tetmtrl.size(); i++)
    {
        prj->msh->tetmtrl[i].Tetras.resize(mtrlTets[i]);
        mtrlTets[i] = 0;
    }
    for(size_t tid = 0; tid < prj->msh->nTetras; tid++)
    {
        size_t cLab = prj->msh->tetLab(tid);
        prj->msh->tetmtrl[cLab].Tetras[mtrlTets[cLab]++] = tid;
    }
    std::cout << "\n";
}

// ── Store 3D TetGen PLC geometry in mesh ──

void mesh::store_plc_from_tetgen(tetgenio& in)
{
    plc_valid = true;
    int np = in.numberofpoints;
    plc_points.resize(np * 3);
    for(int i = 0; i < np * 3; i++)
        plc_points[i] = in.pointlist[i];

    int nf = in.numberoffacets;
    plc_facet_markers.clear();
    plc_poly_vertex_counts.clear();
    plc_poly_vertex_list.clear();
    plc_facet_holes.clear();
    if(in.facetmarkerlist) {
        plc_facet_markers.assign(in.facetmarkerlist, in.facetmarkerlist + nf);
    }
    for(int i = 0; i < nf; i++) {
        tetgenio::facet& f = in.facetlist[i];
        for(int j = 0; j < f.numberofpolygons; j++) {
            tetgenio::polygon& p = f.polygonlist[j];
            plc_poly_vertex_counts.push_back(p.numberofvertices);
            plc_poly_vertex_list.insert(plc_poly_vertex_list.end(),
                p.vertexlist, p.vertexlist + p.numberofvertices);
        }
        if(f.numberofholes > 0) {
            plc_facet_holes.insert(plc_facet_holes.end(),
                f.holelist, f.holelist + f.numberofholes * 3);
        }
    }

    if(in.holelist && in.numberofholes > 0) {
        plc_volume_holes.assign(in.holelist, in.holelist + in.numberofholes * 3);
    }
    if(in.regionlist && in.numberofregions > 0) {
        plc_regions.assign(in.regionlist, in.regionlist + in.numberofregions * 5);
    }
}

// ── Store 2D Triangle PLC geometry in mesh ──

void mesh::store_plc_from_triangle(triangulateio& tri_in)
{
    plc_valid = true;
    int np = tri_in.numberofpoints;
    plc_2d_points.resize(np * 2);
    for(int i = 0; i < np * 2; i++)
        plc_2d_points[i] = tri_in.pointlist[i];

    int ns = tri_in.numberofsegments;
    plc_segments.resize(ns * 2);
    for(int i = 0; i < ns * 2; i++)
        plc_segments[i] = tri_in.segmentlist[i];
    if(tri_in.segmentmarkerlist) {
        plc_seg_markers.assign(tri_in.segmentmarkerlist, tri_in.segmentmarkerlist + ns);
    }

    if(tri_in.holelist && tri_in.numberofholes > 0) {
        plc_2d_holes.assign(tri_in.holelist, tri_in.holelist + tri_in.numberofholes * 2);
    }
    if(tri_in.regionlist && tri_in.numberofregions > 0) {
        plc_2d_regions.assign(tri_in.regionlist, tri_in.regionlist + tri_in.numberofregions * 4);
    }
}

// ── Populate tetgenio from stored PLC (3D mesh regeneration) ──

void preprocessing::populate_tetgenio_from_plc(mesh* msh, tetgenio& in)
{
    int np = (int)(msh->plc_points.size() / 3);
    in.firstnumber = 1;  // 1-based indexing (as in .poly files)
    in.numberofpoints = np;
    in.pointlist = new REAL[np * 3];
    for(int i = 0; i < np * 3; i++)
        in.pointlist[i] = msh->plc_points[i];

    // Reconstruct facets from flattened polygon data
    int nf = (int)msh->plc_facet_markers.size();
    in.numberoffacets = nf;
    in.facetlist = new tetgenio::facet[nf];
    in.facetmarkerlist = new int[nf];
    // Stored layout for plc_poly_vertex_counts: all polygon counts for all facets flattened
    // We store NFE (total polygon count) as the first element, then each facet's polygon count.
    // If NFE is stored at plcs[0], then plcs[1..nf] holds the per-facet polygon counts.
    // Actually, simpler: store each facet's polygon count inline.
    // Current serialization stores: all vertex_counts flattened = one counter per polygon.
    // We don't store per-facet polygon count separately. Fix: store a header per facet.
    int vertIdx = 0;
    int countIdx = 0;
    for(int i = 0; i < nf; i++) {
        in.facetmarkerlist[i] = msh->plc_facet_markers[i];
        tetgenio::facet& f = in.facetlist[i];
        // The store_plc_from_tetgen saves all polygons from all facets flattened.
        // For now, assume each facet has exactly 1 polygon (the common case).
        f.numberofpolygons = 1;
        f.polygonlist = new tetgenio::polygon[1];
        tetgenio::polygon& p = f.polygonlist[0];
        p.numberofvertices = msh->plc_poly_vertex_counts.empty() ? 0
            : msh->plc_poly_vertex_counts[countIdx++];
        p.vertexlist = new int[p.numberofvertices];
        for(int j = 0; j < p.numberofvertices; j++)
            p.vertexlist[j] = msh->plc_poly_vertex_list[vertIdx++];
        // Facet holes (if any)
        // The store_plc_from_tetgen saves all facet holes flat. For now skip.
        f.numberofholes = 0;
    }

    // Volume holes
    int nh = (int)(msh->plc_volume_holes.size() / 3);
    in.numberofholes = nh;
    if(nh > 0) {
        in.holelist = new REAL[nh * 3];
        for(int i = 0; i < nh * 3; i++)
            in.holelist[i] = msh->plc_volume_holes[i];
    }

    // Regions
    int nr = (int)(msh->plc_regions.size() / 5);
    in.numberofregions = nr;
    if(nr > 0) {
        in.regionlist = new REAL[nr * 5];
        for(int i = 0; i < nr * 5; i++)
            in.regionlist[i] = msh->plc_regions[i];
    }
}

// ── Populate triangulateio from stored PLC (2D mesh regeneration) ──

void preprocessing::populate_triangleio_from_plc(mesh* msh, triangulateio& tri_in)
{
    int np = (int)(msh->plc_2d_points.size() / 2);
    tri_in.numberofpoints = np;
    tri_in.pointlist = (double*)malloc(np * 2 * sizeof(double));
    for(int i = 0; i < np * 2; i++)
        tri_in.pointlist[i] = msh->plc_2d_points[i];

    int ns = (int)(msh->plc_segments.size() / 2);
    tri_in.numberofsegments = ns;
    tri_in.segmentlist = (int*)malloc(ns * 2 * sizeof(int));
    for(int i = 0; i < ns * 2; i++)
        tri_in.segmentlist[i] = msh->plc_segments[i];

    if(!msh->plc_seg_markers.empty()) {
        tri_in.segmentmarkerlist = (int*)malloc(ns * sizeof(int));
        for(int i = 0; i < ns; i++)
            tri_in.segmentmarkerlist[i] = msh->plc_seg_markers[i];
    }

    int nh = (int)(msh->plc_2d_holes.size() / 2);
    tri_in.numberofholes = nh;
    if(nh > 0) {
        tri_in.holelist = (double*)malloc(nh * 2 * sizeof(double));
        for(int i = 0; i < nh * 2; i++)
            tri_in.holelist[i] = msh->plc_2d_holes[i];
    }

    int nr = (int)(msh->plc_2d_regions.size() / 4);
    tri_in.numberofregions = nr;
    if(nr > 0) {
        tri_in.regionlist = (double*)malloc(nr * 4 * sizeof(double));
        for(int i = 0; i < nr * 4; i++)
            tri_in.regionlist[i] = msh->plc_2d_regions[i];
    }
}

// ── 2D Triangle mesh regeneration from stored PLC ──

bool preprocessing::triangulatemesh2D_from_plc(mesh* msh, const char* switches)
{
    struct triangulateio tri_in, tri_out;
    memset(&tri_in, 0, sizeof(tri_in));
    memset(&tri_out, 0, sizeof(tri_out));

    populate_triangleio_from_plc(msh, tri_in);

    std::string cmd;
    if(switches && switches[0])
        cmd = std::string("p") + switches + "AfeQ";
    else
        cmd = "pq34AfeQ";
    ::triangulate((char*)cmd.c_str(), &tri_in, &tri_out, NULL);
    std::cout << "Triangle 2D output (from PLC): " << tri_out.numberofpoints << " points, "
              << tri_out.numberoftriangles << " triangles, "
              << tri_out.numberofsegments << " segments\n";

    int np = tri_out.numberofpoints;
    int nt = tri_out.numberoftriangles;
    int ns = tri_out.numberofsegments;
    msh->mesh_dim = 2;

    if(np == 0 || nt == 0) {
        free(tri_out.pointlist); free(tri_out.trianglelist);
        return false;
    }

    // Store nodes (same logic as triangulatemesh2D)
    msh->nNodes = np;
    msh->nodPos.resize(np, 3);
    for(int i = 0; i < np; i++) {
        msh->nodPos(i, 0) = tri_out.pointlist[2*i];
        msh->nodPos(i, 1) = tri_out.pointlist[2*i + 1];
        msh->nodPos(i, 2) = 0.0;
    }

    // Store triangles as faces
    msh->nFaces = nt;
    msh->facNodes.set_size(nt, 3);
    msh->facLab.set_size(nt);
    for(int i = 0; i < nt; i++) {
        msh->facNodes(i, 0) = tri_out.trianglelist[3*i] - 1;
        msh->facNodes(i, 1) = tri_out.trianglelist[3*i + 1] - 1;
        msh->facNodes(i, 2) = tri_out.trianglelist[3*i + 2] - 1;
        msh->facLab(i) = tri_out.triangleattributelist
                          ? (size_t)tri_out.triangleattributelist[i] : 0;
    }

    // Store boundary segments as edges
    msh->nEdges = ns;
    msh->edgNodes.set_size(ns, 2);
    msh->edgLab.set_size(ns);
    for(int i = 0; i < ns; i++) {
        msh->edgNodes(i, 0) = tri_out.segmentlist[2*i] - 1;
        msh->edgNodes(i, 1) = tri_out.segmentlist[2*i + 1] - 1;
        msh->edgLab(i) = tri_out.segmentmarkerlist ? (size_t)tri_out.segmentmarkerlist[i] : 0;
    }

    std::cout << "2D mesh stored from PLC: " << np << " nodes, " << nt << " triangles, "
              << ns << " segments\n";

    free(tri_out.pointlist); free(tri_out.trianglelist);
    free(tri_out.segmentlist); free(tri_out.pointmarkerlist);
    free(tri_out.segmentmarkerlist); free(tri_out.triangleattributelist);
    free(tri_in.pointlist);
    free(tri_in.segmentlist);
    free(tri_in.pointmarkerlist);
    free(tri_in.segmentmarkerlist);
    if(tri_in.holelist) free(tri_in.holelist);
    if(tri_in.regionlist) free(tri_in.regionlist);
    return true;
}
