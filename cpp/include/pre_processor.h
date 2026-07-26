#ifndef PRE_PROCESSOR_H
#define PRE_PROCESSOR_H

#include "project.h"
#include <tetgen.h>

struct triangulateio;

class preprocessing
{
public:
    preprocessing(project* cProj);
    virtual ~preprocessing();

    bool read_poly_for_triangle(const char* polyfile, struct triangulateio* tri_in);
    bool triangulate(const char* name, tetgenio& out, double scaling);
    bool triangulatemesh2D(const char* name, mesh* mesh, const char* switches);
    void Createmesh();
    void load_extra();
    void CopyOldmesh();
    void CopyNewmesh();
    void populate_tetgenio_from_plc(mesh* msh, tetgenio& in);
    void populate_triangleio_from_plc(mesh* msh, struct triangulateio& tri_in);
    bool triangulatemesh2D_from_plc(mesh* msh, const char* switches);

private:
    project* prj;
    bool dbg;
    double scaling;
    tetgenio in, out, addin, bgmin;
};

#endif // PRE_PROCESSOR_H
