// Minimal definitions so the cFES numerical classes can be linked without the
// mesh importer, METIS, or file I/O.
//
// Shape.cpp / Quad.cpp / EleMat.cpp / DoF.cpp / Coupl.cpp / Mtrl.cpp / BC.cpp
// are compiled verbatim from the cFES source tree; this file supplies only the
// constructors and destructors of Project, Mesh and Option that those
// translation units reference.  The harness fills the public Mesh members
// directly, exactly as the real Project constructor would after importing a
// mesh.

#include <fstream>

#include "Project.h"
#include "Mesh.h"
#include "Option.h"

Option::Option() {}
Option::~Option() {}

Mesh::Mesh()
    : nNodes(0), nEdges(0), nFaces(0), nTetras(0), nDomains(0)
{
}

Mesh::~Mesh() {}

Project::Project(std::ofstream&, Option& pOpt)
    : msh(new Mesh()), opt(&pOpt), freq(0.0)
{
}

Project::~Project() {}
