#ifndef TETGEN_H
#define TETGEN_H

#include "Project.h"
#include <tetgen.h>

class TetGen
{
public:
    TetGen(Project*);
    virtual ~TetGen();
    void CreateMesh();
    void CopyOldMesh();
    void CopyNewMesh();
    void LoadExtra();
private:
    Project* prj;
    bool dbg;
    tetgenio in, out, addin, bgmin;
    double scaling;
};

#endif // TETGEN_H
