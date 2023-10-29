#ifndef TETGEN_WR_H
#define TETGEN_WR_H

#include <tetgen.h>
#include "Project.h"


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

#endif // TETGEN_WR_H
