#ifndef TETGENIO_H
#define TETGENIO_H

#include <tetgen.h>
#include "Project.h"


class TetGenIO
{
public:
    TetGenIO(Project*);
    virtual ~TetGenIO();
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

#endif // TETGENIO_H
