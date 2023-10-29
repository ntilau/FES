#ifndef PROJECT_H
#define PROJECT_H

#include "Mesh.h"
#include "Option.h"

class Project
{
public:
    Project(std::ofstream&, Option&);
    void saveFE();
    void saveFEbin();
    void loadFE();
    virtual ~Project();
    Mesh* msh;
    Option* opt;
    double freq;
};

#endif // PROJECT_H
