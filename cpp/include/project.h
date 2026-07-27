#ifndef PROJECT_H
#define PROJECT_H

#include "mesh.h"
#include "option.h"

class project
{
public:
    project(std::ofstream&, option&);
    virtual ~project();
    mesh* msh;
    option* opt;
    double freq;
};

#endif // PROJECT_H
