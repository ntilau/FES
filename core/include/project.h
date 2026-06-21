#ifndef PROJECT_H
#define PROJECT_H

#include "mesh.h"
#include "option.h"

class project
{
public:
    project(std::ofstream&, option&);
    void save_fes();
    void load_fes();
    virtual ~project();
    mesh* msh;
    option* opt;
    double freq;
};

#endif // PROJECT_H
