#ifndef ASSNL_H
#define ASSNL_H

#include <fstream>

class EqSys;
class Mesh;
class Option;
class Project;
class Quad;

class AssNL
{
public:
    AssNL(std::ofstream&, EqSys*);
    virtual ~AssNL();
private:
    Option* opt;
    Mesh* msh;
    Project* prj;
    Quad* quad;
};

#endif // ASSNL_H
