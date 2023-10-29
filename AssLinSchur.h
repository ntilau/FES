#ifndef ASSLINSYMSCHUR_H
#define ASSLINSYMSCHUR_H

#include <fstream>

class EqSys;
class Mesh;
class Option;
class Project;
class Quad;

class AssLinSchur
{
public:
    AssLinSchur(std::ofstream&, EqSys*);
    virtual ~AssLinSchur();
private:
    Option* opt;
    Mesh* msh;
    Project* prj;
    Quad* quad;
};

#endif // ASSLINSYMSCHUR_H
