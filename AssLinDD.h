#ifndef ASSLINDD_H
#define ASSLINDD_H

#include <fstream>

class EqSys;
class Mesh;
class Option;
class Project;
class Quad;

class AssLinDD
{
public:
    AssLinDD(std::ofstream&, EqSys*);
    virtual ~AssLinDD();
private:
    Option* opt;
    Mesh* msh;
    Project* prj;
    Quad* quad;
};

#endif // ASSLINDD_H
