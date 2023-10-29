#ifndef AssLin_H
#define AssLin_H

#include <fstream>

class EqSys;
class Mesh;
class Option;
class Project;
class Quad;

class AssLin
{
public:
    AssLin(std::ofstream&, EqSys*);
    virtual ~AssLin();
private:
    Option* opt;
    Mesh* msh;
    Project* prj;
    Quad* quad;
};

#endif // AssLin_H
