#ifndef ASSELSTAT_H
#define ASSELSTAT_H

#include <fstream>

class EqSys;
class Mesh;
class Option;
class Project;
class Quad;

class AssElStat
{
public:
    AssElStat(std::ofstream&, EqSys*);
    virtual ~AssElStat();
private:
    Option* opt;
    Mesh* msh;
    Project* prj;
    Quad* quad;
};

#endif // ASSELSTAT_H
