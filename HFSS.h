#ifndef HFSS_H
#define HFSS_H

#include <map>
#include "Project.h"

class HFSSPart
{
public:
    HFSSPart() : name(""), material(""), solveInside(false), id(0) {}
    virtual ~HFSSPart() {}
    std::string name;
    std::string material;
    bool solveInside;
    size_t id;
};

class HFSSBnd
{
public:
    HFSSBnd() : name(""), type("") {}
    virtual ~HFSSBnd() {}
    std::string name;
    std::string type;
    std::vector<size_t> faces;
    std::vector<size_t> solids;
    std::vector<size_t> faceIds;
    int numModes;
};

class HFSSMtrl
{
public:
    HFSSMtrl() : permittivity(1), permeability(1), conductivity(0), dielectric_loss_tangent(0) {}
    virtual ~HFSSMtrl() {}
    double permittivity;
    double permeability;
    double conductivity;
    double dielectric_loss_tangent;
    std::string name;
};

class HFSS
{
public:
    HFSS(Project*);
    virtual ~HFSS();
    void ReadMainHFSS();
    void ReadPoints();
    void ReadFaces();
    void ReadHydras();
    void FinalizeMesh();
private:
    Mesh* msh;
    Project* prj;
    std::string name;
    std::map<std::string, HFSSMtrl> mtrls;
    std::vector<size_t> mtrlTag;
    std::vector<size_t> hfssid;
    std::vector<bool> tetFlag;
    std::vector<bool> facFlag;
    std::vector<bool> nodFlag;
    std::vector< std::vector<size_t> > facHFSStag;
    std::vector<HFSSBnd> bnds;
    std::vector<HFSSPart> parts;
    std::map<size_t, std::vector<size_t> > bndMap;
    std::vector<std::vector<size_t> > adjTetra;
    bool debug;
};

#endif // HFSS_H
