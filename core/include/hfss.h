#ifndef hfss_H
#define hfss_H

#include <map>
#include "project.h"

class hfss_part
{
public:
    hfss_part() : name(""), material(""), solve_inside(false), id(0) {}
    virtual ~hfss_part() {}
    std::string name;
    std::string material;
    bool solve_inside;
    size_t id;
};

class hfss_bnd
{
public:
    hfss_bnd() : name(""), type(""), impedance(50.0) {}
    virtual ~hfss_bnd() {}
    std::string name;
    std::string type;
    std::vector<size_t> faces;
    std::vector<size_t> solids;
    std::vector<size_t> face_ids;
    int num_modes;
    double impedance;
};

class hfss_mtrl
{
public:
    hfss_mtrl() : permittivity(1), permeability(1), conductivity(0), dielectric_loss_tangent(0) {}
    virtual ~hfss_mtrl() {}
    double permittivity;
    double permeability;
    double conductivity;
    double dielectric_loss_tangent;
    std::string name;
};

class hfss
{
public:
    hfss(project*);
    virtual ~hfss();
    void ReadMainhfss();
    void read_points();
    void read_faces();
    void read_hydras();
    void Finalizemesh();
private:
    mesh* msh;
    project* prj;
    std::string name;
    std::map<std::string, hfss_mtrl> mtrls;
    std::vector<size_t> mtrlTag;
    std::vector<size_t> hfssid;
    std::vector<bool> tetFlag;
    std::vector<bool> facFlag;
    std::vector<bool> nodFlag;
    std::vector< std::vector<size_t> > fachfsstag;
    std::vector<hfss_bnd> bnds;
    std::vector<hfss_part> parts;
    std::map<size_t, std::vector<size_t> > bndMap;
    std::vector<std::vector<size_t> > adjTetra;
    bool debug;
};

#endif // hfss_H
