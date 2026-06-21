#ifndef LTE_H
#define LTE_H

#include <string>
#include <vector>
#include <map>
#include "project.h"

// ── LTE material definition (read from .mtrl files) ──
struct lte_mtrl {
    std::string name;
    double permittivity = 1.0;
    double permeability = 1.0;
    double conductivity = 0.0;
    double dielectric_loss_tangent = 0.0;
};

// ── LTE part (solid body) ──
struct lte_part {
    std::string name;
    std::string material;
    std::vector<size_t> tet_ids;
};

// ── LTE boundary condition ──
struct lte_bnd {
    std::string name;
    std::string type;
    std::vector<size_t> face_ids;
    int num_modes = 1;
    double impedance = 50.0;
};

// ── LTE fileset import wrapper ──

class lte {
public:
    lte(project* prj);
    virtual ~lte();

private:
    std::string name;
    mesh* msh;
    project* prj;
    bool debug;

    std::map<std::string, lte_mtrl> mtrls;
    std::vector<lte_part> parts;
    std::vector<lte_bnd> bnds;
    std::map<std::string, std::vector<size_t>> faceMap;

    // File resolution: try <basename>.ext first, fall back to 3d.ext
    std::string resolve_file(const std::string& ext) const;
    std::string lte_path(const std::string& file) const;

    void read_mesh();
    void read_bc();
    void read_mtr();
    void read_mtrl_files();

    void parse_position_section(const std::string& data);
    void parse_edge_section(const std::string& data);
    void parse_face_section(const std::string& data);
    void parse_element_section(const std::string& data);
    static void parse_solid_section(const std::string& data, std::vector<lte_part>& parts);
    static void parse_polyface_section(const std::string& data,
                                       std::map<std::string, std::vector<size_t>>& faceMap);

    void Finalizemesh();
    void store_plc();
};

#endif // LTE_H
