#ifndef OPTION_H
#define OPTION_H
#include <iostream>
#include <string>
#include <map>

class option
{
public:
    enum assemb_type { em_e_fd, em_e_fd_dd, em_e_fd_nl, em_e_qs, em_h_qs, em_ez_fd, em_e_tl_eig };
    enum solverType { direct, gmres };
    option();
    virtual ~option();
    void set(const int argc, char* argv[]);
    void apply_cli();  // apply cli_override entries to member fields
    void print_usage(std::ostream& ostr) const;
    void serialize(std::ostream& out) const;
    void deserialize(std::istream& in);

    solverType solver;
    assemb_type assembly;
    std::string name;
    bool dbg;
    bool dbl;
    size_t niter;
    double toll;
    size_t h_ord;
    size_t p_ord;
    double freq;
    double l_freq, h_freq;
    size_t n_freqs;
    size_t n_harm;
    double relax;
    double E[3], k[3];
    bool einc;        // true when +einc was specified
    bool field;
    bool rad;
    double n_theta, n_phi;
    bool poly;
    std::string poly_cmd;
    std::string href_cmd;
    bool dd;
    bool ddn;
    bool dds;
    bool n_jor_gs; // Jacobi or GaussSeidel precond
    size_t n_dd;
    bool nl;
    double power;// scalar coefficient in [W]
    std::map<std::string, double> Vbnd;

    // Thin CLI override storage: key → value strings parsed from +flag <val> pairs.
    // After loading .fes XML, these are applied by name and the file is re-saved.
    // Use the XML element names (same as serialized in option.cpp / project.cpp).
    std::map<std::string, std::string> cli_override;

    static const char* assemb_type_name(assemb_type t);
    static const char* solver_type_name(solverType t);
    static assemb_type assemb_type_from_name(const std::string& s);
    static solverType solver_type_from_name(const std::string& s);
    static assemb_type formula_type_from_name(const std::string& s);
};
#endif // optionH
