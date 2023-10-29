#ifndef OPTION_H
#define OPTION_H
#include <iostream>
#include <string>
#include <map>

class Option
{
public:
    enum AssembType { LIN, DD, NL, STAT };
    enum SolverType { DIRECT, GMRES, MATLAB};
    Option();
    virtual ~Option();
    void set(const int argc, char* argv[]);
    void PrintUsage(std::ostream& ostr);
    SolverType solver;
    AssembType assembly;
    std::string name;
    bool LIMITED;
    bool dbg;
    bool dbl;
    size_t niter;
    double toll;
    size_t hOrd;
    size_t pOrd;
    double freq; // main frequency
    double lFreq, hFreq;
    size_t nFreqs;
    size_t nHarm;
    std::string nlMtrlName;
    double kerr;
    double relax;
    size_t nDD;
    bool sweepFreq;
    bool sparam;
    bool sol;
    bool einc;
    bool tfe;
    double E[3], k[3];
    bool field;
    bool rad;
    double nTheta, nPhi;
    bool hfss;
    bool poly;
    bool unv;
    std::string polyCmd;
    bool href;
    std::string hrefCmd;
    bool verbose;
    bool msh;
    bool dd;
    bool ddn;
    bool dds;
    bool nJorGS; // Jacobi or GaussSeidel precond
    bool nl;
    bool highp;
    double power;// scalar coefficient in [W]
    bool stat;
    bool estat, mstat;
    std::map<std::string, double> Vbnd;
};
#endif // OptionH
