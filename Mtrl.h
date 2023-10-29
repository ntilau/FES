#ifndef MTRL_H
#define MTRL_H

#include <string>
#include <armadillo>

class Mtrl
{
public:
    Mtrl();
    Mtrl(std::string name, std::string sldName, double e, double m, double s, double tand);
    virtual ~Mtrl();
    void updMtrl(double& freq);
    double CalcEpsr2(double& freq);
    void updMtrl();
    double epsr;
    double epsr2;
    double mur;
    double kr;
    double sigma;
    double etaSigma;
    double tand;
    double kerr;
    std::string name;
    std::string sldName;
    size_t label;
    arma::uvec Tetras;
};

#endif // MTRL_H
