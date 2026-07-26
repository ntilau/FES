#ifndef MTRL_H
#define MTRL_H

#include <string>
#include <armadillo>

class mtrl
{
public:
    mtrl();
    mtrl(std::string name, std::string sld_name, double e, double m, double s, double tand);
    virtual ~mtrl();
    void updmtrl(double freq);
    double calc_epsr2(double freq);
    void updmtrl();
    double epsr;
    double epsr2;
    double mur;
    double kr;
    double sigma;
    double eta_sigma;
    double tand;
    double kerr;
    std::string name;
    std::string sld_name;
    size_t label;
    arma::uvec Tetras;
};

#endif // MTRL_H
