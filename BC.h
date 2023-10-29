#ifndef BC_H
#define BC_H

#include <string>
#include <armadillo>

class BC
{
public:
    enum BCTYPE {PerfectE=-1, PerfectH, Radiation, WavePort};
    BC();
    virtual ~BC();
    void setType(std::string);
    size_t label;
    std::string name;
    BCTYPE type;
    int numModes;
    arma::cx_vec ModeBeta;
    arma::cx_mat ModeVec;
    arma::cx_mat ModeVecf;
    arma::umat ModeVecDoF;
    arma::uvec Faces;
};


#endif // BC_H
