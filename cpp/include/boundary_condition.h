#ifndef bc_H
#define bc_H

#include <string>
#include <armadillo>

class bc
{
public:
    enum bcTYPE {perfect_e=-1, perfect_h, radiation, wave_port, lumped_port};
    bc();
    virtual ~bc();
    void set_type(std::string);
    size_t label;
    std::string name;
    bcTYPE type;
    int num_modes;
    double impedance; // lumped port impedance (default 50 Ohm)
    arma::cx_vec mode_beta;
    arma::cx_mat mode_vec;
    arma::cx_mat mode_vecf;
    arma::umat mode_vecdof;
    arma::uvec Faces;
};


#endif // bc_H
