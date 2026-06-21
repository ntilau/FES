#ifndef POSTPROCESSOR_H
#define POSTPROCESSOR_H

#include <fstream>
#include <armadillo>

class eq_sys;

class post_processor
{
public:
    post_processor(eq_sys& sys, std::ofstream& log);
    void save_data();
private:
    eq_sys& sys;
    std::ofstream& log;
    arma::wall_clock tt, lt;
};

#endif // POSTPROCESSOR_H
