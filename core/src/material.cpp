#include "material.h"
#include "constants.h"

mtrl::mtrl() : epsr(1.0), mur(1.0), sigma(0.0), tand(0.0), epsr2(0.0), kr(0.0), kerr(0.0)
{
}

mtrl::~mtrl()
{
}

mtrl::mtrl(std::string sld, std::string mat, double e, double m, double s, double td) :
    name(mat), sld_name(sld), epsr(e), mur(m), sigma(s), tand(td), kr(0.0), kerr(0.0)
{
    updmtrl();
}

void mtrl::updmtrl()
{
    epsr2 = - tand * epsr;
}

void mtrl::updmtrl(double /*freq*/)
{
    epsr2 = - tand * epsr;
}

double mtrl::calc_epsr2(double /*freq*/)
{
    return (- tand * epsr);
}


