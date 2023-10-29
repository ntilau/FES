#include "Mtrl.h"
#include "Const.h"

Mtrl::Mtrl() : epsr(1.0), mur(1.0), sigma(0.0), tand(0.0), epsr2(0.0), kr(0.0), kerr(0.0)
{
}

Mtrl::~Mtrl()
{
}

Mtrl::Mtrl(std::string sld, std::string mat, double e, double m, double s, double td) :
    name(mat), sldName(sld), epsr(e), mur(m), sigma(s), tand(td), kr(0.0), kerr(0.0)
{
    updMtrl();
}

void Mtrl::updMtrl()
{
    epsr2 = - tand * epsr;
}

void Mtrl::updMtrl(double& freq)
{
    epsr2 = - sigma / (2.0 * Const::pi * freq * Const::eps0) - tand * epsr;
}

double Mtrl::CalcEpsr2(double& freq)
{
    return (-sigma / (2.0 * Const::pi * freq * Const::eps0) - tand * epsr);
}


