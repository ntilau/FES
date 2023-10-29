#include "BC.h"

BC::BC() : numModes(1)
{
}

BC::~BC()
{
    ModeBeta.clear();
    ModeVec.clear();
    ModeVecDoF.clear();
}

void BC::setType(std::string tag)
{
    if(tag == "PerfectE")
    {
        type = PerfectE;
    }
    else if(tag == "PerfectH")
    {
        type = PerfectH;
    }
    else if(tag == "Radiation")
    {
        type = Radiation;
    }
    else if(tag == "WavePort")
    {
        type = WavePort;
    }
}
