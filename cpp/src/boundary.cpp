#include "boundary_condition.h"

bc::bc() : num_modes(1), impedance(50.0)
{
}

bc::~bc()
{
    mode_beta.clear();
    mode_vec.clear();
    mode_vecdof.clear();
}

void bc::set_type(std::string tag)
{
    if(tag == "perfect_e" || tag == "PerfectE")
    {
        type = perfect_e;
    }
    else if(tag == "perfect_h" || tag == "PerfectH")
    {
        type = perfect_h;
    }
    else if(tag == "radiation" || tag == "Radiation")
    {
        type = radiation;
    }
    else if(tag == "wave_port" || tag == "WavePort")
    {
        type = wave_port;
    }
    else if(tag == "lumped_port" || tag == "LumpedPort")
    {
        type = lumped_port;
    }
}
