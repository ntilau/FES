#include "Option.h"
#include <string>
#include <iostream>
#include <stdlib.h>
#include <string.h>

Option::Option(): LIMITED(true), hOrd(0), pOrd(1), solver(DIRECT), assembly(LIN), sparam(true), sol(false), field(false),
    hfss(false), poly(false), unv(false), href(false), verbose(true), dbg(false), freq(0), lFreq(0), hFreq(0), nFreqs(1),
    msh(false), einc(false), rad(false), dd(false), nl(false), highp(false), power(1.0), stat(false), tfe(false),
    nJorGS(true), ddn(false), dds(false), dbl(true) {}

Option::~Option() {}

void Option::set(const int argc, char* argv[])
{
    if(argc < 3)
    {
        PrintUsage(std::cout);
        exit(0);
    }
    this->name = std::string(argv[1]);
    if(name.length() == 0)
    {
        PrintUsage(std::cout);
        exit(1);
    }
    this->freq = atof(argv[2]);
    if(freq < 0)
    {
        PrintUsage(std::cout);
        exit(1);
    }
    else if(freq == 0)
    {
        stat = true;
        sparam = false;
        field = true;
        assembly = STAT;
    }
    for(int cnt = 3; cnt < argc; ++cnt)
    {
        if(strcmp(argv[cnt], "++") == 0)
        {
            highp = true;
            continue;
        }
        if(strcmp(argv[cnt], "+dbg") == 0)
        {
            dbg = true;
            continue;
        }
        if(strcmp(argv[cnt], "+dbl") == 0)
        {
            dbl = true;
            continue;
        }
        if(strcmp(argv[cnt], "+sgl") == 0)
        {
            dbl = false;
            continue;
        }
        if(strcmp(argv[cnt], "+hfss") == 0)
        {
            hfss = true;
            continue;
        }
        if(strcmp(argv[cnt], "+unv") == 0)
        {
            unv = true;
            continue;
        }
        if(strcmp(argv[cnt], "+poly") == 0)
        {
            poly = true;
            ++cnt;
            if(cnt >= argc)
            {
                std::cout << "Missing poly command string\n";
                exit(1);
            }
            polyCmd = argv[cnt];
            continue;
        }
        if(strcmp(argv[cnt], "+href") == 0)
        {
            href = true;
            ++cnt;
            if(cnt >= argc)
            {
                std::cout << "Missing href command string\n";
                exit(1);
            }
            hrefCmd = argv[cnt];
            continue;
        }
        if(strcmp(argv[cnt], "+volt") == 0)
        {
            ++cnt;
            if(cnt >= argc)
            {
                std::cout << "Missing applied voltage boundary\n";
                exit(1);
            }
            std::string bnd = argv[cnt];
            ++cnt;
            if(cnt >= argc)
            {
                std::cout << "Missing applied voltage value\n";
                exit(1);
            }
            Vbnd[bnd] = atof(argv[cnt]);
            continue;
        }
        if(strcmp(argv[cnt], "+pow") == 0)
        {
            ++cnt;
            if(cnt >= argc)
            {
                std::cout << "Power scaling assigned to 1\n";
            }
            else
            {
                this->power = atof(argv[cnt]);
            }
            continue;
        }
        if(strcmp(argv[cnt], "+p") == 0)
        {
            ++cnt;
            if(cnt >= argc)
            {
                this->pOrd = 1;
            }
            else
            {
                int tmp = atoi(argv[cnt]);
                if(tmp < 1 && tmp > 4)
                {
                    PrintUsage(std::cout);
                    exit(1);
                }
                this->pOrd = tmp;
            }
            continue;
        }
        if(strcmp(argv[cnt], "+h") == 0)
        {
            ++cnt;
            if(cnt >= argc)
            {
                this->hOrd = 0;
            }
            else
            {
                int tmp = atoi(argv[cnt]);
                if(tmp < 0)
                {
                    PrintUsage(std::cout);
                    exit(1);
                }
                this->hOrd = tmp;
            }
            continue;
        }
        if(strcmp(argv[cnt], "+field") == 0)
        {
            field = true;
            continue;
        }
        if(strcmp(argv[cnt], "+rad") == 0)
        {
            rad = true;
            ++cnt;
            if(cnt >= argc)
            {
                this->nTheta = 101;
                this->nPhi = 201;
            }
            else
            {
                double tmp = atoi(argv[cnt]);
                if(tmp <= 0)
                {
                    PrintUsage(std::cout);
                    exit(1);
                }
                this->nTheta = tmp;
                ++cnt;
                if(cnt >= argc)
                {
                    this->nTheta = 101;
                    this->nPhi = 201;
                }
                else
                {
                    double tmp = atoi(argv[cnt]);
                    if(tmp <= 0)
                    {
                        PrintUsage(std::cout);
                        exit(1);
                    }
                    this->nPhi = tmp;
                }
            }
            continue;
        }
        if(strcmp(argv[cnt], "+msh") == 0)
        {
            msh = true;
            continue;
        }
        if(strcmp(argv[cnt], "+sol") == 0)
        {
            sol = true;
            continue;
        }
        if(strcmp(argv[cnt], "+tfe") == 0)
        {
            tfe = true;
            continue;
        }
        if(strcmp(argv[cnt], "-tfe") == 0)
        {
            tfe = false;
            continue;
        }
        if(strcmp(argv[cnt], "+einc") == 0)
        {
            ++cnt;
            for(size_t i=0; i<3; i++, cnt++)
            {
                if(cnt >= argc)
                {
                    std::cout << "Not enough Einc data\n";
                    exit(1);
                }
                E[i] = atof(argv[cnt]);
            }
            for(size_t i=0; i<3; i++, cnt++)
            {
                if(cnt >= argc)
                {
                    std::cout << "Not enough Einc data\n";
                    exit(1);
                }
                k[i] = atof(argv[cnt]);
            }
            --cnt;
            einc = true;
            sparam = false;
            field = true;
            continue;
        }
        if(strcmp(argv[cnt], "+sparam") == 0)
        {
            sparam = true;
            continue;
        }
        if(strcmp(argv[cnt], "-sparam") == 0)
        {
            sparam = false;
            continue;
        }
        if(strcmp(argv[cnt], "+jc") == 0)
        {
            nJorGS = false;
            continue;
        }
        if(strcmp(argv[cnt], "+gs") == 0)
        {
            nJorGS = true;
            continue;
        }
        if(strcmp(argv[cnt], "+dd") == 0)
        {
            ++cnt;
            if(cnt >= argc)
            {
                this->nDD = 2;
            }
            else
            {
                int tmpdd = atoi(argv[cnt]);
                if(tmpdd <= 0)
                {
                    PrintUsage(std::cout);
                    exit(1);
                }
                nDD = tmpdd;
            }
            assembly = DD;
            dd = true;
            continue;
        }
        if(strcmp(argv[cnt], "+dds") == 0)
        {
            ++cnt;
            if(cnt >= argc)
            {
                this->nDD = 2;
            }
            else
            {
                int tmpdd = atoi(argv[cnt]);
                if(tmpdd <= 0)
                {
                    PrintUsage(std::cout);
                    exit(1);
                }
                nDD = tmpdd;
            }
            assembly = DD;
            dd = true;
            dds = true;
            //tfe = true;
            continue;
        }
        if(strcmp(argv[cnt], "+ddn") == 0)
        {
            ++cnt;
            if(cnt >= argc)
            {
                this->nDD = 2;
            }
            else
            {
                int tmpdd = atoi(argv[cnt]);
                if(tmpdd <= 0)
                {
                    PrintUsage(std::cout);
                    exit(1);
                }
                nDD = tmpdd;
            }
            assembly = DD;
            dd = true;
            ddn = true;
            continue;
        }
        if(strcmp(argv[cnt], "+direct") == 0)
        {
            solver = DIRECT;
            continue;
        }
        if(strcmp(argv[cnt], "+gmres") == 0)
        {
            ++cnt;
            if(cnt >= argc)
            {
                this->toll = 1e-2;
                this->niter = 100;
            }
            else
            {
                double tmp = atof(argv[cnt]);
                if(tmp <= 0)
                {
                    PrintUsage(std::cout);
                    exit(1);
                }
                this->toll = tmp;
                ++cnt;
                if(cnt >= argc)
                {
                    this->toll = 1e-2;
                    this->niter = 100;
                }
                else
                {
                    size_t tmp = atoi(argv[cnt]);
                    if(tmp <= 0)
                    {
                        PrintUsage(std::cout);
                        exit(1);
                    }
                    this->niter = tmp;
                }
            }
            solver = GMRES;
            continue;
        }
        if(strcmp(argv[cnt], "+matlab") == 0)
        {
            solver = MATLAB;
            sparam = false;
            continue;
        }
        if(strcmp(argv[cnt], "+nl") == 0)
        {
            ++cnt;
            if(cnt >= argc)
            {
                std::cout << "Nonlinear options not complete\n";
                exit(1);
            }
            int tmpnh = atoi(argv[cnt]);
            if(tmpnh <= 0)
            {
                PrintUsage(std::cout);
                exit(1);
            }
            nHarm = tmpnh;
            ++cnt;
            if(cnt >= argc)
            {
                std::cout << "Nonlinear options not complete\n";
                exit(1);
            }
            nlMtrlName = argv[cnt];
            ++cnt;
            if(cnt >= argc)
            {
                std::cout << "Nonlinear options not complete\n";
                exit(1);
            }
            kerr = atof(argv[cnt]);
            ++cnt;
            if(cnt >= argc)
            {
                std::cout << "Nonlinear options not complete\n";
                exit(1);
            }
            relax = atof(argv[cnt]);
            assembly = NL;
            this->nl = true;
            this->tfe = true;
            continue;
        }
        if(strcmp(argv[cnt], "+fr") == 0)
        {
            ++cnt;
            if(cnt >= argc)
            {
                this->nFreqs = 1;
                this->lFreq = freq;
                this->hFreq = freq;
            }
            else
            {
                double tmp = atof(argv[cnt]);
                if(tmp <= 0)
                {
                    PrintUsage(std::cout);
                    exit(1);
                }
                this->lFreq = tmp;
                ++cnt;
                if(cnt >= argc)
                {
                    this->nFreqs = 1;
                    this->lFreq = freq;
                    this->hFreq = freq;
                }
                else
                {
                    double tmp = atof(argv[cnt]);
                    if(tmp <= 0)
                    {
                        PrintUsage(std::cout);
                        exit(1);
                    }
                    this->hFreq = tmp;
                    ++cnt;
                    if(cnt >= argc)
                    {
                        this->nFreqs = 1;
                        this->lFreq = freq;
                        this->hFreq = freq;
                    }
                    else
                    {
                        int tmp = atoi(argv[cnt]);
                        if(tmp <= 0)
                        {
                            PrintUsage(std::cout);
                            exit(1);
                        }
                        this->nFreqs = tmp;
                    }
                }
            }
            continue;
        }
        if(strcmp(argv[cnt], "-verbose") == 0)
        {
            verbose = false;
            continue;
        }
        std::cout << "\nCannot parse " << argv[cnt] << "\n";
        PrintUsage(std::cout);
        exit(0);
    }
}

void Option::PrintUsage(std::ostream& ostr)
{
    ostr << "Usage: FE\n";
    ostr << "  name                         project name\n";
    ostr << "  freq                         project frequency [Hz]\n";
    ostr << "        ==  0 Hz  -> Electrostatic formulation\n";
    //ostr << "        <  1 MHz  -> Electric and Magnetic(quasi-static) formulations\n";
    ostr << "        >= 1 MHz  -> Electromagetic formulation\n";
    ostr << "  [++]                         increase process priority\n";
    ostr << "  [+fr $lf $hf $n]             perform discrete frequency sweep\n";
    ostr << "  [+pow $p]                    power scaling at ports (default = 1[W])\n";
    ostr << "  [+p $n]                      select polynomial order (1-3)\n";
    ostr << "  [+h $n]                      perform homogeneous mesh refinement\n";
    ostr << "  [+poly q__a__]               extract Tetgen based project data\n";
    ostr << "  [+hfss]                      extract HFSS project data\n";
    ostr << "  [+tfe]                       Transfinite Elements formulation on ports\n";
    ostr << "  [+href q__a__Y]              quality mesh refinement with TetGen\n";
    ostr << "  [+einc Ex Ey Ez kx ky kz]    apply incident plane wave\n";
    ostr << "                               and disable +sparam\n";
    ostr << "  [+volt $bnd $pot]            apply voltage [V] to PerfectE boundary\n";
    ostr << "  [+matlab]                    dump to Matlab in MatrixMarket format\n";
    ostr << "  [+sgl]                       decrease solver precision to single\n";
    ostr << "  [+direct]                    solve direct\n";
    ostr << "  [+gmres $tol $restart]       solve with GMRes\n";
    ostr << "  [+nl $h $mtrl $kerr $relax]  solve direct Kerr materials with $h harm.\n";
    ostr << "  [+dd $n]                     apply partitioning in $n regions\n";
    ostr << "            [+gs]              Gauss-Seidel precond. (default)\n";
    ostr << "            [+jc]              Jacobi precond.\n";
    ostr << "  [+sol]                       write solution\n";
    ostr << "  [+solid]                     write VTK solids\n";
    ostr << "  [+field]                     write VTK fields\n";
    ostr << "  [+rad $nTheta $nPhi]         write VTK radiation solids\n";
    ostr << "  [+msh]                       write VTK mesh\n";
    ostr << "  [+sparam]                    write S-parameters\n";
    ostr << "  [-verbose]                   be silent\n";
    ostr << "Default Options: +direct +sparam +p 1 (rad. conditions on ports)\n";
    ostr << "Example: FE MagicTee 1e11 +hfss +field\n";
}
