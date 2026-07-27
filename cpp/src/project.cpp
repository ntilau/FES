#include <string>
#include <stdexcept>
#include <iostream>
#include <fstream>

#include "configuration.h"
#include "memory.h"
#include "project.h"
#include "option.h"
#include "pre_processor.h"
#include "mesh.h"

project::project(std::ofstream& logFile, option& pOpt) : opt(&pOpt), msh(new mesh())
{
    arma::wall_clock prjtt;
    prjtt.tic();
    logFile << "% Loading files:\n";
    logFile << "project: " << opt->name << "\n";
    logFile << "Homogeneous refinement: p = " << opt->p_ord << ", h = " << opt->h_ord << "\n";
    std::cout << "project:   " << opt->name << "\n";
    std::cout << "Main frequency: " << opt->freq << "\n";
    std::cout << "p = " << opt->p_ord << ", h = " << opt->h_ord << "\n";
    logFile << "Parsing .poly project file\n";
    preprocessing(this);
    if(opt->h_ord > 0)
    {
        std::cout << "Performing h refinement\n";
        for(size_t i=0; i<opt->h_ord; i++)
        {
            msh->refine_homogeneous();
        }
    }
    // Normalize node order in edges/faces/tetras so mesh is canonical
    msh->normalize_order();
    // mesh statistics
    logFile << "Nodes  = " << msh->nNodes << "\n"
            << "Edges  = " << msh->nEdges << "\n"
            << "Faces  = " << msh->nFaces << "\n"
            << "Tetras = " << msh->nTetras << "\n";
    msh->check_regular(logFile);
    logFile << "++" << prjtt.toc() << " s\n";
    std::cout << "Nodes  = " << msh->nNodes << "\n"
              << "Edges  = " << msh->nEdges << "\n"
              << "Faces  = " << msh->nFaces << "\n"
              << "Tetras = " << msh->nTetras << "\n";
    msh->Savefield(opt->name);
    // mesh partitioning (domain decomposition)
    if(opt->dd)
    {
        msh->Partitionmesh(opt->n_dd);
        logFile << "Domains = " << opt->n_dd << "\n";
        std::cout << "Domains = " << opt->n_dd << "\n";
    }
    mem_stat::print(std::cout);
    mem_stat::print(logFile);
}

project::~project()
{
}
