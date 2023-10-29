#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>

#include "Config.h"
#include "Mem.h"
#include "Project.h"
#include "Option.h"
#include "HFSS.h"
#include "TetGen.h"
#include "Mesh.h"

Project::Project(std::ofstream& logFile, Option& pOpt) : opt(&pOpt), msh(new Mesh())
{
    arma::wall_clock prjtt;
    prjtt.tic();
    logFile << "% Loading files:\n";
    logFile << "Project: " << opt->name << "\n";
    logFile << "Homogeneous refinement: p = " << opt->pOrd << ", h = " << opt->hOrd << "\n";
    if(opt->verbose)
    {
        std::cout << "Project:   " << opt->name << "\n";
        std::cout << "Main frequency: " << opt->freq << "\n";
        std::cout << "p = " << opt->pOrd << ", h = " << opt->hOrd << "\n";
    }
    if(opt->hfss)
    {
        logFile << "Parsing HFSS project files\n";
        HFSS(this);
        saveFE();
    }
    else if(opt->poly)
    {
        logFile << "Parsing TetGen based project file\n";
        TetGen(this);
        saveFE();
    }
    else
    {
        logFile << "Parsing FE project files\n";
        loadFE();
    }
    if(opt->hOrd > 0)
    {
        std::cout << "Performing h refinement\n";
        for(size_t i=0; i<opt->hOrd; i++)
        {
            msh->RefineHomogeneous();
        }
        saveFE();
    }
    if(opt->href & !opt->poly)
    {
        logFile << "Performing TetGen based refinement\n";
        TetGen(this);
        //opt->name +=  "_" + opt->hrefCmd;
        saveFE();
    }
    // Reodering elements
    // msh->Reorder();
    // Mesh statistics
    logFile << "Nodes  = " << msh->nNodes << "\n"
            << "Edges  = " << msh->nEdges << "\n"
            << "Faces  = " << msh->nFaces << "\n"
            << "Tetras = " << msh->nTetras << "\n";
    logFile << "++" << prjtt.toc() << " s\n";
    if(opt->verbose)
        std::cout << "Nodes  = " << msh->nNodes << "\n"
                  << "Edges  = " << msh->nEdges << "\n"
                  << "Faces  = " << msh->nFaces << "\n"
                  << "Tetras = " << msh->nTetras << "\n";
    if(opt->msh)
    {
        msh->SaveField(opt->name);
        exit(0);
    }
    // setting up kerr type
    if(opt->nl)
    {
        for(size_t i=0; i < msh->tetMtrl.size(); i++)
        {
            if(msh->tetMtrl[i].name == opt->nlMtrlName)
            {
                msh->tetMtrl[i].kerr = opt->kerr;
            }
        }
    }
    // mesh partitioning with metis
    if(opt->dd)
    {
        msh->PartitionMesh(opt->nDD);
        logFile << "Domains = " << opt->nDD << "\n";
        if(opt->verbose)
        {
            std::cout << "Domains = " << opt->nDD << "\n";
        }
    }
    if(opt->verbose)
    {
        MemStat::print(std::cout);
    }
    MemStat::print(logFile);
}

Project::~Project()
{
}


void Project::loadFE()
{
    size_t tmpInt;
    double tmpDbl;
    std::string tmpStr;
    std::string line;
    std::ifstream fileName(std::string(opt->name + "_Prj.txt").c_str(), std::ios::in | std::ios::binary);
    if(fileName.is_open())
    {
        while(getline(fileName,line))
        {
            std::istringstream iss(line);
            iss >> tmpStr;
            if(tmpStr == "#Solids")
            {
                iss >> tmpInt;
                for(size_t i = 0; i < tmpInt; i++)
                {
                    Mtrl mtr;
                    mtr.label = i;
                    getline(fileName,line);
                    std::istringstream iss(line);
                    iss >> mtr.sldName;
                    iss >> mtr.epsr;
                    iss >> mtr.mur;
                    iss >> mtr.sigma;
                    iss >> mtr.tand;
                    iss >> mtr.name;
                    mtr.updMtrl();
                    msh->tetMtrl.push_back(mtr);
                }
            }
            else if(tmpStr == "#Boundaries")
            {
                iss >> tmpInt;
                for(size_t i = 0; i < tmpInt; i++)
                {
                    size_t type;
                    BC bc;
                    bc.label = i;
                    getline(fileName,line);
                    std::istringstream iss(line);
                    iss >> bc.name;
                    iss >> type;
                    bc.type = (BC::BCTYPE) type;
                    if(bc.type == BC::WavePort)
                    {
                        iss >> bc.numModes;
                    }
                    msh->facBC.push_back(bc);
                }
            }
            else if(tmpStr == "#Nodes")
            {
                iss >> msh->nNodes;
                msh->nodPos.resize(msh->nNodes,3);
                for(size_t i = 0; i < msh->nNodes; i++)
                {
                    getline(fileName,line);
                    std::istringstream iss(line);
                    iss >> msh->nodPos(i,0);
                    iss >> msh->nodPos(i,1);
                    iss >> msh->nodPos(i,2);
                }
            }
            else if(tmpStr == "#Edges")
            {
                iss >> msh->nEdges;
                msh->edgNodes.resize(msh->nEdges,2);
                for(size_t i = 0; i < tmpInt; i++)
                {
                    getline(fileName,line);
                    std::istringstream iss(line);
                    iss >> msh->edgNodes(i,0);
                    iss >> msh->edgNodes(i,1);
                }
            }
            else if(tmpStr == "#Faces")
            {
                iss >> msh->nFaces;
                msh->facNodes.resize(msh->nFaces,3);
                msh->facEdges.resize(msh->nFaces,3);
                msh->facAdjTet.set_size(msh->nFaces);
                msh->facLab.resize(msh->nFaces);
                msh->facLab.fill(msh->maxLab);
                for(size_t i = 0; i < msh->nFaces; i++)
                {
                    getline(fileName,line);
                    std::istringstream iss(line);
                    iss >> msh->facNodes(i,0);
                    iss >> msh->facNodes(i,1);
                    iss >> msh->facNodes(i,2);
                    iss >> msh->facEdges(i,0);
                    iss >> msh->facEdges(i,1);
                    iss >> msh->facEdges(i,2);
                }
            }
            else if(tmpStr == "#Tetras")
            {
                iss >> msh->nTetras;
                msh->tetNodes.resize(msh->nTetras,4);
                msh->tetEdges.resize(msh->nTetras,6);
                msh->tetFaces.resize(msh->nTetras,4);
                msh->tetLab.resize(msh->nTetras);
                msh->tetLab.fill(msh->maxLab);
                for(size_t i = 0; i < msh->nTetras; i++)
                {
                    getline(fileName,line);
                    std::istringstream iss(line);
                    iss >> msh->tetNodes(i,0);
                    iss >> msh->tetNodes(i,1);
                    iss >> msh->tetNodes(i,2);
                    iss >> msh->tetNodes(i,3);
                    iss >> msh->tetEdges(i,0);
                    iss >> msh->tetEdges(i,1);
                    iss >> msh->tetEdges(i,2);
                    iss >> msh->tetEdges(i,3);
                    iss >> msh->tetEdges(i,4);
                    iss >> msh->tetEdges(i,5);
                    iss >> msh->tetFaces(i,0);
                    iss >> msh->tetFaces(i,1);
                    iss >> msh->tetFaces(i,2);
                    iss >> msh->tetFaces(i,3);
                    arma::uvec adjTet(1);
                    adjTet(0) = i;
                    msh->facAdjTet(msh->tetFaces(i,0)) = arma::join_cols(msh->facAdjTet(msh->tetFaces(i,0)), adjTet);
                    msh->facAdjTet(msh->tetFaces(i,1)) = arma::join_cols(msh->facAdjTet(msh->tetFaces(i,1)), adjTet);
                    msh->facAdjTet(msh->tetFaces(i,2)) = arma::join_cols(msh->facAdjTet(msh->tetFaces(i,2)), adjTet);
                    msh->facAdjTet(msh->tetFaces(i,3)) = arma::join_cols(msh->facAdjTet(msh->tetFaces(i,3)), adjTet);
                }
            }
            else if(tmpStr == "#BoundaryFaces")
            {
                size_t bndFaces, bndSize;
                std::string name;
                iss >> bndFaces;
                for(size_t i = 0; i < bndFaces; i++)
                {
                    getline(fileName,line);
                    std::istringstream iss(line);
                    iss >> name;
                    iss >> bndSize;
                    msh->facBC[i].Faces.resize(bndSize);
                    for(size_t j = 0; j < bndSize; j++)
                    {
                        getline(fileName,line);
                        std::istringstream iss(line);
                        iss >> msh->facBC[i].Faces(j);
                        msh->facLab(msh->facBC[i].Faces(j)) = i;
                    }
                }
            }
            else if(tmpStr == "#SolidTetras")
            {
                size_t sldTetras, sldSize;
                std::string name;
                iss >> sldTetras;
                for(size_t i = 0; i < sldTetras; i++)
                {
                    getline(fileName,line);
                    std::istringstream iss(line);
                    iss >> name;
                    iss >> sldSize;
                    msh->tetMtrl[i].Tetras.resize(sldSize);
                    for(size_t j = 0; j < sldSize; j++)
                    {
                        getline(fileName,line);
                        std::istringstream iss(line);
                        iss >> msh->tetMtrl[i].Tetras(j);
                        msh->tetLab(msh->tetMtrl[i].Tetras(j)) = i;
                    }
                }
            }
        }
    }
    else
    {
        throw std::string(opt->name + "_Prj.txt" + " not available");
    }
}

void Project::saveFE()
{
    std::ofstream out(std::string(opt->name + "_Prj.txt").c_str(), std::ios::out | std::ios::binary | std::ios::ate);
    out << "#Solids " << msh->tetMtrl.size() << "\n";
    for(size_t i = 0; i < msh->tetMtrl.size(); i++)
    {
        out << msh->tetMtrl[i].sldName << " "
            << msh->tetMtrl[i].epsr << " "
            << msh->tetMtrl[i].mur << " "
            << msh->tetMtrl[i].sigma << " "
            << msh->tetMtrl[i].tand << " "
            << msh->tetMtrl[i].name << "\n";
    }
    out << "#Boundaries " << msh->facBC.size() << "\n";
    for(size_t i = 0; i < msh->facBC.size(); i++)
    {
        out << msh->facBC[i].name << " " << msh->facBC[i].type;
        if(msh->facBC[i].type == BC::WavePort)
        {
            out << " " << msh->facBC[i].numModes;
        }
        out << "\n";
    }
    out << "#Nodes " << msh->nNodes << "\n";
    for(size_t i = 0; i < msh->nNodes; i++)
    {
        out << std::scientific
            << std::setprecision(16) << msh->nodPos(i,0) << " "
            << std::setprecision(16) << msh->nodPos(i,1) << " "
            << std::setprecision(16) << msh->nodPos(i,2) << "\n";
    }
    out << "#Edges " << msh->nEdges << "\n";
    for(size_t i = 0; i < msh->nEdges; i++)
    {
        out << msh->edgNodes(i,0) << " "
            << msh->edgNodes(i,1) << "\n";
    }
    out << "#Faces " << msh->nFaces << "\n";
    for(size_t i = 0; i < msh->nFaces; i++)
    {
        out << msh->facNodes(i,0) << " "
            << msh->facNodes(i,1) << " "
            << msh->facNodes(i,2) << " "
            << msh->facEdges(i,0) << " "
            << msh->facEdges(i,1) << " "
            << msh->facEdges(i,2) << "\n";
    }
    out << "#Tetras " << msh->nTetras << "\n";
    for(size_t i = 0; i < msh->nTetras; i++)
    {
        out << msh->tetNodes(i,0) << " "
            << msh->tetNodes(i,1) << " "
            << msh->tetNodes(i,2) << " "
            << msh->tetNodes(i,3) << " "
            << msh->tetEdges(i,0) << " "
            << msh->tetEdges(i,1) << " "
            << msh->tetEdges(i,2) << " "
            << msh->tetEdges(i,3) << " "
            << msh->tetEdges(i,4) << " "
            << msh->tetEdges(i,5) << " "
            << msh->tetFaces(i,0) << " "
            << msh->tetFaces(i,1) << " "
            << msh->tetFaces(i,2) << " "
            << msh->tetFaces(i,3) << "\n";
    }
    out << "#BoundaryFaces " << msh->facBC.size() << "\n";
    for(size_t i = 0; i < msh->facBC.size(); i++)
    {
        out << msh->facBC[i].name << " " << msh->facBC[i].Faces.size() << "\n";
        for(size_t j = 0; j < msh->facBC[i].Faces.size(); j++)
        {
            out << msh->facBC[i].Faces[j] << "\n";
        }
    }
    out << "#SolidTetras " << msh->tetMtrl.size() << "\n";
    for(size_t i = 0; i < msh->tetMtrl.size(); i++)
    {
        out << msh->tetMtrl[i].sldName << " " << msh->tetMtrl[i].Tetras.size() << "\n";
        for(size_t j = 0; j < msh->tetMtrl[i].Tetras.size(); j++)
        {
            out << msh->tetMtrl[i].Tetras[j] << "\n";
        }
    }
    out.close();
}
