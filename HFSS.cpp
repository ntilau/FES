#include "HFSS.h"

HFSS::HFSS(Project* prj): name(prj->opt->name), msh(prj->msh), prj(prj), debug(prj->opt->dbg)
{
    std::cout << "HFSS project files:\n";
    system(std::string("cd .\\" + name + ".hfssresults\\*.results\\*.cmesh && copy /Y current.* ..\\..\\..\\*").c_str());
    ReadMainHFSS();
    std::cout << "Reading points";
    ReadPoints();
    std::cout << ", faces";
    ReadFaces();
    std::cout << ", hydras";
    ReadHydras();
    system(std::string("del /F /Q current.*").c_str());
    FinalizeMesh();
}

HFSS::~HFSS()
{
}

void HFSS::ReadMainHFSS()
{
    int tmpInt;
    double tmpDbl;
    std::string tmpStr;
    std::string partName;
    bool solveInside;
    std::string materialName;
    std::string boundaryName;
    std::string line;
    std::ifstream fileName(std::string(name + ".hfss").c_str());
    if(fileName.is_open())
    {
        while(getline(fileName,line))   //fileName.good())
        {
            std::istringstream iss(line);
            iss >> tmpStr;
            if(tmpStr == "$begin")
            {
                iss >> tmpStr;
                if(tmpStr == "\'Materials\'")
                {
                    while(getline(fileName,line))
                    {
                        std::istringstream iss(line);
                        iss >> tmpStr;
                        if(tmpStr == "$begin")
                        {
                            HFSSMtrl hfssMaterial;
                            std::string tmpMtrl;
                            while(iss.good())
                            {
                                iss >> tmpStr;
                                tmpMtrl.append(tmpStr);
                            }
                            materialName = tmpMtrl.substr(1,tmpMtrl.size()-2);
                            while(getline(fileName,line))
                            {
                                std::istringstream iss(line);
                                iss >> tmpStr; // $begin
                                if(tmpStr.substr(0,12) == "permittivity")
                                {
                                    tmpDbl = atof(std::string(tmpStr.substr(14,tmpStr.size()-15)).data());
                                    hfssMaterial.permittivity = tmpDbl;
                                }
                                else if(tmpStr.substr(0,12) == "permeability")
                                {
                                    tmpDbl = atof(std::string(tmpStr.substr(14,tmpStr.size()-15)).data());
                                    hfssMaterial.permeability = tmpDbl;
                                }
                                else if(tmpStr.substr(0,12) == "conductivity")
                                {
                                    tmpDbl = atof(std::string(tmpStr.substr(14,tmpStr.size()-15)).data());
                                    hfssMaterial.conductivity = tmpDbl;
                                }
                                else if(tmpStr.substr(0,23) == "dielectric_loss_tangent")
                                {
                                    tmpDbl = atof(std::string(tmpStr.substr(25,tmpStr.size()-26)).data());
                                    hfssMaterial.dielectric_loss_tangent = tmpDbl;
                                }
                                else if(tmpStr == "$end")
                                {
                                    std::string tmpMtrl;
                                    while(iss.good())
                                    {
                                        iss >> tmpStr;
                                        tmpMtrl.append(tmpStr);
                                    }
                                    tmpMtrl = tmpMtrl.substr(1,tmpMtrl.size()-2);
                                    if(tmpMtrl == materialName)
                                    {
                                        hfssMaterial.name = materialName;
                                        mtrls[materialName] = hfssMaterial;
                                        break;
                                    }
                                }
                            }
                        }
                        else if(tmpStr == "$end")
                        {
                            iss >> tmpStr;
                            if(tmpStr == "\'Materials\'")
                            {
                                break;
                            }
                        }
                    }
                }
                else if(tmpStr == "\'ToplevelParts\'")
                {
                    while(getline(fileName,line))
                    {
                        std::istringstream iss(line);
                        iss >> tmpStr;
                        if(tmpStr == "$begin")
                        {
                            iss >> tmpStr;
                            if(tmpStr == "\'GeometryPart\'")
                            {
                                HFSSPart hfssPart;
                                while(getline(fileName,line))
                                {
                                    std::istringstream iss(line);
                                    iss >> tmpStr;
                                    if(tmpStr == "$begin")
                                    {
                                        iss >> tmpStr;
                                        if(tmpStr == "\'Attributes\'")
                                        {
                                            while(getline(fileName,line))
                                            {
                                                std::istringstream iss(line);
                                                iss >> tmpStr;
                                                if(tmpStr.substr(0,4) == "Name")
                                                {
                                                    std::string tmpString = tmpStr;
                                                    while(iss.good())
                                                    {
                                                        iss >> tmpStr;
                                                        tmpString.append(tmpStr);
                                                    }
                                                    hfssPart.name = std::string(tmpString.substr(6,tmpString.size()-7));
                                                }
                                                else if(tmpStr.substr(0,13) == "MaterialValue")     // for hfss v13
                                                {
                                                    std::string tmpString = tmpStr;
                                                    while(iss.good())
                                                    {
                                                        iss >> tmpStr;
                                                        tmpString.append(tmpStr);
                                                    }
                                                    hfssPart.material = std::string(tmpString.substr(16,tmpString.size()-18).data());
                                                }
                                                else if(tmpStr.substr(0,12) == "MaterialName")     // for hfss v11
                                                {
                                                    std::string tmpString = tmpStr;
                                                    while(iss.good())
                                                    {
                                                        iss >> tmpStr;
                                                        tmpString.append(tmpStr);
                                                    }
                                                    hfssPart.material = std::string(tmpString.substr(14,tmpString.size()-15));
                                                }
                                                else if(tmpStr.substr(0,11) == "SolveInside")
                                                {
                                                    hfssPart.solveInside = std::string(tmpStr.substr(12,tmpStr.size()-12)) == "true";
                                                }
                                                else if(tmpStr == "$end")
                                                {
                                                    iss >> tmpStr;
                                                    if(tmpStr == "\'Attributes\'")
                                                    {
                                                        break;
                                                    }
                                                }
                                            }
                                        }
                                        else if(tmpStr == "\'Operation\'")
                                        {
                                            while(getline(fileName,line))
                                            {
                                                std::istringstream iss(line);
                                                iss >> tmpStr;
                                                if(tmpStr.substr(0,12) == "ParentPartID")
                                                {
                                                    hfssPart.id = atoi(std::string(tmpStr.substr(13,tmpStr.size()-13)).data());
                                                }
                                                else if(tmpStr == "$end")
                                                {
                                                    iss >> tmpStr;
                                                    if(tmpStr == "\'Operation\'")
                                                    {
                                                        break;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    else if(tmpStr == "$end")
                                    {
                                        iss >> tmpStr;
                                        if(tmpStr == "\'GeometryPart\'")
                                        {
                                            parts.push_back(hfssPart);
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                        else if(tmpStr == "$end")
                        {
                            iss >> tmpStr;
                            if(tmpStr == "\'ToplevelParts\'")
                            {
                                break;
                            }
                        }
                    }
                }
                else if(tmpStr == "\'Boundaries\'")
                {
                    while(getline(fileName,line))
                    {
                        std::istringstream iss(line);
                        iss >> tmpStr; // $begin
                        if(tmpStr == "$begin")
                        {
                            iss >> tmpStr;
                            HFSSBnd hfssBoundary;
                            hfssBoundary.name = tmpStr.substr(1,tmpStr.size()-2);
                            while(getline(fileName,line))
                            {
                                std::istringstream iss(line);
                                iss >> tmpStr; // $begin
                                if(tmpStr.substr(0,9) == "BoundType")
                                {
                                    hfssBoundary.type = std::string(tmpStr.substr(11,tmpStr.size()-11));
                                    while(iss.good())
                                    {
                                        iss >> tmpStr;
                                        hfssBoundary.type += std::string(tmpStr.substr(0,tmpStr.size()-1));
                                    }
                                    if(hfssBoundary.type.substr(0,9) == "Radiation")
                                    {
                                        hfssBoundary.type = "Radiation";
                                    }
                                }
                                else if(tmpStr.substr(0,8) == "NumModes")
                                {
                                    hfssBoundary.numModes =  atoi(std::string(tmpStr.substr(9,tmpStr.size()-1)).data());
                                }
                                else if(tmpStr.substr(0,5) == "Faces")
                                {
                                    hfssBoundary.faces.push_back(atoi(std::string(tmpStr.substr(6,tmpStr.size()-7)).data()));
                                    while(iss.good())
                                    {
                                        iss >> tmpStr;
                                        hfssBoundary.faces.push_back(atoi(std::string(tmpStr.substr(0,tmpStr.size()-1)).data()));
                                    }
                                }
                                else if(tmpStr.substr(0,7) == "Objects")
                                {
                                    hfssBoundary.solids.push_back(atoi(std::string(tmpStr.substr(8,tmpStr.size()-9)).data()));
                                    while(iss.good())
                                    {
                                        iss >> tmpStr;
                                        hfssBoundary.solids.push_back(atoi(std::string(tmpStr.substr(0,tmpStr.size()-1)).data()));
                                    }
                                }
                                else if(tmpStr == "$end")
                                {
                                    iss >> tmpStr;
                                    if(tmpStr == "\'" + hfssBoundary.name + "\'")
                                    {
                                        bnds.push_back(hfssBoundary);
                                        break;
                                    }
                                }
                            }
                        }
                        else if(tmpStr == "$end")
                        {
                            iss >> tmpStr;
                            if(tmpStr == "\'Boundaries\'")
                            {
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
    else
    {
        throw std::string("Cannot find " + name + ".hfss");
    }
    fileName.close();
}

void HFSS::ReadPoints()
{
    size_t numPnts;
    int tmpInt;
    double tmpDbl, x, y, z;
    std::string tmpStr;
    std::string line;
    std::ifstream fileName(std::string("current.pnt").c_str());
    if(fileName.is_open())
    {
        getline(fileName,line);
        std::istringstream iss(line);
        iss >> tmpStr;
        if(tmpStr == "points")
        {
            iss >> numPnts;
        }
        else
        {
            throw std::string("current.pnt is not of points type");
        }
        msh->nNodes = numPnts;
        msh->nodPos.resize(numPnts,3);
        for(size_t i=0; i< numPnts; i++)
        {
            getline(fileName,line);
            std::istringstream iss(line);
            iss >> tmpStr;
            iss >> tmpInt;
            iss >> msh->nodPos(i,0);
            iss >> msh->nodPos(i,1);
            iss >> msh->nodPos(i,2);
        }
        fileName.close();
    }
    else
    {
        throw std::string(name + " project \"current.pnt\" not available");
    }
}

void HFSS::ReadFaces()
{
    size_t numFaces;
    int tmpInt, n0, n1, n2, t0, t1;
    std::vector<size_t> bndTag;
    size_t bndTagNum;
    double tmpDbl;
    std::string tmpStr;
    std::string line;
    std::ifstream fileName(std::string("current.fac").c_str());
    if(fileName.is_open())
    {
        getline(fileName,line);
        std::istringstream iss(line);
        iss >> tmpStr;
        if(tmpStr == "faces_v2")
        {
            iss >> numFaces;
        }
        else
        {
            throw std::string("current.fac is not of faces_v2 type");
        }
        msh->nFaces = numFaces;
        msh->facNodes.resize(numFaces,3);
        facHFSStag.resize(numFaces);
        msh->facLab.resize(numFaces);
        msh->facAdjTet.set_size(numFaces);
        for(size_t i=0; i< numFaces; i++)
        {
            getline(fileName,line);
            std::istringstream iss(line);
            iss >> tmpStr; // f
            iss >> tmpInt; // id
            iss >> msh->facNodes(i,0);
            iss >> msh->facNodes(i,1);
            iss >> msh->facNodes(i,2);
            iss >> tmpStr; // h
            iss >> t0;
            iss >> t1;
            iss >> bndTagNum;
            bndTag.resize(bndTagNum);
            for(size_t ibnd=0; ibnd<bndTagNum; ibnd++)
            {
                iss >> bndTag[ibnd];
            }
            facHFSStag[i] = bndTag;
        }
        {
            getline(fileName,line);
            std::istringstream iss(line);
            iss >> tmpStr;
            if(tmpStr != "end_face")
            {
                throw std::string("end_face not found");
            }
        }
        {
            getline(fileName,line);
            std::istringstream iss(line);
            iss >> tmpStr;
            if(tmpStr == "NumFaces")
            {
                size_t tag, label;
                iss >> bndTagNum;
                for(size_t i = 0; i<bndTagNum; i++)
                {
                    getline(fileName,line);
                    std::istringstream iss(line);
                    iss >> tag;
                    iss >> label;
                    bndMap[label].push_back(tag);
                }
            }
        }
        fileName.close();
    }
    else
    {
        throw std::string(name + " project file \"current.fac\" not available");
    }
}
void HFSS::ReadHydras()
{
    size_t numHydras;
    int tmpInt, n0, n1, n2, n3, f0, f1, f2, f3, b0, b1, b2, b3;
    size_t l0, l1, l2, l3, l4, l5, s0;
    double tmpDbl;
    std::string tmpStr;
    std::string line;
    std::ifstream fileName(std::string("current.hyd").c_str());
    if(fileName.is_open())
    {
        getline(fileName,line);
        std::istringstream iss(line);
        iss >> tmpStr;
        if(tmpStr == "hydras")
        {
            iss >> numHydras;
        }
        else
        {
            throw std::string("current.fac is not of faces_v2 type");
        }
        msh->nTetras = numHydras;
        msh->tetNodes.resize(numHydras,4);
        msh->tetEdges.resize(numHydras,6);
        msh->tetFaces.resize(numHydras,4);
        msh->tetLab.resize(numHydras);
        hfssid.resize(numHydras);
        for(size_t i=0; i< numHydras; i++)
        {
            getline(fileName,line);
            {
                std::istringstream iss(line);
                iss >> tmpStr; // h
                iss >> tmpInt; // id
                iss >> msh->tetNodes(i,0);
                iss >> msh->tetNodes(i,1);
                iss >> msh->tetNodes(i,2);
                iss >> msh->tetNodes(i,3);
            }
            getline(fileName,line);
            {
                std::istringstream iss(line);
                iss >> tmpStr; // f
                iss >> msh->tetFaces(i,0);
                iss >> msh->tetFaces(i,1);
                iss >> msh->tetFaces(i,2);
                iss >> msh->tetFaces(i,3);
            }
            getline(fileName,line);
            {
                std::istringstream iss(line);
                iss >> tmpStr; // b
                iss >> tmpInt;
                iss >> b0;
                iss >> b1;
                iss >> b2;
                iss >> b3;
            }
            getline(fileName,line);
            {
                std::istringstream iss(line);
                iss >> tmpStr; // l
                iss >> l0;
                iss >> l1;
                iss >> l2;
                iss >> l3;
                iss >> l4;
                iss >> l5;
            }
            getline(fileName,line);
            {
                std::istringstream iss(line);
                iss >> tmpStr; // s
                iss >> hfssid[i];
            }
        }
        {
            getline(fileName,line);
            std::istringstream iss(line);
            iss >> tmpStr;
            if(tmpStr != "end_hydra")
            {
                throw std::string("end_hydra not found");
            }
        }
        fileName.close();
        msh->facNodes -= 1;
        msh->tetNodes -= 1;
        msh->tetFaces -= 1;
    }
    else
    {
        throw std::string(name + " project file \"current.hyd\" not available");
    }
}

void HFSS::FinalizeMesh()
{
    std::cout << "\n";
    tetFlag = std::vector<bool>(msh->nTetras, false);
    facFlag = std::vector<bool>(msh->nFaces, false);
  