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
    nodFlag = std::vector<bool>(msh->nNodes, false);
    size_t idx = 0, nidx = 0, eidx = 0, fidx = 0, tidx = 0;
    std::map<std::string, std::vector<size_t> > gSolidTets;
    for(std::vector<HFSSPart>::iterator it = parts.begin(); it != parts.end(); it++)
    {
        if(it->solveInside)
        {
            Mtrl mtr(it->name, it->material, mtrls[it->material].permittivity,
                     mtrls[it->material].permeability, mtrls[it->material].conductivity,
                     mtrls[it->material].dielectric_loss_tangent);
            for(size_t tid = 0; tid < msh->nTetras; tid++)
            {
                if(it->id == hfssid[tid])
                {
                    tetFlag[tid] = true;
                    mtr.label = msh->tetMtrl.size();
                    gSolidTets[it->name].push_back(tid);
                }
            }
            if(gSolidTets[it->name].size() == 0)
            {
                gSolidTets.erase(it->name);
            }
            else
            {
                mtr.Tetras.resize(gSolidTets[it->name].size());
                size_t idx = 0;
                for(std::vector<size_t>::iterator iter = gSolidTets[it->name].begin();
                        iter != gSolidTets[it->name].end(); iter++)
                {
                    mtr.Tetras(idx++) = *iter;
                    msh->tetLab(*iter) = mtr.label;
                }
                std::cout << mtr.sldName << " " << mtr.name << "\n";
                msh->tetMtrl.push_back(mtr);
            }
        }
    }
    for(std::vector<HFSSBnd>::iterator it = bnds.begin();  it != bnds.end(); it++)
    {
        BC bc;
        bc.setType(it->type);
        bc.name = it->name;
        bc.label = msh->facBC.size();
        std::cout << bc.name << " " << bc.type;
        if(bc.type == BC::WavePort)
        {
            bc.numModes = it->numModes;
            std::cout << " " << bc.numModes;
        }
        std::cout << "\n";
        msh->facBC.push_back(bc);
    }
    if(debug)
    {
        std::cout << "TetMap\n";
    }
    std::vector<size_t> tetMap(msh->nTetras,UINT_MAX);
    tidx = 0;
    for(size_t tid = 0; tid < msh->nTetras ; tid++)
    {
        if(tetFlag[tid])
        {
            tetMap[tid] = tidx++;
        }
    }
    if(debug)
    {
        std::cout << "newTetNodes newTetFaces newTetLab " << tidx << "\n";
    }
    arma::umat newTetNodes(tidx,4);
    arma::umat newTetFaces(tidx,4);
    arma::uvec newTetLab(tidx);
    for(size_t tid = 0; tid < msh->nTetras ; tid++)
    {
        if(tetFlag[tid])
        {
            newTetNodes.row(tetMap[tid]) = msh->tetNodes.row(tid);
            newTetFaces.row(tetMap[tid]) = msh->tetFaces.row(tid);
            newTetLab(tetMap[tid]) = msh->tetLab(tid);
        }
    }
    msh->nTetras = tidx;
    msh->tetNodes = newTetNodes;
    msh->tetFaces = newTetFaces;
    msh->tetLab = newTetLab;
    if(debug)
    {
        std::cout << "tetMtrl newTetras\n";
    }
    for(size_t mtrid = 0; mtrid < msh->tetMtrl.size(); mtrid++)
    {
        std::vector<size_t> newTetras;
        for(size_t tid = 0; tid< msh->tetMtrl[mtrid].Tetras.size(); tid++)
        {
            if(tetMap[msh->tetMtrl[mtrid].Tetras(tid)] < UINT_MAX)
            {
                newTetras.push_back(tetMap[msh->tetMtrl[mtrid].Tetras(tid)]);
            }
        }
        msh->tetMtrl[mtrid].Tetras.clear();
        msh->tetMtrl[mtrid].Tetras.resize(newTetras.size());
        for(size_t tid = 0; tid< newTetras.size(); tid++)
        {
            msh->tetMtrl[mtrid].Tetras(tid) = newTetras[tid];
        }
    }
    // reorder nodes and faces
    if(debug)
    {
        std::cout << "nodMap " << msh->nNodes << " facMap " << msh->nFaces << "\n";
    }
    std::vector<size_t> nodMap(msh->nNodes,UINT_MAX);
    std::vector<size_t> facMap(msh->nFaces,UINT_MAX);
    for(size_t tid = 0; tid < msh->nTetras ; tid++)
    {
        for(size_t i=0; i<4; i++)
        {
            if(nodFlag[msh->tetNodes(tid,i)] == false)
            {
                nodFlag[msh->tetNodes(tid,i)] = true;
                nodMap[msh->tetNodes(tid,i)] = nidx++;
            }
            if(facFlag[msh->tetFaces(tid,i)] == false)
            {
                facFlag[msh->tetFaces(tid,i)] = true;
                facMap[msh->tetFaces(tid,i)] = fidx++;
            }
        }
    }
    // finishing with nodes
    if(debug)
    {
        std::cout << "tetNodes tetFaces\n";
    }
    for(size_t tid = 0; tid < msh->nTetras ; tid++)
    {
        for(size_t i=0; i<4; i++)
        {
            msh->tetNodes(tid,i) = nodMap[msh->tetNodes(tid,i)];
            msh->tetFaces(tid,i) = facMap[msh->tetFaces(tid,i)];
        }
        msh->tetNodes.row(tid) = arma::sort(msh->tetNodes.row(tid));
    }
    if(debug)
    {
        std::cout << "facNodes\n";
    }
    for(size_t fid = 0; fid < msh->nFaces ; fid++)
    {
        for(size_t i=0; i<3; i++)
        {
            msh->facNodes(fid,i) = nodMap[msh->facNodes(fid,i)];
        }
        msh->facNodes.row(fid) = arma::sort(msh->facNodes.row(fid));
    }
    if(debug)
    {
        std::cout << "newNodPos " << nidx << "\n";
    }
    arma::mat newNodPos(nidx,3);
    for(size_t nid = 0; nid < msh->nNodes; nid++)
    {
        if(nodFlag[nid])
        {
            newNodPos.row(nodMap[nid]) = msh->nodPos.row(nid);
        }
    }
    if(debug)
    {
        std::cout << "newFacNodes " << fidx << "\n";
    }
    arma::umat newFacNodes(fidx,3);
    for(size_t fid = 0; fid < msh->nFaces; fid++)
    {
        if(facFlag[fid])
        {
            newFacNodes.row(facMap[fid]) = arma::sort(msh->facNodes.row(fid));
        }
    }
    msh->nodPos = newNodPos;
    msh->facNodes = newFacNodes;
    msh->nNodes = nidx;
    msh->nFaces = fidx;
    // remain to assign face labels
    // reorder faces in tetrahedron
    // create edges
    if(debug)
    {
        std::cout << "facLab\n";
    }
    msh->facLab.resize(msh->nFaces);
    msh->facLab.fill(msh->maxLab); // non boundary flag
    for(size_t fid = 0; fid < facHFSStag.size(); fid++)
    {
        for(std::vector<HFSSBnd>::iterator it = bnds.begin(); it != bnds.end(); it++)
        {
            if(it->faces.size())
            {
                for(std::vector<size_t>::iterator itids = it->faces.begin(); itids != it->faces.end(); itids++)
                {
                    if(std::find(facHFSStag[fid].begin(),facHFSStag[fid].end(), *itids) != facHFSStag[fid].end())
                    {
                        for(std::vector<BC>::iterator bndit = msh->facBC.begin();
                                bndit != msh->facBC.end(); bndit++)
                        {
                            if(bndit->name == it->name)
                            {
                                if(debug)
                                {
                                    std::cout << " " << it->name << " " << facMap[fid] << " ";
                                }
                                msh->facLab(facMap[fid]) = bndit->label;
                                arma::uvec nface(1);
                                nface(0) = facMap[fid];
                                bndit->Faces = arma::join_cols(bndit->Faces, nface);
                            }
                        }
                    }
                }
            }
            // solid based boundaries
            if(it->solids.size())
            {
                for(std::vector<size_t>::iterator itids = it->solids.begin(); itids != it->solids.end(); itids++)
                {
                    std::vector<size_t> cid = bndMap[*itids];
                    for(size_t idd = 0; idd<cid.size(); idd++)
                    {
                        if(std::find(facHFSStag[fid].begin(),facHFSStag[fid].end(), cid[idd]) != facHFSStag[fid].end())
                        {
                            for(std::vector<BC>::iterator bndit = msh->facBC.begin();
                                    bndit != msh->facBC.end(); bndit++)
                            {
                                if(bndit->name == it->name)
                                {
                                    if(debug)
                                    {
                                        std::cout << " " << it->name << " " << facMap[fid] << " ";
                                    }
                                    msh->facLab(facMap[fid]) = bndit->label;
                                    arma::uvec nface(1);
                                    nface(0) = facMap[fid];
                                    bndit->Faces = arma::join_cols(bndit->Faces, nface);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    /// reorder faces and create edges
    if(debug)
    {
        std::cout << "tetFaces facAdjTet\n";
    }
    std::vector<size_t> nodes(4), faces(4);
    fidx = 0;
    for(size_t tit = 0; tit < msh->nTetras; tit++)
    {
        nodes[0] = msh->tetNodes(tit,0);
        nodes[1] = msh->tetNodes(tit,1);
        nodes[2] = msh->tetNodes(tit,2);
        nodes[3] = msh->tetNodes(tit,3);
        faces[0] = msh->tetFaces(tit,0);
        faces[1] = msh->tetFaces(tit,1);
        faces[2] = msh->tetFaces(tit,2);
        faces[3] = msh->tetFaces(tit,3);
        for(size_t ii = 0; ii<4; ii++)
        {
            size_t fid = faces[ii];
            if(nodes[1] == msh->facNodes(fid,0) &
                    nodes[2] == msh->facNodes(fid,1) &
                    nodes[3] == msh->facNodes(fid,2))
            {
                msh->tetFaces(tit,0) = fid;
            }
            else if(nodes[0] == msh->facNodes(fid,0) &
                    nodes[2] == msh->facNodes(fid,1) &
                    nodes[3] == msh->facNodes(fid,2))
            {
                msh->tetFaces(tit,1) = fid;
            }
            else if(nodes[0] == msh->facNodes(fid,0) &
                    nodes[1] == msh->facNodes(fid,1) &
                    nodes[3] == msh->facNodes(fid,2))
            {
                msh->tetFaces(tit,2) = fid;
            }
            else if(nodes[0] == msh->facNodes(fid,0) &
                    nodes[1] == msh->facNodes(fid,1) &
                    nodes[2] == msh->facNodes(fid,2))
            {
                msh->tetFaces(tit,3) = fid;
            }
            arma::uvec adjTetId(1);
            adjTetId(0) = tit;
            msh->facAdjTet(fid) = arma::join_cols(msh->facAdjTet(fid), adjTetId);
        }
    }
    /// create edges
    if(debug)
    {
        std::cout << "edgesMap\n";
    }
    idx = 0;
    size_t n0, n1;
    std::map<std::pair<size_t,size_t>, size_t> edgesMap;
    for(size_t tit = 0; tit < msh->nTetras; tit++)
    {
        n0 = msh->facNodes(msh->tetFaces(tit,2),1);
        n1 = msh->facNodes(msh->tetFaces(tit,2),2);
        if(edgesMap.find(std::make_pair(n0,n1)) == edgesMap.end())
        {
            edgesMap[std::make_pair(n0, n1)] = idx++;
        }
        n0 = msh->facNodes(msh->tetFaces(tit,1),0);
        n1 = msh->facNodes(msh->tetFaces(tit,1),2);
        if(edgesMap.find(std::make_pair(n0,n1)) == edgesMap.end())
        {
            edgesMap[std::make_pair(n0, n1)] = idx++;
        }
        n0 = msh->facNodes(msh->tetFaces(tit,1),1);
        n1 = msh->facNodes(msh->tetFaces(tit,1),2);
        if(edgesMap.find(std::make_pair(n0,n1)) == edgesMap.end())
        {
            edgesMap[std::make_pair(n0, n1)] = idx++;
        }
        n0 = msh->facNodes(msh->tetFaces(tit,0),0);
        n1 = msh->facNodes(msh->tetFaces(tit,0),1);
        if(edgesMap.find(std::make_pair(n0,n1)) == edgesMap.end())
        {
            edgesMap[std::make_pair(n0, n1)] = idx++;
        }
        n0 = msh->facNodes(msh->tetFaces(tit,1),0);
        n1 = msh->facNodes(msh->tetFaces(tit,1),1);
        if(edgesMap.find(std::make_pair(n0,n1)) == edgesMap.end())
        {
            edgesMap[std::make_pair(n0, n1)] = idx++;
        }
        n0 = msh->facNodes(msh->tetFaces(tit,2),0);
        n1 = msh->facNodes(msh->tetFaces(tit,2),1);
        if(edgesMap.find(std::make_pair(n0,n1)) == edgesMap.end())
        {
            edgesMap[std::make_pair(n0, n1)] = idx++;
        }
    }
    msh->nEdges = edgesMap.size();
    if(debug)
    {
        std::cout << "edgNodes\n";
    }
    std::map<std::pair<size_t,size_t>, size_t>::iterator emIter;
    msh->edgNodes.resize(msh->nEdges,2);
    for(emIter = edgesMap.begin(); emIter != edgesMap.end(); emIter++)
    {
        msh->edgNodes(emIter->second,0) =  emIter->first.first;
        msh->edgNodes(emIter->second,1) =  emIter->first.second;
    }
    idx=0;
    if(debug)
    {
        std::cout << "facEdges\n";
    }
    msh->facEdges.resize(msh->nFaces,3);
    for(size_t fit = 0; fit < msh->nFaces; fit++)
    {
        n0 = msh->facNodes(fit,1);
        n1 = msh->facNodes(fit,2);
        msh->facEdges(fit,0) = edgesMap[std::make_pair(n0, n1)];
        n0 = msh->facNodes(fit,0);
        n1 = msh->facNodes(fit,2);
        msh->facEdges(fit,1) = edgesMap[std::make_pair(n0, n1)];
        n0 = msh->facNodes(fit,0);
        n1 = msh->facNodes(fit,1);
        msh->facEdges(fit,2) = edgesMap[std::make_pair(n0, n1)];
    }
    if(debug)
    {
        std::cout << "facEdges tetEdges\n";
    }
    for(size_t tit = 0; tit < msh->nTetras; tit++)
    {
        msh->tetEdges(tit,0) = msh->facEdges(msh->tetFaces(tit,2),2);
        msh->tetEdges(tit,1) = msh->facEdges(msh->tetFaces(tit,1),2);
        msh->tetEdges(tit,2) = msh->facEdges(msh->tetFaces(tit,1),1);
        msh->tetEdges(tit,3) = msh->facEdges(msh->tetFaces(tit,0),2);
        msh->tetEdges(tit,4) = msh->facEdges(msh->tetFaces(tit,0),1);
        msh->tetEdges(tit,5) = msh->facEdges(msh->tetFaces(tit,0),0);
    }
}

