#include "TetGen.h"
#include <map>

TetGen::TetGen(Project* cProj): prj(cProj), dbg(cProj->opt->dbg), scaling(1.0)
{
    if(prj->opt->href)
    {
        std::string polyCmd(prj->opt->hrefCmd + "AfeerQ");
        char* name = new char[prj->opt->name.size() + 1];
        strcpy(name, prj->opt->name.c_str());
        char* switches = new char[polyCmd.size() + 1];
        strcpy(switches, polyCmd.c_str());
        CopyOldMesh();
        std::cout << "h-refinement with TetGen command = " << polyCmd << "\n";
        tetrahedralize(switches, &in, &out, &addin, &bgmin);
        CopyNewMesh();
    }
    else
    {
        std::string polyCmd("p" + prj->opt->polyCmd + "AfeeQ");
        char* name = new char[prj->opt->name.size() + 1];
        strcpy(name, prj->opt->name.c_str());
        char* switches = new char[polyCmd.size() + 1];
        strcpy(switches, polyCmd.c_str());
        in.load_poly(name);
        LoadExtra();
        //std::cout << "Switches = " << polyCmd << "\n";
        //std::cout << "Scaling = " << scaling <<  "\n";
        std::cout << "Meshing with TetGen command = " << polyCmd << "\n";
        tetrahedralize(switches, &in, &out, &addin, &bgmin);
        CreateMesh();
    }
}

TetGen::~TetGen()
{
}

void TetGen::CreateMesh()
{
    size_t n0, n1, n2;
    // Nodes
    prj->msh->nNodes = out.numberofpoints;
    prj->msh->nodPos.resize(prj->msh->nNodes,3);
    prj->msh->nodPos.fill(0);
    if(dbg)
    {
        std::cout << "nNodes  = " << prj->msh->nNodes << "\n";
    }
    for(size_t i=0; i<prj->msh->nNodes; i++)
    {
        for(size_t j=0; j<3; j++)
        {
            prj->msh->nodPos(i,j) = out.pointlist[i*3+j];
        }
    }
    //prj->msh->nodPos *= scaling;
    // Edges
    std::map<std::pair<size_t,size_t>, size_t> edgesMap;
    prj->msh->nEdges = out.numberofedges;
    prj->msh->edgNodes.resize(prj->msh->nEdges,2);
    prj->msh->edgNodes.fill(0);
    if(dbg)
    {
        std::cout << "nEdges  = " << prj->msh->nEdges << "\n";
    }
    for(size_t i=0; i<prj->msh->nEdges; i++)
    {
        for(size_t j=0; j<2; j++)
        {
            prj->msh->edgNodes(i,j) = out.edgelist[i*2+j]-1;
        }
        prj->msh->edgNodes.row(i) = arma::sort(prj->msh->edgNodes.row(i));
        n0 = prj->msh->edgNodes(i,0);
        n1 = prj->msh->edgNodes(i,1);
        edgesMap[std::make_pair(n0, n1)] = i;
    }
    // Faces
    std::map<std::pair<size_t,std::pair<size_t,size_t> >, size_t> facesMap;
    prj->msh->nFaces = out.numberoftrifaces;
    prj->msh->facNodes.resize(prj->msh->nFaces,3);
    prj->msh->facNodes.fill(0);
    prj->msh->facEdges.resize(prj->msh->nFaces,3);
    prj->msh->facEdges.fill(0);
    prj->msh->facLab.resize(prj->msh->nFaces);
    prj->msh->facLab.fill(prj->msh->maxLab);
    prj->msh->facAdjTet.set_size(prj->msh->nFaces);
    if(dbg)
    {
        std::cout << "nFaces  = " << prj->msh->nFaces << "\n";
    }
    for(size_t i=0; i<prj->msh->nFaces; i++)
    {
        for(size_t j=0; j<3; j++)
        {
            prj->msh->facNodes(i,j) = out.trifacelist[i*3+j]-1;
        }
        prj->msh->facNodes.row(i) = arma::sort(prj->msh->facNodes.row(i));
        n0 = prj->msh->facNodes(i,1);
        n1 = prj->msh->facNodes(i,2);
        prj->msh->facEdges(i,0) = edgesMap[std::make_pair(n0, n1)];
        n0 = prj->msh->facNodes(i,0);
        n1 = prj->msh->facNodes(i,2);
        prj->msh->facEdges(i,1) = edgesMap[std::make_pair(n0, n1)];
        n0 = prj->msh->facNodes(i,0);
        n1 = prj->msh->facNodes(i,1);
        prj->msh->facEdges(i,2) = edgesMap[std::make_pair(n0, n1)];
        //
        n0 = prj->msh->facNodes(i,0);
        n1 = prj->msh->facNodes(i,1);
        n2 = prj->msh->facNodes(i,2);
        facesMap[std::make_pair(n0, std::make_pair(n1,n2))] = i;
        // labels
        prj->msh->facLab(i) = out.trifacemarkerlist[i];
    }
    // Tetras
    prj->msh->nTetras = out.numberoftetrahedra;
    prj->msh->tetNodes.resize(prj->msh->nTetras,4);
    prj->msh->tetNodes.fill(0);
    prj->msh->tetEdges.resize(prj->msh->nTetras,6);
    prj->msh->tetEdges.fill(0);
    prj->msh->tetFaces.resize(prj->msh->nTetras,4);
    prj->msh->tetFaces.fill(0);
    prj->msh->tetLab.resize(prj->msh->nTetras);
    prj->msh->tetLab.fill(prj->msh->maxLab);
    if(dbg)
    {
        std::cout << "nTetras = " << prj->msh->nTetras << "\n";
    }
    for(size_t i=0; i<prj->msh->nTetras; i++)
    {
        for(size_t j=0; j<4; j++)
        {
            prj->msh->tetNodes(i,j) = out.tetrahedronlist[i*4+j]-1;
        }
        prj->msh->tetNodes.row(i) = arma::sort(prj->msh->tetNodes.row(i));
        //
        n0 = prj->msh->tetNodes(i,0);
        n1 = prj->msh->tetNodes(i,1);
        prj->msh->tetEdges(i,0) = edgesMap[std::make_pair(n0, n1)];
        n0 = prj->msh->tetNodes(i,0);
        n1 = prj->msh->tetNodes(i,2);
        prj->msh->tetEdges(i,1) = edgesMap[std::make_pair(n0, n1)];
        n0 = prj->msh->tetNodes(i,0);
        n1 = prj->msh->tetNodes(i,3);
        prj->msh->tetEdges(i,2) = edgesMap[std::make_pair(n0, n1)];
        n0 = prj->msh->tetNodes(i,1);
        n1 = prj->msh->tetNodes(i,2);
        prj->msh->tetEdges(i,3) = edgesMap[std::make_pair(n0, n1)];
        n0 = prj->msh->tetNodes(i,1);
        n1 = prj->msh->tetNodes(i,3);
        prj->msh->tetEdges(i,4) = edgesMap[std::make_pair(n0, n1)];
        n0 = prj->msh->tetNodes(i,2);
        n1 = prj->msh->tetNodes(i,3);
        prj->msh->tetEdges(i,5) = edgesMap[std::make_pair(n0, n1)];
        //
        n0 = prj->msh->tetNodes(i,1);
        n1 = prj->msh->tetNodes(i,2);
        n2 = prj->msh->tetNodes(i,3);
        prj->msh->tetFaces(i,0) = facesMap[std::make_pair(n0, std::make_pair(n1, n2))];
        n0 = prj->msh->tetNodes(i,0);
        n1 = prj->msh->tetNodes(i,2);
        n2 = prj->msh->tetNodes(i,3);
        prj->msh->tetFaces(i,1) = facesMap[std::make_pair(n0, std::make_pair(n1, n2))];
        n0 = prj->msh->tetNodes(i,0);
        n1 = prj->msh->tetNodes(i,1);
        n2 = prj->msh->tetNodes(i,3);
        prj->msh->tetFaces(i,2) = facesMap[std::make_pair(n0, std::make_pair(n1, n2))];
        n0 = prj->msh->tetNodes(i,0);
        n1 = prj->msh->tetNodes(i,1);
        n2 = prj->msh->tetNodes(i,2);
        prj->msh->tetFaces(i,3) = facesMap[std::make_pair(n0, std::make_pair(n1, n2))];
        // labels
        prj->msh->tetLab(i) = out.tetrahedronattributelist[i];
        // adjTets
        arma::uvec adjTetId(1);
        adjTetId(0) = i;
        for(size_t j=0; j<4; j++)
        {
            size_t fid = prj->msh->tetFaces(i,j);
            prj->msh->facAdjTet(fid) = arma::join_cols(prj->msh->facAdjTet(fid), adjTetId);
        }
    }
    for(size_t i = 0; i < prj->msh->facBC.size(); i++)
    {
        size_t cLab = prj->msh->facBC[i].label;
        if(dbg)
        {
            std::cout << prj->msh->facBC[i].name << " " << cLab << "\n";
        }
        for(size_t fid = 0; fid < prj->msh->nFaces; fid++)
        {
            if(cLab == prj->msh->facLab(fid))
            {
                arma::uvec cFace(1);
                cFace(0) = fid;
                prj->msh->facBC[i].Faces = arma::join_cols(prj->msh->facBC[i].Faces, cFace);
            }
        }
    }
    for(size_t i = 0; i < prj->msh->tetMtrl.size(); i++)
    {
        size_t cLab = prj->msh->tetMtrl[i].label;
        if(dbg)
        {
            std::cout << prj->msh->tetMtrl[i].name << " " << cLab << "\n";
        }
        size_t tetcnt = 0;
        for(size_t tid = 0; tid < prj->msh->nTetras; tid++)
        {
            if(cLab == prj->msh->tetLab(tid))
            {
//                arma::uvec cTet(1);
//                cTet(0) = tid;
//                prj->msh->tetMtrl[i].Tetras = arma::join_cols(prj->msh->tetMtrl[i].Tetras, cTet);
                ++tetcnt;
            }
        }
        prj->msh->tetMtrl[i].Tetras.resize(tetcnt);
        tetcnt = 0;
        for(size_t tid = 0; tid < prj->msh->nTetras; tid++)
        {
            if(cLab == prj->msh->tetLab(tid))
            {
                prj->msh->tetMtrl[i].Tetras(tetcnt++) = tid;
            }
        }
    }
}

void TetGen::LoadExtra()
{
    size_t tmpInt;
    double tmpDbl;
    std::string tmpStr;
    std::string line;
    std::ifstream fileName(std::string(prj->opt->name + ".poly").c_str(), std::ios::in);
    if(fileName.is_open())
    {
        while(getline(fileName,line))
        {
            std::istringstream iss(line);
            iss >> tmpStr;
            /*
            if(tmpStr == "#Scaling")
            {
                iss >> scaling;
                tmpStr.clear();
            }
            */
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
                    prj->msh->tetMtrl.push_back(mtr);
                    std::cout << mtr.sldName << " " << mtr.name << "\n";
                }
                tmpStr.clear();
            }
            if(tmpStr == "#Boundaries")
            {
                iss >> tmpInt;
                for(size_t i = 0; i < tmpInt; i++)
                {
                    std::string type;
                    BC bc;
                    getline(fileName,line);
                    std::istringstream iss(line);
                    iss >> bc.name;
                    iss >> bc.label;
                    iss >> type;
                    bc.setType(type);
                    if(bc.type == BC::WavePort)
                    {
                        iss >> bc.numModes;
                    }
                    prj->msh->facBC.push_back(bc);
                    std::cout << bc.name << " " << type << " " << bc.numModes <<"\n";
                }
                tmpStr.clear();
            }
        }
    }
}


void TetGen::CopyOldMesh()
{
    // Nodes
    in.firstnumber = 0;
    in.numberofpoints = prj->msh->nNodes;
    in.pointlist = new REAL[3*in.numberofpoints];
    for(size_t i=0; i<prj->msh->nNodes; i++)
    {
        for(size_t j=0; j<3; j++)
        {
            in.pointlist[i*3+j] = prj->msh->nodPos(i,j);
        }
    }
    // Edges
    in.numberofedges = prj->msh->nEdges;
    in.edgelist = new int[in.numberofedges*2];
    for(size_t i=0; i<prj->msh->nEdges; i++)
    {
        for(size_t j=0; j<2; j++)
        {
            in.edgelist[i*2+j] = prj->msh->edgNodes(i,j);
        }
    }
    // Faces
    in.numberoftrifaces = prj->msh->nFaces;
    in.trifacelist = new int[in.numberoftrifaces*3];
    in.trifacemarkerlist = new int[in.numberoftrifaces];
    for(size_t i=0; i<prj->msh->nFaces; i++)
    {
        for(size_t j=0; j<3; j++)
        {
            in.trifacelist[i*3+j] = prj->msh->facNodes(i,j);
        }
        // labels
        in.trifacemarkerlist[i] = prj->msh->facLab(i)+1;
        //std::cout << in.trifacemarkerlist[i] << " ";
    }
    // Tetras
    in.numberoftetrahedra = prj->msh->nTetras;
    in.tetrahedronlist = new int[in.numberoftetrahedra*4];
    in.numberoftetrahedronattributes = 1;
    in.tetrahedronattributelist = new double[in.numberoftetrahedra];
    for(size_t i=0; i<prj->msh->nTetras; i++)
    {
        for(size_t j=0; j<4; j++)
        {
            in.tetrahedronlist[i*4+j] = prj->msh->tetNodes(i,j);
        }
        // labels
        in.tetrahedronattributelist[i] = prj->msh->tetLab(i)+1;
        //std::cout << in.tetrahedronattributelist[i] << " ";
    }
}

void TetGen::CopyNewMesh()
{
    size_t n0, n1, n2;
    // Nodes
    //std::cout << out.numberoftetrahedronattributes << "\n\n";
    prj->msh->nNodes = out.numberofpoints;
    prj->msh->nodPos.clear();
    prj->msh->nodPos.resize(prj->msh->nNodes,3);
    prj->msh->nodPos.fill(0);
    if(dbg)
    {
        std::cout << "nNodes  = " << prj->msh->nNodes << "\n";
    }
    std::cout << "Nodes ";
    for(size_t i=0; i<prj->msh->nNodes; i++)
    {
        for(size_t j=0; j<3; j++)
        {
            prj->msh->nodPos(i,j) = out.pointlist[i*3+j];
        }
    }
    //prj->msh->nodPos *= scaling;
    //std::cout << "nodes done.\n";
    // Edges
    std::map<std::pair<size_t,size_t>, size_t> edgesMap;
    prj->msh->nEdges = out.numberofedges;
    prj->msh->edgNodes.clear();
    prj->msh->edgNodes.resize(prj->msh->nEdges,2);
    prj->msh->edgNodes.fill(0);
    if(dbg)
    {
        std::cout << "nEdges  = " << prj->msh->nEdges << "\n";
    }
    std::cout << "Edges ";
    for(size_t i=0; i<prj->msh->nEdges; i++)
    {
        for(size_t j=0; j<2; j++)
        {
            prj->msh->edgNodes(i,j) = out.edgelist[i*2+j];
        }
        prj->msh->edgNodes.row(i) = arma::sort(prj->msh->edgNodes.row(i));
        n0 = prj->msh->edgNodes(i,0);
        n1 = prj->msh->edgNodes(i,1);
        edgesMap[std::make_pair(n0, n1)] = i;
    }
    //std::cout << "edges done.\n";
    // Faces
    std::map<std::pair<size_t,std::pair<size_t,size_t> >, size_t> facesMap;
    prj->msh->nFaces = out.numberoftrifaces;
    prj->msh->facNodes.clear();
    prj->msh->facNodes.resize(prj->msh->nFaces,3);
    prj->msh->facNodes.fill(0);
    prj->msh->facEdges.clear();
    prj->msh->facEdges.resize(prj->msh->nFaces,3);
    prj->msh->facEdges.fill(0);
    prj->msh->facLab.clear();
    prj->msh->facLab.resize(prj->msh->nFaces);
    prj->msh->facLab.fill(prj->msh->maxLab);
    prj->msh->facAdjTet.set_size(prj->msh->nFaces);
    if(dbg)
    {
        std::cout << "nFaces  = " << prj->msh->nFaces << "\n";
    }
    std::cout << "Faces ";
    for(size_t i=0; i<prj->msh->nFaces; i++)
    {
        for(size_t j=0; j<3; j++)
        {
            prj->msh->facNodes(i,j) = out.trifacelist[i*3+j];
        }
        prj->msh->facNodes.row(i) = arma::sort(prj->msh->facNodes.row(i));
        n0 = prj->msh->facNodes(i,1);
        n1 = prj->msh->facNodes(i,2);
        prj->msh->facEdges(i,0) = edgesMap[std::make_pair(n0, n1)];
        n0 = prj->msh->facNodes(i,0);
        n1 = prj->msh->facNodes(i,2);
        prj->msh->facEdges(i,1) = edgesMap[std::make_pair(n0, n1)];
        n0 = prj->msh->facNodes(i,0);
        n1 = prj->msh->facNodes(i,1);
        prj->msh->facEdges(i,2) = edgesMap[std::make_pair(n0, n1)];
        //
        n0 = prj->msh->facNodes(i,0);
        n1 = prj->msh->facNodes(i,1);
        n2 = prj->msh->facNodes(i,2);
        facesMap[std::make_pair(n0, std::make_pair(n1,n2))] = i;
        // labels
        //std::cout << out.trifacemarkerlist[i] << "\n";
        //if(out.trifacemarkerlist[i] > -1)
        prj->msh->facLab(i) = out.trifacemarkerlist[i]-1;
        //else
        //prj->msh->facLab(i) = prj->msh->maxLab;
        //std::cout << prj->msh->facLab(i) << " ";
    }
    //std::cout << "faces done.\n";
    // Tetras
    prj->msh->nTetras = out.numberoftetrahedra;
    prj->msh->tetNodes.clear();
    prj->msh->tetNodes.resize(prj->msh->nTetras,4);
    prj->msh->tetNodes.fill(0);
    prj->msh->tetEdges.clear();
    prj->msh->tetEdges.resize(prj->msh->nTetras,6);
    prj->msh->tetEdges.fill(0);
    prj->msh->tetFaces.clear();
    prj->msh->tetFaces.resize(prj->msh->nTetras,4);
    prj->msh->tetFaces.fill(0);
    prj->msh->tetLab.clear();
    prj->msh->tetLab.resize(prj->msh->nTetras);
    prj->msh->tetLab.fill(0);
    if(dbg)
    {
        std::cout << "nTetras = " << prj->msh->nTetras << "\n";
    }
    std::cout << "Tetras ";
    for(size_t i=0; i<prj->msh->nTetras; i++)
    {
        for(size_t j=0; j<4; j++)
        {
            prj->msh->tetNodes(i,j) = out.tetrahedronlist[i*4+j];
        }
        prj->msh->tetNodes.row(i) = arma::sort(prj->msh->tetNodes.row(i));
        //
        n0 = prj->msh->tetNodes(i,0);
        n1 = prj->msh->tetNodes(i,1);
        prj->msh->tetEdges(i,0) = edgesMap[std::make_pair(n0, n1)];
        n0 = prj->msh->tetNodes(i,0);
        n1 = prj->msh->tetNodes(i,2);
        prj->msh->tetEdges(i,1) = edgesMap[std::make_pair(n0, n1)];
        n0 = prj->msh->tetNodes(i,0);
        n1 = prj->msh->tetNodes(i,3);
        prj->msh->tetEdges(i,2) = edgesMap[std::make_pair(n0, n1)];
        n0 = prj->msh->tetNodes(i,1);
        n1 = prj->msh->tetNodes(i,2);
        prj->msh->tetEdges(i,3) = edgesMap[std::make_pair(n0, n1)];
        n0 = prj->msh->tetNodes(i,1);
        n1 = prj->msh->tetNodes(i,3);
        prj->msh->tetEdges(i,4) = edgesMap[std::make_pair(n0, n1)];
        n0 = prj->msh->tetNodes(i,2);
        n1 = prj->msh->tetNodes(i,3);
        prj->msh->tetEdges(i,5) = edgesMap[std::make_pair(n0, n1)];
        //
        n0 = prj->msh->tetNodes(i,1);
        n1 = prj->msh->tetNodes(i,2);
        n2 = prj->msh->tetNodes(i,3);
        prj->msh->tetFaces(i,0) = facesMap[std::make_pair(n0, std::make_pair(n1, n2))];
        n0 = prj->msh->tetNodes(i,0);
        n1 = prj->msh->tetNodes(i,2);
        n2 = prj->msh->tetNodes(i,3);
        prj->msh->tetFaces(i,1) = facesMap[std::make_pair(n0, std::make_pair(n1, n2))];
        n0 = prj->msh->tetNodes(i,0);
        n1 = prj->msh->tetNodes(i,1);
        n2 = prj->msh->tetNodes(i,3);
        prj->msh->tetFaces(i,2) = facesMap[std::make_pair(n0, std::make_pair(n1, n2))];
        n0 = prj->msh->tetNodes(i,0);
        n1 = prj->msh->tetNodes(i,1);
        n2 = prj->msh->tetNodes(i,2);
        prj->msh->tetFaces(i,3) = facesMap[std::make_pair(n0, std::make_pair(n1, n2))];
        // labels
        if(out.tetrahedronattributelist != NULL)
        {
            prj->msh->tetLab(i) = out.tetrahedronattributelist[i]-1;
            //std::cout << prj->msh->tetLab(i) << "\n";
        }
        // adjTets
        arma::uvec adjTetId(1);
        adjTetId(0) = i;
        for(size_t j=0; j<4; j++)
        {
            size_t fid = prj->msh->tetFaces(i,j);
            prj->msh->facAdjTet(fid) = arma::join_cols(prj->msh->facAdjTet(fid), adjTetId);
        }
    }
    //std::cout << "tets done.\n";
    if(dbg)
    {
        std::cout << "nFacBC = " << prj->msh->facBC.size() << "\n";
    }
    std::cout << "BC ";
    for(size_t i = 0; i < prj->msh->facBC.size(); i++)
    {
        prj->msh->facBC[i].Faces.clear();
        size_t cLab = prj->msh->facBC[i].label;
        //std::cout << prj->msh->facBC[i].name << " " << cLab << "\n";
        for(size_t fid = 0; fid < prj->msh->nFaces; fid++)
        {
            if(cLab == prj->msh->facLab(fid))
            {
                arma::uvec cFace(1);
                cFace(0) = fid;
                prj->msh->facBC[i].Faces = arma::join_cols(prj->msh->facBC[i].Faces, cFace);
            }
        }
    }
    if(dbg)
    {
        std::cout << "nTetMtrl = " << prj->msh->tetMtrl.size() << "\n";
    }
    std::cout << "Mtrl";
    std::vector<size_t> MtrlTets(prj->msh->tetMtrl.size(),0), LabMap(prj->msh->tetMtrl.size(),0);
    for(size_t tid = 0; tid < prj->msh->nTetras; tid++)
    {
        MtrlTets[prj->msh->tetLab(tid)]++;
    }
    for(size_t i = 0; i < prj->msh->tetMtrl.size(); i++)
    {
        prj->msh->tetMtrl[i].Tetras.resize(MtrlTets[i]);
        MtrlTets[i] = 0;
    }
    for(size_t tid = 0; tid < prj->msh->nTetras; tid++)
    {
        size_t cLab = prj->msh->tetLab(tid);
        prj->msh->tetMtrl[cLab].Tetras[MtrlTets[cLab]++] = tid;
    }
    std::cout << "\n";
}
