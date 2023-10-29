#include "Mesh.h"
#include "Project.h"
//#include "Config.h"
#include <iomanip>

extern "C" {
#include <metis.h>
}
#include <stdlib.h>     /* malloc, free, rand */

Mesh::Mesh()
{
}

Mesh::~Mesh()
{
    Clear();
}

void Mesh::Clear()
{
    tetNodes.clear();
    tetEdges.clear();
    tetFaces.clear();
    facNodes.clear();
    facEdges.clear();
    edgNodes.clear();
    nodPos.clear();
    facLab.clear();
    tetLab.clear();
    tetDom.clear();
    facOnRadBnd.clear();
    facAdjTet.clear();
    edgAdjFac.clear();
    domTetras.clear();
    domFaces.clear();
    facBC.clear();
}

void Mesh::RefineHomogeneous()
{
}

void Mesh::SaveField(std::string FieldName)
{
    std::ofstream outField(std::string(FieldName + "_Mesh.vtk").data());
    outField << "# vtk DataFile Version 2.0\n";
    outField << "Mesh data\n";
    outField << "ASCII\n";
    outField << "DATASET UNSTRUCTURED_GRID\n";
    outField << "POINTS " << nNodes << " double \n";
    for(size_t i = 0; i < nNodes; i++)
    {
        outField << std::setprecision(16) << nodPos(i,0) << " ";
        outField << std::setprecision(16) << nodPos(i,1) << " ";
        outField << std::setprecision(16) << nodPos(i,2) << "\n";
    }
    outField << "CELLS " << nTetras << " " << 5*nTetras << "\n";
    for(size_t i = 0; i < nTetras; i++)
    {
        outField << 4 << " ";
        outField << tetNodes(i,0) << " ";
        outField << tetNodes(i,1) << " ";
        outField << tetNodes(i,2) << " ";
        outField << tetNodes(i,3) << "\n";
    }
    outField << "CELL_TYPES " << nTetras << "\n";
    for(size_t i = 0; i < nTetras; i++)
    {
        outField << 10 << "\n";
    }
    outField.close();
}

void Mesh::PartitionMesh(int nparts)
{
    std::cout << "Mesh partitioning";
    arma::wall_clock mt;
    mt.tic();
    nDomains = nparts;
    int ne = (int) nTetras;
    int nn = (int) nNodes;
    idxtype* elmnts = (idxtype*) malloc(4*ne*sizeof(idxtype));
    for(size_t i=0; i < nTetras; i++)
    {
        for(size_t j=0; j<4; j++)
        {
            elmnts[4*i+j] = tetNodes(i,j);
        }
    }
    int etype = 2; // tetrahedra
    int numflag = 0; // numbering starts from 0
    int edgecut; // number of cut edges
    idxtype* epart = (idxtype*) malloc(ne*sizeof(idxtype));
    idxtype* npart = (idxtype*) malloc(nn*sizeof(idxtype));
    // int METIS PartMeshNodal( idx_t *ne, idx_t *nn, idx_t *eptr, idx_t *eind, idx_t *vwgt, idx_t *vsize,
    //idx_t *nparts, real t *tpwgts, idx_t *options, idx_t *objval, idx_t *epart, idx_t *npart)
    if(nparts > 1)
    {
        METIS_PartMeshNodal(&ne, &nn, elmnts, &etype, &numflag, &nparts, &edgecut, epart, npart);
    }
    else
    {
        for(size_t i=0; i< ne; i++)
        {
            epart[i] = 0;
        }
    }
    std::cout << ".";
    tetDom.resize(nTetras);
    for(size_t i=0; i< ne; i++)
    {
        tetDom(i) = epart[i];
    }
    delete epart;
    delete npart;
    domTetras.set_size(nparts);
    domFaces.set_size(nparts);
    std::vector<size_t> domcnt(nparts,0);
    #pragma omp parallel for
    for(size_t tit = 0; tit < nTetras; tit++)
    {
        #pragma omp critical
        {
            ++domcnt[tetDom(tit)];
        }
    }
    for(size_t did = 0; did < nparts; did++)
    {
        domTetras(did).resize(domcnt[did]);
        //std::cout << domcnt[did] << " ";
        domcnt[did] = 0;
    }
    #pragma omp parallel for
    for(size_t tit = 0; tit < nTetras; tit++)
    {
        size_t dom = tetDom(tit);
        #pragma omp critical
        {
            domTetras(dom)(domcnt[dom]++) = tit;
        }
//        arma::uvec tid(1);
//        tid(0) = tit;
//        #pragma omp critical
//        {
//            domTetras(tetDom(tit)) = arma::join_cols(domTetras(tetDom(tit)), tid);
//        }
    }
    std::cout << ".";
    #pragma omp parallel for
    for(size_t fif = 0; fif < nFaces; fif++)
    {
        if(facAdjTet(fif).n_rows > 1)
        {
            arma::uvec fid(1);
            fid(0) = fif;
            arma::uvec adjTet = facAdjTet(fif);
            if(tetDom(adjTet(0)) != tetDom(adjTet(1)))
            {
                #pragma omp critical
                {
                    domFaces(tetDom(adjTet(0))) = arma::join_cols(domFaces(tetDom(adjTet(0))), fid);
                    domFaces(tetDom(adjTet(1))) = arma::join_cols(domFaces(tetDom(adjTet(1))), fid);
                }
            }
        }
    }
    std::cout << ".";
    std::cout << " " << mt.toc() << " s, ";
    std::ofstream outField(std::string("Partitions_Mesh.vtk").data());
    outField << "# vtk DataFile Version 2.0\n";
    outField << "Mesh data\n";
    outField << "ASCII\n";
    outField << "DATASET UNSTRUCTURED_GRID\n";
    outField << "POINTS " << nNodes << " double \n";
    for(size_t i = 0; i < nNodes; i++)
    {
        outField << std::setprecision(16) << nodPos(i,0) << " ";
        outField << std::setprecision(16) << nodPos(i,1) << " ";
        outField << std::setprecision(16) << nodPos(i,2) << "\n";
    }
    outField << "CELLS " << nTetras << " " << 5*nTetras << "\n";
    for(size_t i = 0; i < nTetras; i++)
    {
        outField << 4 << " ";
        outField << tetNodes(i,0) << " ";
        outField << tetNodes(i,1) << " ";
        outField << tetNodes(i,2) << " ";
        outField << tetNodes(i,3) << "\n";
    }
    outField << "CELL_TYPES " << nTetras << "\n";
    for(size_t i = 0; i < nTetras; i++)
    {
        outField << 10 << "\n";
    }
    outField << "CELL_DATA " << nTetras  << "\n";
    outField << "SCALARS Domain float" << "\n";
    outField << "LOOKUP_TABLE default" << "\n";
    for(size_t i = 0; i < nTetras; i++)
    {
        outField << (float) tetDom(i) << "\n";
    }
    outField.close();
}

void Mesh::Reorder()
{
    std::cout << "Metis reording\n";
    int ne = (int) nTetras;
    int ns = (int) nEdges;
    int nn = (int) nNodes;
    idxtype* elmnts = (idxtype*) malloc(4*ne*sizeof(idxtype));
    for(size_t i=0; i < nTetras; i++)
    {
        for(size_t j=0; j<4; j++)
        {
            elmnts[4*i+j] = tetNodes(i,j);
        }
    }
    int etype = 2; // tetrahedra
    int numflag = 0; // numbering starts from 0
    int edgecut; // number of cut edges
    idxtype* xadj = (idxtype*) malloc((nn+1)*sizeof(idxtype));
    idxtype* adjncy = (idxtype*) malloc((2*ns)*sizeof(idxtype));
    // int METIS_MeshToNodal(idx_t *ne, idx_t *nn, idx_t *eptr, idx_t *eind, idx_t *numflag, idx_t **xadj, idx_t **adjncy)
    // (&ne, &nn, elmnts, &etype, &numflag, &nparts, &edgecut, epart, npart);
    METIS_MeshToNodal(&ne, &nn, elmnts, &etype, &numflag, xadj, adjncy);
    //int METIS_NodeND(idx_t *nvtxs, idx_t *xadj, idx_t *adjncy, idx_t *vwgt, idx_t *options, idx_t *perm, idx_t *iperm)
    int vwgt = 0;
    int options[] = {0,0,15};
    idxtype* perm = (idxtype*) malloc((nn)*sizeof(idxtype));
    idxtype* iperm = (idxtype*) malloc((nn)*sizeof(idxtype));
    METIS_NodeND(&nn, xadj, adjncy, &vwgt, options, perm, iperm);
    int maxperm = 0;
//    for(size_t i=0; i< nn; i++) {
//        maxperm = std::max(maxperm,iperm[i]);
//        //std::cout << perm[i]  << " " << iperm[i] << "\t";
//    }
    //std::cout << maxperm << "\n";
    /// reconstruct mesh
    arma::mat newNodPos(nNodes,3);
    for(size_t nid = 0; nid < nNodes; nid++)
    {
        newNodPos.row(nid) = nodPos.row(perm[nid]);
    }
    nodPos = newNodPos;
    arma::umat newTetNodes(nTetras,4);
    for(size_t tit = 0; tit < nTetras; tit++)
    {
        for(size_t nid=0; nid< 4; nid++)
        {
            newTetNodes(tit,nid) = (size_t)iperm[tetNodes(tit,nid)];
        }
        newTetNodes.row(tit) = arma::sort(newTetNodes.row(tit));
    }
    tetNodes = newTetNodes;
    arma::umat newFacNodes(nFaces,3);
    for(size_t tit = 0; tit < nFaces; tit++)
    {
        for(size_t nid=0; nid< 3; nid++)
        {
            newFacNodes(tit,nid) = iperm[facNodes(tit,nid)];
        }
        newFacNodes.row(tit) = arma::sort(newFacNodes.row(tit));
    }
    facNodes = newFacNodes;
    arma::umat newEdgNodes(nEdges,2);
    for(size_t tit = 0; tit < nEdges; tit++)
    {
        for(size_t nid=0; nid< 2; nid++)
        {
            newEdgNodes(tit,nid) = iperm[edgNodes(tit,nid)];
        }
        newEdgNodes.row(tit) = arma::sort(newEdgNodes.row(tit));
    }
    edgNodes = newEdgNodes;
    delete perm;
    delete iperm;
}


arma::mat Mesh::tetGeo(size_t id)
{
    arma::mat cGeo(4,3);
    cGeo.row(0) = nodPos.row(tetNodes(id,0));
    cGeo.row(1) = nodPos.row(tetNodes(id,1));
    cGeo.row(2) = nodPos.row(tetNodes(id,2));
    cGeo.row(3) = nodPos.row(tetNodes(id,3));
    return cGeo;
}

arma::mat Mesh::facGeo(size_t id)
{
    arma::mat cGeo(3,3);
    cGeo.row(0) = nodPos.row(facNodes(id,0));
    cGeo.row(1) = nodPos.row(facNodes(id,1));
    cGeo.row(2) = nodPos.row(facNodes(id,2));
    return cGeo;
}

arma::mat Mesh::facGeo2(size_t id)
{
    arma::mat cGeo(3,3);
    cGeo.row(0) = nodPos.row(facNodes(id,0));
    cGeo.row(1) = nodPos.row(facNodes(id,1));
    cGeo.row(2) = nodPos.row(facNodes(id,2));
    arma::vec v0 = cGeo(0,arma::span::all).st();
    arma::vec v1 = cGeo(1,arma::span::all).st();
    arma::vec v2 = cGeo(2,arma::span::all).st();
    v1 -= v0;
    v2 -= v0;
    arma::vec u = v1 / arma::norm(v1,2);
    arma::vec n = arma::cross(v1,v2);
    n /= arma::norm(n,2);
    arma::vec v = arma::cross(n,u);
    arma::mat cGeo2(3,2);
    cGeo2.fill(0);
    cGeo2(1,0) = arma::dot((cGeo.row(1)-cGeo.row(0)).st(), u);
    cGeo2(2,0) = arma::dot((cGeo.row(2)-cGeo.row(0)).st(), u);
    cGeo2(2,1) = arma::dot((cGeo.row(2)-cGeo.row(0)).st(), v);
    return cGeo2;
}

arma::vec Mesh::intNode(size_t id)
{
    arma::vec nod(3);
    arma::uvec nfac = facNodes.row(id).st();
    arma::uvec ntet = tetNodes.row(facAdjTet(id)(0)).st();
    size_t intid;
    for(size_t i = 0; i<4; i++)
    {
        bool found = true;
        intid = ntet(i);
        for(size_t j = 0; j<3; j++)
            if(ntet(i) == nfac(j))
            {
                found = false;
            }
        if(found)
        {
            break;
        }
    }
    nod = nodPos.row(intid).st();
    return nod;
}

arma::vec Mesh::intNode(size_t id, size_t& RefFace)
{
    arma::vec nod(3);
    arma::uvec nfac = facNodes.row(id).st();
    arma::uvec ntet = tetNodes.row(facAdjTet(id)(RefFace)).st();
    size_t intid = 0;
    for(size_t i = 0; i<4; i++)
    {
        bool found = true;
        intid = ntet(i);
        for(size_t j = 0; j<3; j++)
            if(ntet(i) == nfac(j))
            {
                found = false;
            }
        if(found)
        {
            RefFace = i;
            break;
        }
    }
    nod = nodPos.row(intid).st();
    return nod;
}
