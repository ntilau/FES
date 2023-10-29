#include "DoF.h"
#include <string>

DoF::DoF(Project* prj, size_t dim, size_t id)
{
    switch(dim)
    {
    case 3: // TETRAHEDRON
        switch(prj->opt->pOrd)
        {
        case 1:
            v.resize(6);
            s.resize(4);
            v = prj->msh->tetEdges.row(id).st();
            s = prj->msh->tetNodes.row(id).st();
            break;
        case 2:
            v.resize(20);
            s.resize(10);
            s.rows(0,3) = prj->msh->tetNodes.row(id).st();
            s.rows(4,9) = prj->msh->nNodes + prj->msh->tetEdges.row(id).st();
            v.rows(0,5) = prj->msh->tetEdges.row(id).st();
            v.rows(6,11) = prj->msh->nEdges + prj->msh->tetEdges.row(id).st();
            for(size_t i=0; i<4; i++)
            {
                v(12+2*i) = 2*prj->msh->nEdges + prj->msh->tetFaces(id,i);
                v(13+2*i) = 2*prj->msh->nEdges +
                            prj->msh->nFaces +
                            prj->msh->tetFaces(id,i);
            }
            break;
        case 3:
            v.resize(45);
            s.resize(20);
            s.rows(0,3) = prj->msh->tetNodes.row(id).st();
            s.rows(4,9) = prj->msh->nNodes + prj->msh->tetEdges.row(id).st();
            s.rows(10,15) = prj->msh->nNodes + prj->msh->nEdges + prj->msh->tetEdges.row(id).st();
            s.rows(16,19) = prj->msh->nNodes + 2*prj->msh->nEdges + prj->msh->tetFaces.row(id).st();
            v.rows(0,5) = prj->msh->tetEdges.row(id).st();
            v.rows(6,11) = prj->msh->nEdges + prj->msh->tetEdges.row(id).st();
            for(size_t i=0; i<4; i++)
            {
                v(12+2*i) = 2*prj->msh->nEdges + prj->msh->tetFaces(id,i);
                v(13+2*i) = 2*prj->msh->nEdges + prj->msh->nFaces + prj->msh->tetFaces(id,i);
                v(26+4*i) = 3*prj->msh->nEdges + 2*prj->msh->nFaces + prj->msh->tetFaces(id,i);
                v(27+4*i) = 3*prj->msh->nEdges + 3*prj->msh->nFaces + prj->msh->tetFaces(id,i);
                v(28+4*i) = 3*prj->msh->nEdges + 4*prj->msh->nFaces + prj->msh->tetFaces(id,i);
                v(29+4*i) = 3*prj->msh->nEdges + 5*prj->msh->nFaces + prj->msh->tetFaces(id,i);
            }
            v.rows(20,25) = 2*prj->msh->nEdges + 2*prj->msh->nFaces + prj->msh->tetEdges.row(id).st();
            v(42) = 3*prj->msh->nEdges + 6*prj->msh->nFaces + id;
            v(43) = 3*prj->msh->nEdges + 6*prj->msh->nFaces + prj->msh->nTetras + id;
            v(44) = 3*prj->msh->nEdges + 6*prj->msh->nFaces + 2*prj->msh->nTetras + id;
            break;
        default:
            throw std::string("3D DoF mapping order not yet implemented");
        }
        break;
    case 2: // TRIANGLE
        switch(prj->opt->pOrd)
        {
        case 1:
            v.resize(3);
            s.resize(3);
            v = prj->msh->facEdges.row(id).st();
            s = prj->msh->facNodes.row(id).st();
            break;
        case 2:
            s.resize(6);
            v.resize(8);
            s.rows(0,2) = prj->msh->facNodes.row(id).st();
            s.rows(3,5) = prj->msh->nNodes + prj->msh->facEdges.row(id).st();
            v.rows(0,2) = prj->msh->facEdges.row(id).st();
            v.rows(3,5) = prj->msh->nEdges + prj->msh->facEdges.row(id).st();
            v(6) = 2*prj->msh->nEdges + id;
            v(7) = 2*prj->msh->nEdges + prj->msh->nFaces + id;
            break;
        case 3:
            s.resize(10);
            v.resize(15);
            s.rows(0,2) = prj->msh->facNodes.row(id).st();
            s.rows(3,5) = prj->msh->nNodes + prj->msh->facEdges.row(id).st();
            s.rows(6,8) = prj->msh->nNodes + prj->msh->nEdges + prj->msh->facEdges.row(id).st();
            s(9) = prj->msh->nNodes + 2*prj->msh->nEdges + id;
            v.rows(0,2) = prj->msh->facEdges.row(id).st();
            v.rows(3,5) = prj->msh->nEdges + prj->msh->facEdges.row(id).st();
            v(6) = 2*prj->msh->nEdges + id;
            v(7) = 2*prj->msh->nEdges + prj->msh->nFaces + id;
            v.rows(8,10) = 2*prj->msh->nEdges + 2*prj->msh->nFaces + prj->msh->facEdges.row(id).st();
            v(11) = 3*prj->msh->nEdges + 2*prj->msh->nFaces + id;
            v(12) = 3*prj->msh->nEdges + 3*prj->msh->nFaces + id;
            v(13) = 3*prj->msh->nEdges + 4*prj->msh->nFaces + id;
            v(14) = 3*prj->msh->nEdges + 5*prj->msh->nFaces + id;
            break;
        default:
            throw std::string("2D DoF mapping order not yet implemented");
        }
        break;
    default:
        throw std::string("dim error in DoF()");
    }
}

DoF::DoF(Project* prj)
{
    switch(prj->opt->pOrd)
    {
    case 1:
        DoFnumv = prj->msh->nEdges;
        DoFnums = prj->msh->nNodes;
        break;
    case 2:
        DoFnumv = 2*(prj->msh->nEdges + prj->msh->nFaces);
        DoFnums = prj->msh->nNodes + prj->msh->nEdges;
        break;
    case 3:
        DoFnumv = 3*(prj->msh->nEdges + prj->msh->nTetras) + 6*prj->msh->nFaces;
        DoFnums = prj->msh->nNodes + 2*prj->msh->nEdges + prj->msh->nFaces;
        break;
    default:
        throw std::string("Order not implemented yet - CalcDoFnumv()");
    }
}

DoF::~DoF()
{
    s.clear();
    v.clear();
}
