#include "degree_of_freedom.h"
#include <stdexcept>
#include <string>

dof::dof(project* prj, size_t dim, size_t id)
{
    switch(dim)
    {
    case 3: // TETRAHEDRON
        switch(prj->opt->p_ord)
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
        case 4:
            v.resize(84);
            s.resize(35);
            s.rows(0,3) = prj->msh->tetNodes.row(id).st();
            s.rows(4,9) = prj->msh->nNodes + prj->msh->tetEdges.row(id).st();
            s.rows(10,15) = prj->msh->nNodes + prj->msh->nEdges + prj->msh->tetEdges.row(id).st();
            s.rows(16,21) = prj->msh->nNodes + 2*prj->msh->nEdges + prj->msh->tetEdges.row(id).st();
            s.rows(22,25) = prj->msh->nNodes + 3*prj->msh->nEdges + prj->msh->tetFaces.row(id).st();
            s.rows(26,29) = prj->msh->nNodes + 3*prj->msh->nEdges + prj->msh->nFaces + prj->msh->tetFaces.row(id).st();
            s.rows(30,33) = prj->msh->nNodes + 3*prj->msh->nEdges + 2*prj->msh->nFaces + prj->msh->tetFaces.row(id).st();
            s(34) = prj->msh->nNodes + 3*prj->msh->nEdges + 3*prj->msh->nFaces + id;
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
            { size_t base = 3*prj->msh->nEdges + 6*prj->msh->nFaces + 3*prj->msh->nTetras;
            v.rows(45,50) = base + prj->msh->tetEdges.row(id).st();
            for(size_t i=0; i<4; i++)
            {
                v(51+6*i) = base + prj->msh->nEdges + prj->msh->tetFaces(id,i);
                v(52+6*i) = base + prj->msh->nEdges + prj->msh->nFaces + prj->msh->tetFaces(id,i);
                v(53+6*i) = base + prj->msh->nEdges + 2*prj->msh->nFaces + prj->msh->tetFaces(id,i);
                v(54+6*i) = base + prj->msh->nEdges + 3*prj->msh->nFaces + prj->msh->tetFaces(id,i);
                v(55+6*i) = base + prj->msh->nEdges + 4*prj->msh->nFaces + prj->msh->tetFaces(id,i);
                v(56+6*i) = base + prj->msh->nEdges + 5*prj->msh->nFaces + prj->msh->tetFaces(id,i);
            }
            size_t volBase = base + prj->msh->nEdges + 6*prj->msh->nFaces;
            v(75) = volBase + id;
            v(76) = volBase + prj->msh->nTetras + id;
            v(77) = volBase + 2*prj->msh->nTetras + id;
            v(78) = volBase + 3*prj->msh->nTetras + id;
            v(79) = volBase + 4*prj->msh->nTetras + id;
            v(80) = volBase + 5*prj->msh->nTetras + id;
            v(81) = volBase + 6*prj->msh->nTetras + id;
            v(82) = volBase + 7*prj->msh->nTetras + id;
            v(83) = volBase + 8*prj->msh->nTetras + id; }
            break;
        default:
            throw std::runtime_error("3D dof mapping order not yet implemented");
        }
        break;
    case 2: // TRIANGLE
        switch(prj->opt->p_ord)
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
        case 4:
            s.resize(15);
            v.resize(24);
            s.rows(0,2) = prj->msh->facNodes.row(id).st();
            s.rows(3,5) = prj->msh->nNodes + prj->msh->facEdges.row(id).st();
            s.rows(6,8) = prj->msh->nNodes + prj->msh->nEdges + prj->msh->facEdges.row(id).st();
            s.rows(9,11) = prj->msh->nNodes + 2*prj->msh->nEdges + prj->msh->facEdges.row(id).st();
            s(12) = prj->msh->nNodes + 3*prj->msh->nEdges + id;
            s(13) = prj->msh->nNodes + 3*prj->msh->nEdges + prj->msh->nFaces + id;
            s(14) = prj->msh->nNodes + 3*prj->msh->nEdges + 2*prj->msh->nFaces + id;
            v.rows(0,2) = prj->msh->facEdges.row(id).st();
            v.rows(3,5) = prj->msh->nEdges + prj->msh->facEdges.row(id).st();
            v(6) = 2*prj->msh->nEdges + id;
            v(7) = 2*prj->msh->nEdges + prj->msh->nFaces + id;
            v.rows(8,10) = 2*prj->msh->nEdges + 2*prj->msh->nFaces + prj->msh->facEdges.row(id).st();
            v(11) = 3*prj->msh->nEdges + 2*prj->msh->nFaces + id;
            v(12) = 3*prj->msh->nEdges + 3*prj->msh->nFaces + id;
            v(13) = 3*prj->msh->nEdges + 4*prj->msh->nFaces + id;
            v(14) = 3*prj->msh->nEdges + 5*prj->msh->nFaces + id;
            v.rows(15,17) = 3*prj->msh->nEdges + 6*prj->msh->nFaces + prj->msh->facEdges.row(id).st();
            v(18) = 4*prj->msh->nEdges + 6*prj->msh->nFaces + id;
            v(19) = 4*prj->msh->nEdges + 7*prj->msh->nFaces + id;
            v(20) = 4*prj->msh->nEdges + 8*prj->msh->nFaces + id;
            v(21) = 4*prj->msh->nEdges + 9*prj->msh->nFaces + id;
            v(22) = 4*prj->msh->nEdges + 10*prj->msh->nFaces + id;
            v(23) = 4*prj->msh->nEdges + 11*prj->msh->nFaces + id;
            break;
        default:
            throw std::runtime_error("2D dof mapping order not yet implemented");
        }
        break;
    default:
        throw std::runtime_error("dim error in dof()");
    }
}

dof::dof(project* prj)
{
    switch(prj->opt->p_ord)
    {
    case 1:
        dofnumv = prj->msh->nEdges;
        dofnums = prj->msh->nNodes;
        break;
    case 2:
        dofnumv = 2*(prj->msh->nEdges + prj->msh->nFaces);
        dofnums = prj->msh->nNodes + prj->msh->nEdges;
        break;
    case 3:
        dofnumv = 3*(prj->msh->nEdges + prj->msh->nTetras) + 6*prj->msh->nFaces;
        dofnums = prj->msh->nNodes + 2*prj->msh->nEdges + prj->msh->nFaces;
        break;
    case 4:
        dofnumv = 4*prj->msh->nEdges + 12*prj->msh->nFaces + 12*prj->msh->nTetras;
        dofnums = prj->msh->nNodes + 3*prj->msh->nEdges + 3*prj->msh->nFaces + prj->msh->nTetras;
        break;
    default:
        throw std::runtime_error("Order not implemented yet - CalcDoFnumv()");
    }
}

dof::~dof()
{
    s.clear();
    v.clear();
}
