#include "field.h"
#include "degree_of_freedom.h"
#include "shape.h"
#include "constants.h"

#include <sstream>
#include <fstream>


/// need for 2nd order visualization

field::field(project* prj, arma::cx_mat& sol, double cfreq) : prj(prj), frstPntData(true), freq(cfreq)
{
    // mesh
    Nodes = prj->msh->nodPos;
    nCells = prj->msh->nTetras;
    Cells = prj->msh->tetNodes;
    Dumpmesh();
    // refinement and solution
    arma::mat locNode(4,3); // for node field evaluation
    locNode.fill(0);
    locNode(1,0) = 1.0;
    locNode(2,1) = 1.0;
    locNode(3,2) = 1.0;
    arma::mat locTetNode(1,3);
    locTetNode.fill(0.25);
    CellVal.resize(Cells.n_rows,3);
    NodeVal.resize(Nodes.n_rows,3);
    if(prj->opt->nl)
    {
        size_t dofnum = dof(prj).dofnumv;
        for(size_t jj = 0; jj < prj->opt->n_harm; jj++)
        {
            CellVal.fill(0);
            NodeVal.fill(0);
            cnt = jj+1;
            arma::cx_vec fullSol = sol.col(0);
            arma::vec SumTimes(Nodes.n_rows);
            SumTimes.fill(0);
            for(size_t tit = 0; tit < prj->msh->nTetras; tit++)
            {
                dof cdof(prj, 3, tit);
                jacobian cJac(3, prj->msh->tet_geo(tit));
                arma::cx_vec tSol = fullSol.elem(cdof.v + jj*dofnum);
                shape shp(prj->opt->p_ord, 3, shape::hcurl, locTetNode.row(0), &cJac);
                CellVal.row(tit) = (shp.Nv*tSol).st();
                for(size_t i = 0; i < locNode.n_rows; i++)
                {
                    shape shp(prj->opt->p_ord, 3, shape::hcurl, locNode.row(i), &cJac);
                    NodeVal.row(prj->msh->tetNodes(tit,i)) += (shp.Nv*tSol).st();
                    SumTimes(prj->msh->tetNodes(tit,i)) += 1.0;
                }
            }
            for(size_t i = 0; i < NodeVal.n_rows; i++)
            {
                NodeVal.row(i) *= std::sqrt(2.0); // rms to amplitude
                NodeVal.row(i) /= SumTimes(i);
            }
            dump_efield();
            frstPntData = false;
        }
    }
    if(prj->opt->assembly == option::em_e_qs)
    {
        CellVal.fill(0);
        NodeVal.fill(0);
        cnt = 0;
        arma::cx_vec fullSol = sol.col(0);
        arma::vec SumTimes(Nodes.n_rows);
        SumTimes.fill(0);
        for(size_t tit = 0; tit < prj->msh->nTetras; tit++)
        {
            dof cdof(prj, 3, tit);
            jacobian cJac(3, prj->msh->tet_geo(tit));
            arma::cx_vec tSol = fullSol.elem(cdof.s);
            for(size_t i = 0; i < locNode.n_rows; i++)
            {
                shape shp(prj->opt->p_ord, 3, shape::hgrad, locNode.row(i), &cJac);
                NodeVal(prj->msh->tetNodes(tit,i),0) += arma::cx_mat(shp.Ns*tSol)(0,0);
                SumTimes(prj->msh->tetNodes(tit,i)) += 1.0;
            }
        }
        for(size_t i = 0; i < NodeVal.n_rows; i++)
        {
            NodeVal.row(i) /= SumTimes(i);
        }
        dump_vpot();
        frstPntData = false;
        CellVal.fill(0);
        NodeVal.fill(0);
        cnt = 1;
        SumTimes.fill(0);
        for(size_t tit = 0; tit < prj->msh->nTetras; tit++)
        {
            dof cdof(prj, 3, tit);
            jacobian cJac(3, prj->msh->tet_geo(tit));
            arma::cx_vec tSol = fullSol.elem(cdof.s);
            for(size_t i = 0; i < locNode.n_rows; i++)
            {
                shape shp(prj->opt->p_ord, 3, shape::hgrad, locNode.row(i), &cJac);
                NodeVal.row(prj->msh->tetNodes(tit,i)) -= (shp.dNs*tSol).st();
                SumTimes(prj->msh->tetNodes(tit,i)) += 1.0;
            }
        }
        for(size_t i = 0; i < NodeVal.n_rows; i++)
        {
            NodeVal.row(i) /= SumTimes(i);
        }
        dump_efield();
        frstPntData = false;
    }
    else
    {
        // ── 2D TMz field (scalar Ez) ──
        if(prj->opt->assembly == option::em_ez_fd)
        {
            cnt = 0;
            arma::cx_vec fullSol = arma::sum(sol, 1);
            NodeVal.fill(0);
            size_t nn = std::min(Nodes.n_rows, fullSol.n_elem);
            for(size_t i = 0; i < nn; i++)
                NodeVal(i, 2) = fullSol(i) * std::sqrt(2.0); // rms to amplitude
            dump_efield();
            frstPntData = false;
            for(size_t jj = 0; jj < sol.n_cols && sol.n_cols < 10; jj++)
            {
                cnt = jj + 1;
                NodeVal.fill(0);
                arma::cx_vec colSol = sol.col(jj);
                for(size_t i = 0; i < nn && i < colSol.n_elem; i++)
                    NodeVal(i, 2) = colSol(i) * std::sqrt(2.0);
                dump_efield();
            }
        }
        else
        {
            CellVal.fill(0);
            NodeVal.fill(0);
            cnt = 0;
            arma::cx_vec fullSol = arma::sum(sol,1);
            arma::vec SumTimes(Nodes.n_rows);
            SumTimes.fill(0);
            for(size_t tit = 0; tit < prj->msh->nTetras; tit++)
            {
                dof cdof(prj, 3, tit);
                jacobian cJac(3, prj->msh->tet_geo(tit));
                arma::cx_vec tSol = fullSol.elem(cdof.v);
                shape shp(prj->opt->p_ord, 3, shape::hcurl, locTetNode.row(0), &cJac);
                CellVal.row(tit) = (shp.Nv*tSol).st();
                for(size_t i = 0; i < locNode.n_rows; i++)
                {
                    shape shp(prj->opt->p_ord, 3, shape::hcurl, locNode.row(i), &cJac);
                    NodeVal.row(prj->msh->tetNodes(tit,i)) += (shp.Nv*tSol).st();
                    SumTimes(prj->msh->tetNodes(tit,i)) += 1.0;
                }
            }
            for(size_t i = 0; i < NodeVal.n_rows; i++)
            {
                NodeVal.row(i) *= std::sqrt(2.0); // rms to amplitude
                NodeVal.row(i) /= SumTimes(i);
            }
            dump_efield();
            frstPntData = false;
        for(size_t jj = 0; jj < sol.n_cols && sol.n_cols < 10; jj++)
        {
            // Efield
            CellVal.fill(0);
            NodeVal.fill(0);
            cnt = jj+1;
            arma::cx_vec fullSol = sol.col(jj);
            arma::vec SumTimes(Nodes.n_rows);
            SumTimes.fill(0);
            for(size_t tit = 0; tit < prj->msh->nTetras; tit++)
            {
                dof cdof(prj, 3, tit);
                jacobian cJac(3, prj->msh->tet_geo(tit));
                arma::cx_vec tSol = fullSol.elem(cdof.v);
                shape shp(prj->opt->p_ord, 3, shape::hcurl, locTetNode.row(0), &cJac);
                CellVal.row(tit) = (shp.Nv*tSol).st();
                for(size_t i = 0; i < locNode.n_rows; i++)
                {
                    shape shp(prj->opt->p_ord, 3, shape::hcurl, locNode.row(i), &cJac);
                    NodeVal.row(prj->msh->tetNodes(tit,i)) += (shp.Nv*tSol).st();
                    SumTimes(prj->msh->tetNodes(tit,i)) += 1.0;
                }
            }
            for(size_t i = 0; i < NodeVal.n_rows; i++)
            {
                NodeVal.row(i) *= std::sqrt(2.0); // rms to amplitude
                NodeVal.row(i) /= SumTimes(i);
            }
            dump_efield();
            // Hfield
            CellVal.fill(0);
            NodeVal.fill(0);
            std::complex<double> Hconst(0.0, 1.0 / (2.0*consts::pi*consts::mu0*cfreq));
            cnt = jj+1;
            fullSol = sol.col(jj);
            SumTimes.fill(0);
            for(size_t tit = 0; tit < prj->msh->nTetras; tit++)
            {
                dof cdof(prj, 3, tit);
                jacobian cJac(3, prj->msh->tet_geo(tit));
                arma::cx_vec tSol = fullSol.elem(cdof.v);
                shape shp(prj->opt->p_ord, 3, shape::hcurl, locTetNode.row(0), &cJac);
                CellVal.row(tit) = (shp.dNv*tSol).st();
                for(size_t i = 0; i < locNode.n_rows; i++)
                {
                    shape shp(prj->opt->p_ord, 3, shape::hcurl, locNode.row(i), &cJac);
                    NodeVal.row(prj->msh->tetNodes(tit,i)) += (Hconst*shp.dNv*tSol).st();
                    SumTimes(prj->msh->tetNodes(tit,i)) += 1.0;
                }
            }
            for(size_t i = 0; i < NodeVal.n_rows; i++)
            {
                NodeVal.row(i) *= std::sqrt(2.0); // rms to amplitude
                NodeVal.row(i) /= SumTimes(i);
            }
            dump_hfield();
        }
    }
    }
}
field::~field()
{
}
void field::Dumpmesh()
{
    std::stringstream tmp;
    tmp << freq;
    if(prj->opt->nl)
    {
        tmp << "_nl" << prj->opt->n_harm << "_pow" << prj->opt->power;
    }
    std::ofstream outfield(std::string(prj->opt->name + "_" + tmp.str() + ".vtk").data());
    outfield << "# vtk DataFile Version 2.0\n";
    outfield << "Solution data\n";
    outfield << "ASCII\n";
    outfield << "DATASET UNSTRUCTURED_GRID\n";
    outfield << "POINTS " << Nodes.n_rows << " float \n";
    for(size_t i= 0; i < Nodes.n_rows; i++)
    {
        outfield << (float) Nodes(i,0) << " ";
        outfield << (float) Nodes(i,1) << " ";
        outfield << (float) Nodes(i,2) << "\n";
    }
    if(prj->opt->assembly == option::em_ez_fd && prj->msh->nFaces > 0)
    {
        outfield << "CELLS " << prj->msh->nFaces << " " << 4 * prj->msh->nFaces << "\n";
        for(size_t i = 0; i < prj->msh->nFaces; i++)
            outfield << "3 " << prj->msh->facNodes(i,0) << " "
                     << prj->msh->facNodes(i,1) << " " << prj->msh->facNodes(i,2) << "\n";
        outfield << "CELL_TYPES " << prj->msh->nFaces << "\n";
        for(size_t i = 0; i < prj->msh->nFaces; i++)
            outfield << "5\n";
    }
    else
    {
        outfield << "CELLS " << Cells.n_rows << " " << 5*Cells.n_rows << "\n";
        for(size_t i = 0; i < Cells.n_rows; i++)
        {
            outfield << 4 << " ";
            outfield << Cells(i,0) << " ";
            outfield << Cells(i,1) << " ";
            outfield << Cells(i,2) << " ";
            outfield << Cells(i,3) << "\n";
        }
        outfield << "CELL_TYPES " << Cells.n_rows << "\n";
        for(size_t i = 0; i < Cells.n_rows; i++)
            outfield << 10 << "\n";
    }
}
void field::dump_efield()
{
    std::stringstream tmp;
    tmp << freq;
    if(prj->opt->nl)
    {
        tmp << "_nl" << prj->opt->n_harm << "_pow" << prj->opt->power;
    }
    std::ofstream outfield(std::string(prj->opt->name + "_" + tmp.str() + ".vtk").data(), std::ios::app);
    if(frstPntData)
    {
        outfield << "POINT_DATA " << NodeVal.n_rows  << "\n";
    }
    outfield << "SCALARS " << prj->freq << "_E_norm_" << cnt <<"_[V/m] float 1\n";
    outfield << "LOOKUP_TABLE jet\n";
    for(size_t i = 0; i < NodeVal.n_rows; i++)
    {
        outfield << (float) arma::norm(arma::abs(NodeVal.row(i)),2) << "\n";
    }
    outfield << "VECTORS " << prj->freq << "_E_abs_" << cnt <<"_[V/m] float\n";
    for(size_t i = 0; i < NodeVal.n_rows; i++)
    {
        outfield << (float) std::abs(NodeVal(i,0)) << " ";
        outfield << (float) std::abs(NodeVal(i,1)) << " ";
        outfield << (float) std::abs(NodeVal(i,2)) << "\n";
    }
    outfield << "VECTORS " << prj->freq << "_E_real_" << cnt <<"_[V/m] float\n";
    for(size_t i = 0; i < NodeVal.n_rows; i++)
    {
        outfield << (float) std::real(NodeVal(i,0)) << " ";
        outfield << (float) std::real(NodeVal(i,1)) << " ";
        outfield << (float) std::real(NodeVal(i,2)) << "\n";
    }
    outfield << "VECTORS " << prj->freq << "_E_imag_" << cnt <<"_[V/m] float\n";
    for(size_t i = 0; i < NodeVal.n_rows; i++)
    {
        outfield << (float) std::imag(NodeVal(i,0)) << " ";
        outfield << (float) std::imag(NodeVal(i,1)) << " ";
        outfield << (float) std::imag(NodeVal(i,2)) << "\n";
    }
    outfield.close();
}

void field::dump_hfield()
{
    std::stringstream tmp;
    tmp << freq;
    if(prj->opt->nl)
    {
        tmp << "_nl" << prj->opt->n_harm << "_pow" << prj->opt->power;
    }
    std::ofstream outfield(std::string(prj->opt->name + "_" + tmp.str() + ".vtk").data(), std::ios::app);
    if(frstPntData)
    {
        outfield << "POINT_DATA " << NodeVal.n_rows  << "\n";
    }
    outfield << "SCALARS " << prj->freq << "_H_norm_" << cnt <<"_[A/m] float 1\n";
    outfield << "LOOKUP_TABLE jet\n";
    for(size_t i = 0; i < NodeVal.n_rows; i++)
    {
        outfield << (float) arma::norm(arma::abs(NodeVal.row(i)),2) << "\n";
    }
    outfield << "VECTORS " << prj->freq << "_H_abs_" << cnt <<"_[A/m] float\n";
    for(size_t i = 0; i < NodeVal.n_rows; i++)
    {
        outfield << (float) std::abs(NodeVal(i,0)) << " ";
        outfield << (float) std::abs(NodeVal(i,1)) << " ";
        outfield << (float) std::abs(NodeVal(i,2)) << "\n";
    }
    outfield << "VECTORS " << prj->freq << "_H_real_" << cnt <<"_[A/m] float\n";
    for(size_t i = 0; i < NodeVal.n_rows; i++)
    {
        outfield << (float) std::real(NodeVal(i,0)) << " ";
        outfield << (float) std::real(NodeVal(i,1)) << " ";
        outfield << (float) std::real(NodeVal(i,2)) << "\n";
    }
    outfield << "VECTORS " << prj->freq << "_H_imag_" << cnt <<"_[A/m] float\n";
    for(size_t i = 0; i < NodeVal.n_rows; i++)
    {
        outfield << (float) std::imag(NodeVal(i,0)) << " ";
        outfield << (float) std::imag(NodeVal(i,1)) << " ";
        outfield << (float) std::imag(NodeVal(i,2)) << "\n";
    }
    outfield.close();
}

void field::dump_vpot()
{
    std::stringstream tmp;
    tmp << freq;
    if(prj->opt->nl)
    {
        tmp << "_nl" << prj->opt->n_harm << "_pow" << prj->opt->power;
    }
    std::ofstream outfield(std::string(prj->opt->name + "_" + tmp.str() + ".vtk").data(), std::ios::app);
    if(frstPntData)
    {
        outfield << "POINT_DATA " << NodeVal.n_rows  << "\n";
    }
    outfield << "SCALARS " << prj->freq << "_Phi_" << cnt <<"_[V] float 1\n";
    outfield << "LOOKUP_TABLE jet\n";
    for(size_t i = 0; i < NodeVal.n_rows; i++)
    {
        outfield << (float) arma::norm(arma::abs(NodeVal.row(i)),2) << "\n";
    }
    outfield.close();
}
