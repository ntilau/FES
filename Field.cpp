#include "Field.h"
#include "DoF.h"
#include "Shape.h"
#include "Const.h"


/// need for 2nd order visualization

Field::Field(Project* prj, arma::cx_mat& sol, double& cfreq) : prj(prj), frstPntData(true), freq(cfreq)
{
    // Mesh
    Nodes = prj->msh->nodPos;
    nCells = prj->msh->nTetras;
    Cells = prj->msh->tetNodes;
    DumpMesh();
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
        size_t DoFnum = DoF(prj).DoFnumv;
        for(size_t jj = 0; jj < prj->opt->nHarm; jj++)
        {
            CellVal.fill(0);
            NodeVal.fill(0);
            cnt = jj+1;
            arma::cx_vec fullSol = sol.col(0);
            arma::vec SumTimes(Nodes.n_rows);
            SumTimes.fill(0);
            #pragma omp parallel for
            for(size_t tit = 0; tit < prj->msh->nTetras; tit++)
            {
                DoF cDoF(prj, 3, tit);
                Jacobian cJac(3, prj->msh->tetGeo(tit));
                arma::cx_vec tSol = fullSol.elem(cDoF.v + jj*DoFnum);
                Shape shp(prj->opt->pOrd, 3, Shape::Hcurl, locTetNode.row(0), &cJac);
                CellVal.row(tit) = (shp.Nv*tSol).st();
                for(size_t i = 0; i < locNode.n_rows; i++)
                {
                    Shape shp(prj->opt->pOrd, 3, Shape::Hcurl, locNode.row(i), &cJac);
                    NodeVal.row(prj->msh->tetNodes(tit,i)) += (shp.Nv*tSol).st();
                    SumTimes(prj->msh->tetNodes(tit,i)) += 1.0;
                }
            }
            #pragma omp parallel for
            for(size_t i = 0; i < NodeVal.n_rows; i++)
            {
                NodeVal.row(i) *= std::sqrt(2.0); // rms to amplitude
                NodeVal.row(i) /= SumTimes(i);
            }
            DumpEfield();
            frstPntData = false;
        }
    }
    if(prj->opt->stat)
    {
        CellVal.fill(0);
        NodeVal.fill(0);
        cnt = 0;
        arma::cx_vec fullSol = sol.col(0);
        arma::vec SumTimes(Nodes.n_rows);
        SumTimes.fill(0);
        #pragma omp parallel for
        for(size_t tit = 0; tit < prj->msh->nTetras; tit++)
        {
            DoF cDoF(prj, 3, tit);
            Jacobian cJac(3, prj->msh->tetGeo(tit));
            arma::cx_vec tSol = fullSol.elem(cDoF.s);
            for(size_t i = 0; i < locNode.n_rows; i++)
            {
                Shape shp(prj->opt->pOrd, 3, Shape::Hgrad, locNode.row(i), &cJac);
                NodeVal(prj->msh->tetNodes(tit,i),0) += arma::cx_mat(shp.Ns*tSol)(0,0);
                SumTimes(prj->msh->tetNodes(tit,i)) += 1.0;
            }
        }
        #pragma omp parallel for
        for(size_t i = 0; i < NodeVal.n_rows; i++)
        {
            NodeVal.row(i) /= SumTimes(i);
        }
        DumpVpot();
        frstPntData = false;
        CellVal.fill(0);
        NodeVal.fill(0);
        cnt = 1;
        SumTimes.fill(0);
        #pragma omp parallel for
        for(size_t tit = 0; tit < prj->msh->nTetras; tit++)
        {
            DoF cDoF(prj, 3, tit);
            Jacobian cJac(3, prj->msh->tetGeo(tit));
            arma::cx_vec tSol = fullSol.elem(cDoF.s);
            for(size_t i = 0; i < locNode.n_rows; i++)
            {
                Shape shp(prj->opt->pOrd, 3, Shape::Hgrad, locNode.row(i), &cJac);
                NodeVal.row(prj->msh->tetNodes(tit,i)) -= (shp.dNs*tSol).st();
                SumTimes(prj->msh->tetNodes(tit,i)) += 1.0;
            }
        }
        #pragma omp parallel for
        for(size_t i = 0; i < NodeVal.n_rows; i++)
        {
            NodeVal.row(i) /= SumTimes(i);
        }
        DumpEfield();
        frstPntData = false;
    }
    else
    {
        CellVal.fill(0);
        NodeVal.fill(0);
        cnt = 0;
        arma::cx_vec fullSol = arma::sum(sol,1);
        arma::vec SumTimes(Nodes.n_rows);
        SumTimes.fill(0);
        #pragma omp parallel for
        for(size_t tit = 0; tit < prj->msh->nTetras; tit++)
        {
            DoF cDoF(prj, 3, tit);
            Jacobian cJac(3, prj->msh->tetGeo(tit));
            arma::cx_vec tSol = fullSol.elem(cDoF.v);
            Shape shp(prj->opt->pOrd, 3, Shape::Hcurl, locTetNode.row(0), &cJac);
            CellVal.row(tit) = (shp.Nv*tSol).st();
            for(size_t i = 0; i < locNode.n_rows; i++)
            {
                Shape shp(prj->opt->pOrd, 3, Shape::Hcurl, locNode.row(i), &cJac);
                NodeVal.row(prj->msh->tetNodes(tit,i)) += (shp.Nv*tSol).st();
                SumTimes(prj->msh->tetNodes(tit,i)) += 1.0;
            }
        }
        #pragma omp parallel for
        for(size_t i = 0; i < NodeVal.n_rows; i++)
        {
            NodeVal.row(i) *= std::sqrt(2.0); // rms to amplitude
            NodeVal.row(i) /= SumTimes(i);
        }
        DumpEfield();
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
            #pragma omp parallel for
            for(size_t tit = 0; tit < prj->msh->nTetras; tit++)
            {
                DoF cDoF(prj, 3, tit);
                Jacobian cJac(3, prj->msh->tetGeo(tit));
                arma::cx_vec tSol = fullSol.elem(cDoF.v);
                Shape shp(prj->opt->pOrd, 3, Shape::Hcurl, locTetNode.row(0), &cJac);
                CellVal.row(tit) = (shp.Nv*tSol).st();
                for(size_t i = 0; i < locNode.n_rows; i++)
                {
                    Shape shp(prj->opt->pOrd, 3, Shape::Hcurl, locNode.row(i), &cJac);
                    NodeVal.row(prj->msh->tetNodes(tit,i)) += (shp.Nv*tSol).st();
                    SumTimes(prj->msh->tetNodes(tit,i)) += 1.0;
                }
            }
            #pragma omp parallel for
            for(size_t i = 0; i < NodeVal.n_rows; i++)
            {
                NodeVal.row(i) *= std::sqrt(2.0); // rms to amplitude
                NodeVal.row(i) /= SumTimes(i);
            }
            DumpEfield();
            // Hfield
            CellVal.fill(0);
            NodeVal.fill(0);
            std::complex<double> Hconst(0.0, 1.0 / (2.0*Const::pi*Const::mu0*cfreq));
            cnt = jj+1;
            fullSol = sol.col(jj);
            //arma::vec SumTimes(Nodes.n_rows);
            SumTimes.fill(0);
            #pragma omp parallel for
            for(size_t tit = 0; tit < prj->msh->nTetras; tit++)
            {
                DoF cDoF(prj, 3, tit);
                Jacobian cJac(3, prj->msh->tetGeo(tit));
                arma::cx_vec tSol = fullSol.elem(cDoF.v);
                Shape shp(prj->opt->pOrd, 3, Shape::Hcurl, locTetNode.row(0), &cJac);
                CellVal.row(tit) = (shp.dNv*tSol).st();
                for(size_t i = 0; i < locNode.n_rows; i++)
                {
                    Shape shp(prj->opt->pOrd, 3, Shape::Hcurl, locNode.row(i), &cJac);
                    NodeVal.row(prj->msh->tetNodes(tit,i)) += (Hconst*shp.dNv*tSol).st();
                    SumTimes(prj->msh->tetNodes(tit,i)) += 1.0;
                }
            }
            #pragma omp parallel for
            for(size_t i = 0; i < NodeVal.n_rows; i++)
            {
                NodeVal.row(i) *= std::sqrt(2.0); // rms to amplitude
                NodeVal.row(i) /= SumTimes(i);
            }
            DumpHfield();
        }
    }
}
Field::~Field()
{
}
void Field::DumpMesh()
{
    std::stringstream tmp;
    tmp << freq;
    if(prj->opt->nl)
    {
        tmp << "_nl" << prj->opt->nHarm << "_pow" << prj->opt->power;
    }
    std::ofstream outField(std::string(prj->opt->name + "_" + tmp.str() + ".vtk").data());
    outField << "# vtk DataFile Version 2.0\n";
    outField << "Solution data\n";
    outField << "ASCII\n";
    outField << "DATASET UNSTRUCTURED_GRID\n";
    outField << "POINTS " << Nodes.n_rows << " float \n";
    for(size_t i= 0; i < Nodes.n_rows; i++)
    {
        outField << (float) Nodes(i,0) << " ";
        outField << (float) Nodes(i,1) << " ";
        outField << (float) Nodes(i,2) << "\n";
    }
    outField << "CELLS " << Cells.n_rows << " " << 5*Cells.n_rows << "\n";
    for(size_t i = 0; i < Cells.n_rows; i++)
    {
        outField << 4 << " ";
        outField << Cells(i,0) << " ";
        outField << Cells(i,1) << " ";
        outField << Cells(i,2) << " ";
        outField << Cells(i,3) << "\n";
    }
    outField << "CELL_TYPES " << Cells.n_rows << "\n";
    for(size_t i = 0; i < Cells.n_rows; i++)
    {
        outField << 10 << "\n";
    }
}
void Field::DumpEfield()
{
    std::stringstream tmp;
    tmp << freq;
    if(prj->opt->nl)
    {
        tmp << "_nl" << prj->opt->nHarm << "_pow" << prj->opt->power;
    }
    std::ofstream outField(std::string(prj->opt->name + "_" + tmp.str() + ".vtk").data(), std::ios::app);
    if(frstPntData)
    {
        outField << "POINT_DATA " << NodeVal.n_rows  << "\n";
    }
    outField << "SCALARS " << prj->freq << "_E_norm_" << cnt <<"_[V/m] float 1\n";
    outField << "LOOKUP_TABLE jet\n";
    for(size_t i = 0; i < NodeVal.n_rows; i++)
    {
        outField << (float) arma::norm(arma::abs(NodeVal.row(i)),2) << "\n";
    }
    outField << "VECTORS " << prj->freq << "_E_abs_" << cnt <<"_[V/m] float\n";
    for(size_t i = 0; i < NodeVal.n_rows; i++)
    {
        outField << (float) std::abs(NodeVal(i,0)) << " ";
        outField << (float) std::abs(NodeVal(i,1)) << " ";
        outField << (float) std::abs(NodeVal(i,2)) << "\n";
    }
    outField << "VECTORS " << prj->freq << "_E_real_" << cnt <<"_[V/m] float\n";
    for(size_t i = 0; i < NodeVal.n_rows; i++)
    {
        outField << (float) std::real(NodeVal(i,0)) << " ";
        outField << (float) std::real(NodeVal(i,1)) << " ";
        outField << (float) std::real(NodeVal(i,2)) << "\n";
    }
    outField << "VECTORS " << prj->freq << "_E_imag_" << cnt <<"_[V/m] float\n";
    for(size_t i = 0; i < NodeVal.n_rows; i++)
    {
        outField << (float) std::imag(NodeVal(i,0)) << " ";
        outField << (float) std::imag(NodeVal(i,1)) << " ";
        outField << (float) std::imag(NodeVal(i,2)) << "\n";
    }
    outField.close();
}

void Field::DumpHfield()
{
    std::stringstream tmp;
    tmp << freq;
    if(prj->opt->nl)
    {
        tmp << "_nl" << prj->opt->nHarm << "_pow" << prj->opt->power;
    }
    std::ofstream outField(std::string(prj->opt->name + "_" + tmp.str() + ".vtk").data(), std::ios::app);
    if(frstPntData)
    {
        outField << "POINT_DATA " << NodeVal.n_rows  << "\n";
    }
    outField << "SCALARS " << prj->freq << "_H_norm_" << cnt <<"_[A/m] float 1\n";
    outField << "LOOKUP_TABLE jet\n";
    for(size_t i = 0; i < NodeVal.n_rows; i++)
    {
        outField << (float) arma::norm(arma::abs(NodeVal.row(i)),2) << "\n";
    }
    outField << "VECTORS " << prj->freq << "_H_abs_" << cnt <<"_[A/m] float\n";
    for(size_t i = 0; i < NodeVal.n_rows; i++)
    {
        outField << (float) std::abs(NodeVal(i,0)) << " ";
        outField << (float) std::abs(NodeVal(i,1)) << " ";
        outField << (float) std::abs(NodeVal(i,2)) << "\n";
    }
    outField << "VECTORS " << prj->freq << "_H_real_" << cnt <<"_[A/m] float\n";
    for(size_t i = 0; i < NodeVal.n_rows; i++)
    {
        outField << (float) std::real(NodeVal(i,0)) << " ";
        outField << (float) std::real(NodeVal(i,1)) << " ";
        outField << (float) std::real(NodeVal(i,2)) << "\n";
    }
    outField << "VECTORS " << prj->freq << "_H_imag_" << cnt <<"_[A/m] float\n";
    for(size_t i = 0; i < NodeVal.n_rows; i++)
    {
        outField << (float) std::imag(NodeVal(i,0)) << " ";
        outField << (float) std::imag(NodeVal(i,1)) << " ";
        outField << (float) std::imag(NodeVal(i,2)) << "\n";
    }
    outField.close();
}

void Field::DumpVpot()
{
    std::stringstream tmp;
    tmp << freq;
    if(prj->opt->nl)
    {
        tmp << "_nl" << prj->opt->nHarm << "_pow" << prj->opt->power;
    }
    std::ofstream outField(std::string(prj->opt->name + "_" + tmp.str() + ".vtk").data(), std::ios::app);
    if(frstPntData)
    {
        outField << "POINT_DATA " << NodeVal.n_rows  << "\n";
    }
    outField << "SCALARS " << prj->freq << "_Phi_" << cnt <<"_[V] float 1\n";
    outField << "LOOKUP_TABLE jet\n";
    for(size_t i = 0; i < NodeVal.n_rows; i++)
    {
        outField << (float) arma::norm(arma::abs(NodeVal.row(i)),2) << "\n";
    }
    outField.close();
}
