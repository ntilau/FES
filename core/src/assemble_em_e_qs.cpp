#include "assembler.h"
#include <stdexcept>

#include <armadillo>
#include "configuration.h"
#include "memory.h"
#include "equation_system.h"
#include "degree_of_freedom.h"
#include "element_matrix.h"
#include "shape.h"
#include "mesh.h"
#include "boundary_condition.h"
#include "option.h"

#include <cfloat>
#include <complex>
#include <map>

void assembler_em_e_qs::assemble(std::ofstream& logFile, eq_sys& sys)
{
    mesh* msh = sys.msh;
    option* opt = sys.opt;
    project* prj = sys.prj;
    quad* quadr = sys.quadr;
    logFile << "% Assembly Electrostatic:\n";
    logFile << "In solids: ";
    arma::wall_clock tt, lt;
    tt.tic();

    // 2D branch — P1 triangle assembly
    if(msh->nTetras == 0)
    {
        lt.tic();
        sys.set_dofnum(msh->nNodes);
        sys.set_symm_flag(0);

            std::cout << "FE dof = " << sys.get_dofnum() << " ";

        // Material map
        std::map<size_t, const mtrl*> mtrlMap;
        for(const auto& m : msh->tetmtrl) mtrlMap[m.label] = &m;

        // bc label → name/type map
        std::map<size_t, std::string> bcNameMap;
        std::map<size_t, bc::bcTYPE> bcTypeMap;
        for(const auto& bc : msh->facbc) {
            bcTypeMap[bc.label] = bc.type;
            bcNameMap[bc.label] = bc.name;
        }

        // P1 stiffness assembly via triplet map
        std::map<std::pair<arma::uword, arma::uword>, std::complex<double>> A_map;
        for(size_t t = 0; t < msh->nFaces; t++) {
            int n0 = msh->facNodes(t,0), n1 = msh->facNodes(t,1), n2 = msh->facNodes(t,2);
            double x0 = msh->nodPos(n0,0), y0 = msh->nodPos(n0,1);
            double x1 = msh->nodPos(n1,0), y1 = msh->nodPos(n1,1);
            double x2 = msh->nodPos(n2,0), y2 = msh->nodPos(n2,1);
            double J11 = x1-x0, J12 = y1-y0, J21 = x2-x0, J22 = y2-y0;
            double detJ = J11*J22 - J12*J21, area = 0.5*std::abs(detJ);
            if(area < 1e-30) continue;

            double epsr = 1.0;
            size_t lab = msh->facLab(t);
            auto it = mtrlMap.find(lab);
            if(it != mtrlMap.end()) epsr = it->second->epsr;

            double invJ00 = J22/detJ, invJ01 = -J12/detJ;
            double invJ10 = -J21/detJ, invJ11 = J11/detJ;
            double g1x = invJ00*(-1)+invJ01*(-1), g1y = invJ10*(-1)+invJ11*(-1);
            double g2x = invJ00*1+invJ01*0,       g2y = invJ10*1+invJ11*0;
            double g3x = invJ00*0+invJ01*1,        g3y = invJ10*0+invJ11*1;
            double gx[3] = {g1x,g2x,g3x}, gy[3] = {g1y,g2y,g3y};
            int nodes[3] = {n0,n1,n2};
            for(int i=0;i<3;i++) for(int j=0;j<3;j++)
                A_map[{nodes[i], nodes[j]}] += area * epsr * (gx[i]*gx[j] + gy[i]*gy[j]);
        }

        // Build sparse A from triplet map
        {
            std::vector<arma::uword> rows, cols;
            std::vector<std::complex<double>> vals;
            for(auto& kv : A_map) {
                rows.push_back(kv.first.first);
                cols.push_back(kv.first.second);
                vals.push_back(kv.second);
            }
            arma::umat locs(2, rows.size());
            for(size_t k = 0; k < rows.size(); k++) { locs(0,k) = rows[k]; locs(1,k) = cols[k]; }
            arma::cx_vec cvals(vals.data(), vals.size());
            sys.A_mat() = eq_sys::mat_row_type(locs, cvals, sys.get_dofnum(), sys.get_dofnum(), true, false);
        }

        // Voltage bcs from perfect_e edges (via triplet map to handle shared nodes)
        {
            std::map<std::pair<arma::uword, arma::uword>, std::complex<double>> B_map;
            for(size_t s = 0; s < msh->nEdges; s++) {
                size_t mkr = msh->edgLab(s);
                auto bcIt = bcTypeMap.find(mkr);
                if(bcIt != bcTypeMap.end() && bcIt->second == bc::perfect_e) {
                    double V = opt->Vbnd[bcNameMap[mkr]];
                    sys.Dirdofs_vec() = arma::join_cols(sys.Dirdofs_vec(), arma::uvec{msh->edgNodes(s,0)});
                    sys.Dirdofs_vec() = arma::join_cols(sys.Dirdofs_vec(), arma::uvec{msh->edgNodes(s,1)});
                    if(V != 0.0) {
                        B_map[{msh->edgNodes(s,0), 0}] = std::complex<double>(V, 0);
                        B_map[{msh->edgNodes(s,1), 0}] = std::complex<double>(V, 0);
                    }
                }
            }
            if(!B_map.empty()) {
                std::vector<arma::uword> bRows, bCols;
                std::vector<std::complex<double>> bVals;
                for(auto& kv : B_map) {
                    bRows.push_back(kv.first.first);
                    bCols.push_back(kv.first.second);
                    bVals.push_back(kv.second);
                }
                arma::umat bLocs(2, bRows.size());
                for(size_t k = 0; k < bRows.size(); k++) { bLocs(0,k) = bRows[k]; bLocs(1,k) = bCols[k]; }
                arma::cx_vec bCvals(bVals.data(), bVals.size());
                sys.B_mat() = eq_sys::mat_col_type(bLocs, bCvals, sys.get_dofnum(), 1, true, false);
            } else {
                sys.B_mat() = eq_sys::mat_col_type(sys.get_dofnum(), 1);
            }
        }
        if(sys.Dirdofs_vec().n_elem > 0)
            sys.Dirdofs_vec() = arma::unique(sys.Dirdofs_vec());

        logFile << lt.toc() << " s\n";
        sys.set_wave_ports_num(0);
        sys.set_dofreal(sys.get_dofnum());

        // Enforce Dirichlet bcs via copy-then-modify
        if(sys.Dirdofs_vec().n_elem > 0) {
            arma::cx_vec Brhs(sys.get_dofnum(), arma::fill::zeros);
            for(size_t i = 0; i < sys.Dirdofs_vec().n_elem; i++) {
                arma::uword d = sys.Dirdofs_vec()(i);
                Brhs(d) = sys.B_mat()(d, 0);
            }
            for(size_t i = 0; i < sys.Dirdofs_vec().n_elem; i++) {
                arma::uword d = sys.Dirdofs_vec()(i);
                // Save row entries, zero row
                std::vector<arma::uword> rcols;
                for(auto it = sys.A_mat().begin_row(d); it != sys.A_mat().end_row(d); ++it)
                    rcols.push_back(it.col());
                for(auto col : rcols) sys.A_mat()(d, col) = 0;
                // Save col entries, adjust RHS, zero col
                std::vector<std::pair<arma::uword, std::complex<double>>> centries;
                for(auto it = sys.A_mat().begin_col(d); it != sys.A_mat().end_col(d); ++it)
                    centries.push_back({it.row(), *it});
                for(auto& e : centries) {
                    if(e.first != d) Brhs(e.first) -= e.second * Brhs(d);
                    sys.A_mat()(e.first, d) = 0;
                }
                sys.A_mat()(d, d) = 1.0;
            }
            sys.B_mat().zeros();
            for(size_t i = 0; i < sys.get_dofnum(); i++)
                if(Brhs(i) != 0.0)
                    sys.B_mat()(i, 0) = Brhs(i);
        }
        if(sys.B_mat().n_nonzero == 0)
            throw std::runtime_error("Null Right Hand Side");

        logFile << " " << lt.toc() << " s\n";
        logFile << "+" << tt.toc() << "s\n";
        return;
    }

    sys.set_dofnum(dof(prj).dofnums);
        std::cout << "FE dof = " << sys.get_dofnum() << " ";
    sys.set_symm_flag(0);
    sys.A_mat().set_size(sys.get_dofnum(), sys.get_dofnum());
    sys.B_mat().set_size(sys.get_dofnum(), 1);
    lt.tic();
    for(size_t id = 0; id < msh->nTetras; id++)
    {
        mtrl* cmtrl = &(msh->tetmtrl[msh->tetLab(id)]);
        ele_mat lMat(opt->p_ord, 3, msh->tet_geo(id), quadr, cmtrl, shape::hgrad);
        dof cdof(prj, 3, id);
        for(int i=0; i<cdof.s.n_rows; i++)
        {
            for(int j=0; j<cdof.s.n_rows; j++)
            {
                sys.A_mat()(cdof.s(i),cdof.s(j)) += lMat.S(i,j);
            }
        }
    }
    logFile << lt.toc() << " s\n";
        std::cout << lt.toc() << " s\n";
    mem_stat::print(logFile);
        mem_stat::print(std::cout);
    logFile << "On boundaries:\n";
    for(size_t bcid = 0; bcid < msh->facbc.size(); bcid++)
    {
        bc* bc = &(msh->facbc[bcid]);
        if(bc->type == bc::perfect_e)
        {
            lt.tic();
            double f = opt->Vbnd[bc->name];
                std::cout << bc->name << "(" << f << "V)";
            logFile << "\t" << bc->name << ": ";
            for(size_t fid = 0; fid < bc->Faces.size(); fid++)
            {
                dof cdof(prj, 2, bc->Faces(fid));
                {
                    sys.Dirdofs_vec() = arma::join_cols(sys.Dirdofs_vec(), cdof.s);
                    for(int i=0; i<cdof.s.n_rows; i++)
                    {
                        sys.B_mat()(cdof.s(i),0) = f;
                    }
                }
            }
            sys.Dirdofs_vec() = arma::unique(sys.Dirdofs_vec());
            logFile << lt.toc() << " s\n";
                std::cout << " ";
        }
    }
    sys.set_wave_ports_num(0);
    sys.set_dofreal(sys.get_dofnum());
        std::cout << "\nSYS dof = " << sys.get_dofreal() << "\n";
    std::vector<size_t> DirIds(sys.Dirdofs_vec().size());
    for(size_t i=0; i<sys.Dirdofs_vec().size(); i++)
    {
        DirIds[i] = sys.Dirdofs_vec()(i);
    }
    // Enforce Dirichlet bcs: clear rows/cols, set diagonal to 1
    arma::cx_vec Bcol0 = arma::cx_vec(sys.B_mat().col(0));
    arma::cx_vec Bnew = sys.A_mat() * Bcol0;
    for(size_t i=0; i<DirIds.size(); i++)
    {
        Bnew(DirIds[i]) = 0;
        // A row: zero out
        for(auto it = sys.A_mat().begin_row(DirIds[i]); it != sys.A_mat().end_row(DirIds[i]); )
        {
            size_t col = it.col(); ++it;
            sys.A_mat()(DirIds[i], col) = std::complex<double>(0,0);
        }
        // A col: zero out
        for(auto it = sys.A_mat().begin_col(DirIds[i]); it != sys.A_mat().end_col(DirIds[i]); )
        {
            size_t row = it.row(); ++it;
            sys.A_mat()(row, DirIds[i]) = std::complex<double>(0,0);
        }
    }
    for(size_t i=0; i<DirIds.size(); i++)
    {
        Bcol0(DirIds[i]) -= Bnew(DirIds[i]);
        sys.A_mat()(DirIds[i], DirIds[i]) = 1.0;
    }
    // Write Bcol0 back to B
    for(size_t i=0; i<sys.get_dofnum(); i++)
    {
        if(Bcol0(i) != std::complex<double>(0,0))
            sys.B_mat()(i,0) = Bcol0(i);
    }
    if(sys.B_mat().n_nonzero == 0)
    {
        throw std::runtime_error("Null Right Hand Side");
    }
    logFile << " " << lt.toc() << " s\n";
    logFile << "+" << tt.toc() << "s\n";
}

