#include "assembler.h"
#include <stdexcept>

#include <armadillo>
#include "constants.h"
#include "memory.h"
#include "equation_system.h"
#include "degree_of_freedom.h"
#include "element_matrix.h"
#include "eigen_solver.h"
#include "mesh.h"
#include "option.h"

#include <cfloat> // for DBL_MIN

void assembler_em_e_fd_dd::assemble(std::ofstream& logFile, eq_sys& sys)
{
    mesh* msh = sys.msh;
    option* opt = sys.opt;
    project* prj = sys.prj;
    quad* quadr = sys.quadr;
    read_port_amplitudes(logFile, sys);
    logFile << "In solids: ";
    arma::wall_clock tt, lt;
    tt.tic();
    double k0 = 2.0 * consts::pi * sys.get_freq() / consts::c0;
    double kk = k0*k0;
    sys.set_dofnum(dof(prj).dofnumv);
        std::cout << "RAW FE dof = " << sys.get_dofnum() << "\n";
    ///
    arma::field<arma::uvec> Bnddofv, Bnddofs, Domdofmap, BnddofmapJ;
    arma::field<std::vector<bool> > Intdofv;
    Intdofv.set_size(msh->nDomains);
    Bnddofv.set_size(msh->nDomains);
    Domdofmap.set_size(msh->nDomains);
    BnddofmapJ.set_size(msh->nDomains);
    lt.tic();
    for(size_t did = 0; did < msh->nDomains; did++)
    {
        std::vector<bool> dofinternal(sys.get_dofnum(), true);
        arma::uvec bndFaces = msh->domFaces(did);
        for(size_t fif = 0; fif < bndFaces.n_rows; fif++)
        {
            dof cdof(prj, 2, bndFaces(fif));
            Bnddofv(did) = arma::unique(arma::join_cols(Bnddofv(did), cdof.v));
            for(size_t id=0; id<cdof.v.n_rows; id++)
            {
                dofinternal[cdof.v(id)] = false;
            }
        }
        Intdofv(did) = dofinternal;
    }
    sys.doflevel_vec().clear();
    size_t gidx = 0;
    sys.doflevel_vec().push_back(gidx);
    for(size_t did = 0; did < msh->nDomains; did++)
    {
        arma::uvec intTetras = msh->domTetras(did);
        arma::uvec domdofmap(sys.get_dofnum());
        arma::uvec bndJdofmap(sys.get_dofnum());
        arma::uvec bndRdofmap(sys.get_dofnum());
        domdofmap.fill(UINT_MAX);
        bndJdofmap.fill(UINT_MAX);
        bndRdofmap.fill(UINT_MAX);
        std::vector<bool> dofinternal = Intdofv(did);
        std::cout << "Domain " << did << ": " << gidx << "-";
        for(size_t tit = 0; tit < intTetras.n_rows; tit++)
        {
            dof cdof(prj, 3, intTetras(tit));
            for(size_t id=0; id<cdof.v.n_rows; id++)
            {
                if(dofinternal[cdof.v(id)])
                {
                    domdofmap(cdof.v(id)) = gidx++;
                    dofinternal[cdof.v(id)] = false;
                }
            }
        }
        std::cout << gidx-1 << ", ";
        std::cout << gidx << "-";
        arma::uvec domBnddofv = Bnddofv(did);
        for(size_t id = 0; id < domBnddofv.n_rows; id++)
        {
            domdofmap(domBnddofv(id)) = gidx++;
        }
        Domdofmap(did) = domdofmap;
        std::cout << gidx-1;
        if(!opt->ddn)
        {
            std::cout << ", " << gidx << "-";
            for(size_t id = 0; id < domBnddofv.n_rows; id++)
            {
                bndJdofmap(domBnddofv(id)) = gidx++;
            }
            BnddofmapJ(did) = bndJdofmap;
            std::cout << gidx-1;
        }
        sys.doflevel_vec().push_back(gidx);
        std::cout << "\n";
    }
    size_t dofnum = gidx;
    sys.Invdofmapv_vec().resize(dofnum);
    sys.Invdofmapv_vec().fill(UINT_MAX);
    sys.dofmapv_vec().resize(dofnum);
    sys.dofmapv_vec().fill(UINT_MAX);
    std::vector<bool> mapped(sys.get_dofnum(), true);
    for(size_t did = 0; did < msh->nDomains; did++)
    {
        arma::uvec domdofmap = Domdofmap(did);
        for(size_t i = 0; i < sys.get_dofnum(); i++)
        {
            if(mapped[i])
            {
                if(domdofmap(i) < UINT_MAX)
                {
                    sys.Invdofmapv_vec()(domdofmap(i)) = i;
                    sys.dofmapv_vec()(i) = domdofmap(i);
                    mapped[i] = false;
                }
            }
        }
    }
    sys.set_dofnum(dofnum);
    ///
    std::vector<bool> doftoLeave(dofnum, true);
        std::cout << "FE dof = " << dofnum << " ";
    sys.set_symm_flag(0);
    sys.A_mat() = eq_sys::mat_row_type(dofnum, dofnum);
    sys.PR_mat() = eq_sys::mat_row_type(dofnum, dofnum);
    sys.B_mat().zeros();
    sys.Sol_mat().clear();
    sys.Sp_mat().clear();
    lt.tic();

    // Use map-based assembly for A to avoid Armadillo SpMat cache issues
    std::map<std::pair<arma::uword, arma::uword>, std::complex<double>> A_map;
    for(size_t id = 0; id < msh->nTetras; id++)
    {
        mtrl* cmtrl = &(msh->tetmtrl[msh->tetLab(id)]);
        ele_mat lMat(opt->p_ord, 3, msh->tet_geo(id), quadr, cmtrl, shape::hcurl);
        dof cdof(prj, 3, id);
        cdof.v = Domdofmap(msh->tetDom(id)).elem(cdof.v);
        for(int i=0; i<cdof.v.n_rows; i++)
        {
            for(int j=0; j<cdof.v.n_rows; j++)
            {
                if(cdof.v(i)<=cdof.v(j))
                {
                    A_map[{cdof.v(i), cdof.v(j)}] += lMat.S(i,j) + k0*lMat.Z(i,j) - kk*lMat.T(i,j);
                }
            }
        }
    }
    // Build A from map
    sys.A_mat() = build_sparse(A_map, dofnum);
    A_map.clear();
    // Armadillo SpMat workaround: writing a non-zero value forces subsequent
    // A(i,j)+= to use the MapMat cache instead of CSC fast-path, preventing
    // data loss when cache syncs rebuild the CSC structure.
    const std::complex<double> cache_seed(1e-100, 0);
    sys.A_mat()(0, 0) = cache_seed;  sys.A_mat().sync();
    sys.PR_mat()(0, 0) = cache_seed; sys.PR_mat().sync();
    logFile << lt.toc() << " s\n";
        std::cout << lt.toc() << " s\n";
    mem_stat::print(logFile);
        mem_stat::print(std::cout);
    logFile << "On boundaries:\n";
    for(size_t bcid = 0; bcid < msh->facbc.size(); bcid++)
    {
        bc* bc = &(msh->facbc[bcid]);
        switch(bc->type)
        {
        case bc::perfect_e :
            lt.tic();
                std::cout << bc->name;
            logFile << "\t" << bc->name << ": ";
            for(size_t fid = 0; fid < bc->Faces.size(); fid++)
            {
                dof cdof(prj, 2, bc->Faces(fid));
                arma::uvec adjTet = msh->facAdjTet(bc->Faces(fid));
                cdof.v = Domdofmap(msh->tetDom(adjTet(0))).elem(cdof.v);
                {
                    sys.Dirdofs_vec() = arma::join_cols(sys.Dirdofs_vec(), cdof.s);
                    sys.Dirdofv_vec() = arma::join_cols(sys.Dirdofv_vec(), cdof.v);
                    for(size_t dofid=0; dofid < cdof.v.n_rows; dofid++)
                    {
                        doftoLeave[cdof.v(dofid)] = false;
                    }
                }
            }
            sys.Dirdofs_vec() = arma::unique(sys.Dirdofs_vec());
            sys.Dirdofv_vec() = arma::unique(sys.Dirdofv_vec());
            logFile << lt.toc() << " s\n";
                std::cout << " ";
            break;
        case bc::wave_port:
            break;
        case bc::radiation:
            lt.tic();
                std::cout << bc->name;
            logFile << "\t" << bc->name << ": ";
            for(size_t fid = 0; fid < bc->Faces.size(); fid++)
            {
                arma::uvec adjTet = msh->facAdjTet(bc->Faces(fid));
                mtrl* cmtrl = &(msh->tetmtrl[msh->tetLab(adjTet(0))]);
                std::complex<double> epsr(cmtrl->epsr, cmtrl->calc_epsr2(sys.get_freq()));
                double mur = cmtrl->mur;
                ele_mat lMat(opt->p_ord, 2, msh->fac_geo(bc->Faces(fid)), quadr,
                            cmtrl, msh->int_node(bc->Faces(fid)));
                dof cdof(prj, 2, bc->Faces(fid));
                cdof.v = Domdofmap(msh->tetDom(adjTet(0))).elem(cdof.v);
                for(int i=0; i<cdof.v.n_rows; i++)
                {
                    for(int j=0; j<cdof.v.n_rows; j++)
                    {
                        if(cdof.v(i)<=cdof.v(j))
                        {
                            sys.A_mat()(cdof.v(i),cdof.v(j)) += std::complex<double>(0.0,k0)*lMat.Tt(i,j)*std::sqrt(epsr/mur);
                        }
                    }
                }
            }
            logFile << lt.toc() << " s\n";
                std::cout << " ";
            break;
        case bc::perfect_h:
            lt.tic();
                std::cout << bc->name;
            logFile << "\t" << bc->name << ": ";
            logFile << lt.toc() << " s\n";
                std::cout << " ";
            break;
        default:
            throw std::runtime_error("Wrong boundary type");
        }
    }
	sys.set_wave_ports_num(0);
	sys.set_wave_ports_dofnum(0);
	sys.wave_portIds_vec().reset();
	compute_waveport_modes(logFile, sys, msh, opt, prj, quadr, k0, kk, doftoLeave, lt, &Domdofmap);
    size_t reg1, reg2;
    std::complex<double> epsrd1, epsrd2, murd1, murd2;
    for(size_t did = 0; did < msh->nDomains; did++)
    {
        arma::uvec bndFaces = msh->domFaces(did);
        for(size_t fif = 0; fif < bndFaces.n_rows; fif++)
        {
            arma::uvec adjTet = msh->facAdjTet(bndFaces(fif));
            mtrl* cmtrl0 = &(msh->tetmtrl[msh->tetLab(adjTet(0))]);
            mtrl* cmtrl1 = &(msh->tetmtrl[msh->tetLab(adjTet(1))]);
            ele_mat lMat(opt->p_ord, 2, msh->fac_geo(bndFaces(fif)), quadr,
                        cmtrl0, msh->int_node(bndFaces(fif)));
            reg1 = did;
            if(msh->tetDom(adjTet(0)) == reg1)
            {
                epsrd1 = std::complex<double>(cmtrl0->epsr, cmtrl0->calc_epsr2(sys.get_freq()));
                murd1 = std::complex<double>(cmtrl0->mur,0.0);
                epsrd2 = std::complex<double>(cmtrl1->epsr, cmtrl1->calc_epsr2(sys.get_freq()));
                murd2 = std::complex<double>(cmtrl1->mur,0.0);
                reg2 = msh->tetDom(adjTet(1));
            }
            else
            {
                epsrd2 = std::complex<double>(cmtrl0->epsr, cmtrl0->calc_epsr2(sys.get_freq()));
                murd2 = std::complex<double>(cmtrl0->mur,0.0);
                epsrd1 = std::complex<double>(cmtrl1->epsr, cmtrl1->calc_epsr2(sys.get_freq()));
                murd1 = std::complex<double>(cmtrl1->mur,0.0);
                reg2 = msh->tetDom(adjTet(0));
            }
            dof cdof1(prj, 2, bndFaces(fif));
            cdof1.v = Domdofmap(reg1).elem(cdof1.v);
            dof cdof2(prj, 2, bndFaces(fif));
            cdof2.v = Domdofmap(reg2).elem(cdof2.v);
            if(opt->ddn)
            {
                jacobian* cJac30 = new jacobian(3, prj->msh->tet_geo(adjTet(0)));
                jacobian* cJac31 = new jacobian(3, prj->msh->tet_geo(adjTet(1)));
                arma::mat cGeo = prj->msh->fac_geo(bndFaces(fif));
                size_t RefFace = 0;
                arma::mat int_node = prj->msh->int_node(bndFaces(fif), RefFace);
                //std::cout << RefFace << "\n";
                arma::mat RefGeo(3,3);
                RefGeo.fill(0);
                arma::uvec map3to2(3); //first order
                switch(RefFace)
                {
                case 0:
                    RefGeo(0,0) = 1.0;
                    RefGeo(1,0) = -1.0;
                    RefGeo(1,1) = 1.0;
                    RefGeo(2,0) = -1.0;
                    RefGeo(2,2) = 1.0;
                    map3to2(0) = 5;
                    map3to2(1) = 4;
                    map3to2(2) = 3;
                    break;
                case 1:
                    RefGeo(1,1) = 1.0;
                    RefGeo(2,2) = 1.0;
                    map3to2(0) = 5;
                    map3to2(1) = 2;
                    map3to2(2) = 1;
                    break;
                case 2:
                    RefGeo(1,0) = 1.0;
                    RefGeo(2,2) = 1.0;
                    map3to2(0) = 4;
                    map3to2(1) = 2;
                    map3to2(2) = 0;
                    break;
                case 3:
                    RefGeo(1,0) = 1.0;
                    RefGeo(2,1) = 1.0;
                    map3to2(0) = 3;
                    map3to2(1) = 1;
                    map3to2(2) = 0;
                    break;
                }
                arma::vec v0 = cGeo.row(0).st();
                arma::vec v1 = cGeo.row(1).st();
                arma::vec v2 = cGeo.row(2).st();
                arma::vec vIn = ((v0+v1+v2)/3.0) - int_node;
                vIn /= arma::norm(vIn,2);
                v1 -= v0;
                v2 -= v0;
                arma::vec u = v1 / arma::norm(v1,2);
                arma::vec n = arma::cross(v1,v2);
                n *= arma::dot(n,vIn);
                n /= arma::norm(n,2);
                arma::vec v = arma::cross(n,u);
                arma::mat cGeo2(3,2);
                cGeo2.fill(0);
                cGeo2(1,0) = arma::dot(v1, u);
                cGeo2(2,0) = arma::dot(v2, u);
                cGeo2(2,1) = arma::dot(v2, v);
                jacobian* cJac2 = new jacobian(2, cGeo2);
                arma::cx_mat Dij, Dji, S, T;
                switch(opt->p_ord)
                {
                case 1:
                    Dij.resize(6,6);
                    Dji.resize(6,6);
                    S.resize(6,6);
                    T.resize(6,6);
                    break;
                case 2:
                    Dij.resize(20,20);
                    S.resize(6,6);
                    break;
                case 3:
                    Dij.resize(45,45);
                    S.resize(6,6);
                    break;
                default:
                    throw std::runtime_error("2D ele_mat order not yet implemented");
                }
                for(size_t iq=0; iq< quadr->wq2.n_rows; iq++)
                {
                    arma::rowvec LocPnt =  RefGeo.row(0) +
                                           RefGeo.row(1)*quadr->xq2(iq,0) +
                                           RefGeo.row(2)*quadr->xq2(iq,1);
                    //arma::vec GlobPnt =  v0 + v1*quadr->xq2(iq,0) + v2*quadr->xq2(iq,1);
                    shape shp0(opt->p_ord, 3, shape::hcurl, LocPnt, cJac30);
                    shape shp1(opt->p_ord, 3, shape::hcurl, LocPnt, cJac31);
                    //arma::mat dshape(2,6);
                    for(size_t i=0; i < shp0.Nv.n_cols; i++)
                    {
                        //shape(0,i) = arma::dot(n, shp.dNv.col(i));
                        shp0.Nv.col(i) = arma::cross(arma::cross(n, shp0.Nv.col(i)),n);
                        shp1.Nv.col(i) = arma::cross(arma::cross(n, shp1.Nv.col(i)),n);
                        shp0.dNv.col(i) = -n% shp0.dNv.col(i);
                        shp1.dNv.col(i) = -n% shp1.dNv.col(i);
                    }
//                    S += arma::conv_to<arma::cx_mat>::from(shp.dNv.t()*shp.dNv)*quadr->wq2(iq);
                    //Dij += arma::conv_to<arma::cx_mat>::from(shp.Nv.t()*shp.dNv)*quadr->wq2(iq);
//                    Dji += arma::conv_to<arma::cx_mat>::from(shp.dNv.t()*shp.Nv)*quadr->wq2(iq);
//                    T += arma::conv_to<arma::cx_mat>::from(shp.Nv.t()*shp.Nv)*quadr->wq2(iq);
//                    Dij += arma::conv_to<arma::cx_mat>::from(shp.dNv.t()*shp.Nv + shp.Nv.t()*shp.dNv)*quadr->wq2(iq);
                    Dij += arma::conv_to<arma::cx_mat>::from(shp1.dNv.t()*shp0.Nv + shp0.Nv.t()*shp1.dNv)*quadr->wq2(iq);
                }
                Dij *= cJac2->detJ;
//                Dji *= cJac2->detJ;
//                S *= cJac2->detJ;
//                T *= cJac2->detJ;
                Dij = /*arma::diagmat*/(Dij.submat(map3to2,map3to2));
//                Dji = Dji.submat(map3to2,map3to2);
//                S = S.submat(map3to2,map3to2);
//                T = T.submat(map3to2,map3to2);
                // Dij: cross-derivative coupling matrix for DD transmission conditions
                for(int i=0; i<cdof1.v.n_rows; i++)
                {
                    for(int j=0; j<cdof1.v.n_rows; j++)
                    {
                        if(cdof1.v(i)<=cdof1.v(j))
                        {
                            sys.A_mat()(cdof1.v(i),cdof1.v(j)) += lMat.Tt(i,j) * std::complex<double>(0.0, k0);// +
                            //Dij(i,j) + Dji(j,i) + S(i,j) *std::complex<double>(0.0, 1.0/(k0));
                        }
                        sys.PR_mat()(cdof2.v(i),cdof1.v(j)) -= lMat.Tt(i,j) * std::complex<double>(0.0, k0) + Dij(i,j);// + Dji(i,j) - S(i,j);
                        // *std::complex<double>(0.0, 1.0/(k0));
                    }
                }
            }
            else
            {
                dof cdof1j(prj, 2, bndFaces(fif));
                cdof1j.v = BnddofmapJ(reg1).elem(cdof1j.v);
                dof cdof2j(prj, 2, bndFaces(fif));
                cdof2j.v = BnddofmapJ(reg2).elem(cdof2j.v);
                for(int i=0; i<cdof1.v.n_rows; i++)
                {
                    for(int j=0; j<cdof1.v.n_rows; j++)
                    {
                        if(cdof1.v(i)<=cdof1j.v(j))
                        {
                            sys.A_mat()(cdof1.v(i),cdof1j.v(j)) += lMat.Tt(i,j) * k0;// * std::sqrt(murd1*epsrd1);
                        }
                        if(cdof1j.v(i)<=cdof1.v(j))
                        {
                            sys.A_mat()(cdof1j.v(i),cdof1.v(j)) += lMat.Tt(j,i) * k0;// * std::sqrt(murd1*epsrd1);
                        }
                        if(cdof1j.v(i)<=cdof1j.v(j))
                        {
                            sys.A_mat()(cdof1j.v(i),cdof1j.v(j)) += lMat.Tt(i,j) * std::complex<double>(0.0, k0);
                        }
                        sys.PR_mat()(cdof1j.v(i),cdof2.v(j)) -= lMat.Tt(i,j) * k0;
                        sys.PR_mat()(cdof1j.v(i),cdof2j.v(j)) += lMat.Tt(i,j) * std::complex<double>(0.0, k0);
                    }
                }
            }
        }
    }
    mem_stat::print(logFile);
    // Rebuild A and PR from triplets to capture all assembly contributions
    sys.A_mat().sync();
    {
        arma::umat locsA(2, sys.A_mat().n_nonzero);
        arma::cx_vec valsA(sys.A_mat().n_nonzero);
        size_t k = 0;
        for(auto it = sys.A_mat().begin(); it != sys.A_mat().end(); ++it, ++k) {
            locsA(0,k) = it.row(); locsA(1,k) = it.col(); valsA(k) = *it;
        }
        sys.A_mat() = eq_sys::mat_row_type(locsA, valsA, sys.get_dofnum(), sys.get_dofnum(), true, true);
    }
    sys.PR_mat().sync();
    if(sys.PR_mat().n_nonzero > 0) {
        arma::umat locsP(2, sys.PR_mat().n_nonzero);
        arma::cx_vec valsP(sys.PR_mat().n_nonzero);
        size_t k = 0;
        for(auto it = sys.PR_mat().begin(); it != sys.PR_mat().end(); ++it, ++k) {
            locsP(0,k) = it.row(); locsP(1,k) = it.col(); valsP(k) = *it;
        }
        sys.PR_mat() = eq_sys::mat_row_type(locsP, valsP, sys.get_dofnum(), sys.get_dofnum(), true, true);
    }
    logFile << "Finishing:\n";
    lt.tic();
    if(sys.get_wave_ports_num() > 0)
{
        {
            // TFE modal expansion for DD
            arma::cx_mat Meig, MAcoeff;
            Meig = arma::cx_mat(sys.get_wave_ports_num(), sys.get_wave_ports_dofnum(), arma::fill::zeros);
            MAcoeff = arma::cx_mat(sys.get_wave_ports_num(), sys.get_wave_ports_num(), arma::fill::zeros);
            sys.B_mat().set_size( sys.get_wave_ports_num(), sys.get_wave_ports_num());
            arma::cx_mat B_exc(sys.get_wave_ports_num(), sys.get_wave_ports_num(), arma::fill::zeros);
            size_t idx = 0;
            size_t jdx = 0;
            for(size_t bcid = 0; bcid < msh->facbc.size(); bcid++)
            {
                bc* bc = &(msh->facbc[bcid]);
                if(bc->type == bc::wave_port)
                {
                    for(size_t i=0; i<bc->num_modes; i++)
                    {
                        std::complex<double> sqrtBeta(std::sqrt(std::complex<double>(0.0, k0*consts::z0*opt->power)/bc->mode_beta(i)));
                        MAcoeff(idx,idx) = std::complex<double>(0.0, k0*consts::z0*opt->power);
                        sys.B_mat()(idx,idx) = std::complex<double>(0.0, 2*k0*consts::z0*opt->power);
                        B_exc(idx,idx) = std::complex<double>(0.0, 2*k0*consts::z0*opt->power);
                        for(size_t j=0; j< bc->mode_vec.n_rows; j++)
                        {
                            Meig(idx,jdx+j) = sqrtBeta * bc->mode_vec(j,i);
                        }
                        idx++;
                    }
                    jdx += bc->mode_vec.n_rows;
                    for(size_t i=0; i<bc->mode_vecdof.size(); i++)
                    {
                        bc->mode_vecdof(i) = sys.Invdofmapv_vec()(bc->mode_vecdof(i));
                    }
                }
            }
            sys.Nonwave_portIds_vec().resize(sys.get_dofnum()-(sys.wave_portIds_vec().n_rows+sys.Dirdofv_vec().n_rows));
            sys.non_dir_ids_vec().resize(sys.get_dofnum()-(sys.wave_portIds_vec().n_rows+sys.Dirdofv_vec().n_rows));
            idx = 0;
            for(size_t i = 0; i<sys.get_dofnum(); i++)
            {
                if(doftoLeave[i] == true)
                {
                    sys.Nonwave_portIds_vec()(idx) = i;
                    sys.non_dir_ids_vec()[idx++] = i;
                }
            }
            sys.set_dofreal(sys.non_dir_ids_vec().size() + sys.get_wave_ports_num());
                std::cout << "\nSYS dof = " << sys.get_dofreal() << "\n";
            logFile << "\tSYS dof = " << sys.get_dofreal() << ", ";
            sys.B_mat().set_size( sys.get_dofreal(), sys.get_wave_ports_num());
            for(size_t bi=0; bi<sys.get_wave_ports_num(); bi++)
                for(size_t bj=0; bj<sys.get_wave_ports_num(); bj++)
                    if(std::abs(B_exc(bi,bj)) > 0) sys.B_mat()(bi,bj) = B_exc(bi,bj);
            std::vector<size_t> nnWPids(sys.non_dir_ids_vec().size());
            std::vector<size_t> WPids(sys.wave_portIds_vec().n_rows);
            for(size_t i=0; i<sys.non_dir_ids_vec().size(); i++)
            {
                nnWPids[i] = sys.non_dir_ids_vec()[i];
            }
            for(size_t i=0; i<sys.wave_portIds_vec().n_rows; i++)
            {
                WPids[i] = sys.wave_portIds_vec()(i);
            }
            size_t wnum = WPids.size(), nnum = nnWPids.size();
            std::unordered_map<size_t, size_t> nnWPids_map, WPids_map;
            for(size_t k=0; k<nnWPids.size(); k++) nnWPids_map[nnWPids[k]] = k;
            for(size_t k=0; k<WPids.size(); k++) WPids_map[WPids[k]] = k;

            // Step 1: Extract Aii and PRii BEFORE zeroing diagonal
            std::vector<arma::uword> aii_rows, aii_cols, prii_rows, prii_cols;
            std::vector<std::complex<double>> aii_vals, prii_vals;
            for(auto it = sys.A_mat().begin(); it != sys.A_mat().end(); ++it) {
                auto ri = nnWPids_map.find(it.row());
                if(ri == nnWPids_map.end()) continue;
                auto ci = nnWPids_map.find(it.col());
                if(ci == nnWPids_map.end()) continue;
                aii_rows.push_back(sys.get_wave_ports_num() + ri->second);
                aii_cols.push_back(sys.get_wave_ports_num() + ci->second);
                aii_vals.push_back(*it);
            }
            for(auto it = sys.PR_mat().begin(); it != sys.PR_mat().end(); ++it) {
                auto ri = nnWPids_map.find(it.row());
                if(ri == nnWPids_map.end()) continue;
                auto ci = nnWPids_map.find(it.col());
                if(ci == nnWPids_map.end()) continue;
                prii_rows.push_back(sys.get_wave_ports_num() + ri->second);
                prii_cols.push_back(sys.get_wave_ports_num() + ci->second);
                prii_vals.push_back(*it);
            }

            // Step 2: Save diagonal and zero it in A
            arma::cx_vec Adiag_vals(sys.get_dofnum(), arma::fill::zeros);
            for(size_t i=0; i<sys.get_dofnum(); i++) {
                Adiag_vals(i) = sys.A_mat()(i,i);
                sys.A_mat()(i,i) *= 0.0;
            }
            sys.A_mat().sync();

            // Step 3: Extract A_wp, A_wp_nn, A_nn_wp from zeroed-diagonal A
            arma::cx_mat A_wp(wnum, wnum, arma::fill::zeros);
            arma::cx_mat A_wp_nn(wnum, nnum, arma::fill::zeros);
            arma::cx_mat A_nn_wp(nnum, wnum, arma::fill::zeros);
            arma::cx_mat Ad_wp(wnum, wnum, arma::fill::zeros);
            for(auto it = sys.A_mat().begin(); it != sys.A_mat().end(); ++it) {
                size_t r = it.row(), c = it.col();
                std::complex<double> val = *it;
                auto wi = WPids_map.find(r);
                if(wi != WPids_map.end()) {
                    auto wj = WPids_map.find(c);
                    if(wj != WPids_map.end()) A_wp(wi->second, wj->second) += val;
                    auto ni = nnWPids_map.find(c);
                    if(ni != nnWPids_map.end()) A_wp_nn(wi->second, ni->second) += val;
                }
                auto ni = nnWPids_map.find(r);
                if(ni != nnWPids_map.end()) {
                    auto wj = WPids_map.find(c);
                    if(wj != WPids_map.end()) A_nn_wp(ni->second, wj->second) += val;
                }
            }
            for(size_t i=0; i<sys.get_dofnum(); i++) {
                auto wi = WPids_map.find(i);
                if(wi != WPids_map.end()) Ad_wp(wi->second, wi->second) = Adiag_vals(i);
            }

            // TFE modal reduction
            arma::cx_mat Attred = Meig * A_wp + Meig * A_wp.t() + Meig * Ad_wp;
            arma::cx_mat top_left = Attred * Meig.t() + MAcoeff;
            arma::cx_mat Atired = Meig * A_wp_nn + Meig * A_nn_wp.t();

            // Add top-left block (lower triangle, matching GMM convention)
            for(size_t i=0; i < sys.get_wave_ports_num(); i++)
                for(size_t j=0; j <= i; j++) {
                    aii_rows.push_back(i); aii_cols.push_back(j);
                    aii_vals.push_back(top_left(i,j));
                }
            // Add off-diagonal block
            for(size_t i=0; i < sys.get_wave_ports_num(); i++)
                for(size_t j=0; j < nnum; j++) {
                    aii_rows.push_back(i); aii_cols.push_back(sys.get_wave_ports_num() + j);
                    aii_vals.push_back(Atired(i,j));
                }

            // Build Anew and PRnew from triplets
            arma::umat locsA(2, aii_rows.size());
            arma::cx_vec cvsA(aii_vals.data(), aii_vals.size());
            for(size_t k=0; k<aii_rows.size(); k++) {
                locsA(0,k) = aii_rows[k]; locsA(1,k) = aii_cols[k];
            }
            eq_sys::mat_row_type Anew(locsA, cvsA, sys.get_dofreal(), sys.get_dofreal(), true, true);

            eq_sys::mat_row_type PRnew(sys.get_dofreal(), sys.get_dofreal());
            if(!prii_rows.empty()) {
                arma::umat locsP(2, prii_rows.size());
                arma::cx_vec cvsP(prii_vals.data(), prii_vals.size());
                for(size_t k=0; k<prii_rows.size(); k++) {
                    locsP(0,k) = prii_rows[k]; locsP(1,k) = prii_cols[k];
                }
                PRnew = eq_sys::mat_row_type(locsP, cvsP, sys.get_dofreal(), sys.get_dofreal(), true, true);
            }

            std::swap(sys.A_mat(), Anew);
            std::swap(sys.PR_mat(), PRnew);
            Anew.zeros();
            PRnew.zeros();
            Meig.zeros();
            MAcoeff.zeros();
            nnWPids.clear();
            WPids.clear();
            for(size_t i = 0; i<sys.non_dir_ids_vec().size(); i++)
            {
                sys.non_dir_ids_vec()[i] =  sys.Invdofmapv_vec()(sys.non_dir_ids_vec()[i]);
            }
            std::vector<size_t> newdoflevel;
            idx = 0;
            newdoflevel.push_back(idx);
            idx = sys.get_wave_ports_num();
            for(size_t irg=1; irg< sys.doflevel_vec().size(); irg++)
            {
                for(size_t rgid=sys.doflevel_vec()[irg-1]; rgid<sys.doflevel_vec()[irg]; rgid++)
                {
                    if(doftoLeave[rgid] == true)
                    {
                        idx++;
                    }
                }
                newdoflevel.push_back(idx);
            }
            sys.doflevel_vec() = newdoflevel;
        }
    }
    else if(opt->einc)
    {
        arma::vec kEinc(3), polEinc(3);
        kEinc(0) = opt->k[0];
        kEinc(1) = opt->k[1];
        kEinc(2) = opt->k[2];
        polEinc(0) = opt->E[0];
        polEinc(1) = opt->E[1];
        polEinc(2) = opt->E[2];
        kEinc *= k0;
        polEinc /= std::sqrt(2.0); // Vrms
        sys.set_dofreal(dofnum);
            std::cout << "\nSYS dof = " << sys.get_dofreal() << "\n";
        logFile << "\tSYS dof = " << sys.get_dofreal() << ", ";
        sys.B_mat().set_size( dofnum, 1);
        for(size_t bcid = 0; bcid < msh->facbc.size(); bcid++)
        {
            bc* bc = &(msh->facbc[bcid]);
            if(bc->type == bc::radiation)
            {
                for(size_t fid = 0; fid < bc->Faces.size(); fid++)
                {
                    arma::uvec adjTet = msh->facAdjTet(bc->Faces(fid));
                    double epsr = msh->tetmtrl[msh->tetLab(adjTet(0))].epsr;
                    double mur = msh->tetmtrl[msh->tetLab(adjTet(0))].mur;
                    ele_mat lMat(opt->p_ord, 2, msh->fac_geo(bc->Faces(fid)), quadr,
                                &(msh->tetmtrl[msh->tetLab(adjTet(0))]),
                                msh->int_node(bc->Faces(fid)), kEinc, polEinc);
                    dof cdof(prj, 2, bc->Faces(fid));
                    cdof.v = Domdofmap(msh->tetDom(adjTet(0))).elem(cdof.v);
                    for(int i=0; i<cdof.v.n_rows; i++)
                    {
                        sys.B_mat()(cdof.v(i),0) += lMat.f(i) * std::complex<double>(0.0,k0);
                    }
                }
            }
        }
        std::vector<size_t> DirIds(sys.Dirdofv_vec().size());
        for(size_t i=0; i<sys.Dirdofv_vec().size(); i++)
        {
            DirIds[i] = sys.Dirdofv_vec()(i);
        }
        for(size_t dirid : DirIds) {
            std::vector<size_t> colsToClear;
            for(auto it = sys.A_mat().begin_row(dirid); it != sys.A_mat().end_row(dirid); ++it)
                colsToClear.push_back(it.col());
            for(size_t col : colsToClear)
                sys.A_mat()(dirid, col) = std::complex<double>(0,0);
            std::vector<size_t> rowsToClear;
            for(auto it = sys.A_mat().begin_col(dirid); it != sys.A_mat().end_col(dirid); ++it)
                rowsToClear.push_back(it.row());
            for(size_t row : rowsToClear)
                sys.A_mat()(row, dirid) = std::complex<double>(0,0);
        }
        for(size_t i=0; i<DirIds.size(); i++)
        {
            sys.A_mat()(DirIds[i],DirIds[i]) = 1.0;
        }
    }
    if(sys.B_mat().n_nonzero == 0)
    {
        throw std::runtime_error("Null Right Hand Side");
    }
    logFile << " " << lt.toc() << " s\n";
    logFile << "+" << tt.toc() << " s\n";
}

