#include "assembler.h"
#include "equation_system.h"
#include "mesh.h"
#include "option.h"
#include "constants.h"
#include "memory.h"
#include "boundary_condition.h"
#include "quadrature.h"
#include "shape.h"
#include "degree_of_freedom.h"

#include <armadillo>
#include <iostream>
#include <map>
#include <set>
#include <unordered_map>

void assembler_em_ez_fd::assemble(std::ofstream& logFile, eq_sys& sys)
{
    mesh* msh = sys.msh;
    option* opt = sys.opt;
    logFile << "% Assembly 2D TMz:\n";
    arma::wall_clock tt, lt;
    tt.tic();

    double k0 = 2.0 * consts::pi * sys.get_freq() / consts::c0;
    double k0sq = k0 * k0;
    std::cout << "k0 = " << k0 << " ";

    // Material map
    std::map<size_t, const mtrl*> mtrlMap;
    for(const auto& m : msh->tetmtrl) mtrlMap[m.label] = &m;

    // ── Build full 2D edge connectivity (used by dofmap) ──
    // Reset first so the function always rebuilds from scratch.
    msh->facEdges.reset();
    msh->build_2d_edge_connectivity();

    // ── DOF numbering (shared dofmap) ──
    size_t totalDof = dof(sys.prj).dofnums;
    sys.set_dofnum(totalDof);
    sys.set_dofreal(totalDof);
    std::cout << "FE dof = " << totalDof << " ";
    sys.set_symm_flag(0);
    sys.B_mat().zeros(totalDof, 1);

    // ── Assembly (unified: same quadrature + shape for all p) ──
    lt.tic();
    std::map<std::pair<arma::uword, arma::uword>, std::complex<double>> A_map;
    quad quadr(opt->p_ord + 1);
    for(size_t t = 0; t < msh->nFaces; t++)
    {
        arma::mat triGeo(3, 2);
        for(int k = 0; k < 3; k++) {
            size_t nid = msh->facNodes(t, k);
            triGeo(k,0) = msh->nodPos(nid,0);
            triGeo(k,1) = msh->nodPos(nid,1);
        }
        jacobian cJac(2, triGeo);
        double area = 0.5 * std::abs(cJac.detJ);
        if(area < 1e-30) continue;

        double epsr = 1.0, mur = 1.0;
        size_t lab = msh->facLab(t);
        auto it = mtrlMap.find(lab);
        if(it != mtrlMap.end()) { epsr = it->second->epsr; mur = it->second->mur; }
        double invMur = 1.0 / mur;

        // Local-to-global DOF mapping via shared dofmap
        arma::uvec l2g = dof(sys.prj, 2, t).s;
        size_t nld = l2g.n_elem;

        arma::mat Se(nld,nld,arma::fill::zeros), Te(nld,nld,arma::fill::zeros);
        for(size_t iq = 0; iq < quadr.wq2.n_elem; iq++) {
            shape shp(opt->p_ord, 2, shape::hgrad, quadr.xq2.row(iq), &cJac);
            double w = quadr.wq2(iq) * std::abs(cJac.detJ);
            Se += arma::trans(shp.dNs) * shp.dNs * (w * invMur);
            Te += arma::trans(shp.Ns) * shp.Ns * (w * epsr);
        }
        for(size_t i = 0; i < nld; i++)
            for(size_t j = 0; j < nld; j++)
                A_map[{l2g(i), l2g(j)}] += std::complex<double>(Se(i,j) - k0sq * Te(i,j), 0.0);
    }

    // Build sparse A from map
    sys.A_mat() = build_sparse(A_map, totalDof);
    logFile << "\tAssembly: " << lt.toc() << " s\n";
    mem_stat::print(logFile);

    // ── bc type map ──
    std::map<size_t, bc::bcTYPE> bcTypeMap;
    for(const auto& bc : msh->facbc) bcTypeMap[bc.label] = bc.type;

    // ── Dirichlet DOFs ──
    size_t nV = msh->nNodes;
    size_t p1 = opt->p_ord > 1 ? opt->p_ord - 1 : 0;
    size_t nE = msh->nEdges; // total edges (after build_2d_edge_connectivity)
    std::set<size_t> dirNodeSet;
    for(size_t s = 0; s < msh->nEdges; s++) {
        size_t mkr = msh->edgLab(s);
        auto bcIt = bcTypeMap.find(mkr);
        if(bcIt != bcTypeMap.end() && bcIt->second == bc::perfect_e) {
            dirNodeSet.insert(msh->edgNodes(s,0));
            dirNodeSet.insert(msh->edgNodes(s,1));
            // All p_ord-1 edge DOF levels (blocked scheme from dofmap)
            for(size_t lev = 0; lev < p1; lev++)
                dirNodeSet.insert(nV + lev * nE + s);
        }
    }
    sys.Dirdofs_vec().set_size(dirNodeSet.size());
    size_t di = 0;
    for(auto n : dirNodeSet) sys.Dirdofs_vec()(di++) = n;

    // ── Check waveports ──
    size_t nwave_ports = 0;
    for(const auto& bc : msh->facbc) if(bc.type == bc::wave_port) nwave_ports++;

    if(nwave_ports > 0) {
    // ── BEGIN WAVEPORT ──
    logFile << "On boundaries:\n"; lt.tic();
    sys.set_wave_ports_num(0); sys.wave_portIds_vec().reset();
    std::set<size_t> wpNodeSet;

    // Port height geometry factor for power normalization
    double ymin=1e30, ymax=-1e30;
    for(size_t s=0; s<msh->nEdges; s++) {
        auto mk=msh->edgLab(s); auto bi=bcTypeMap.find(mk);
        if(bi!=bcTypeMap.end() && bi->second==bc::wave_port) {
            double y0=msh->nodPos(msh->edgNodes(s,0),1);
            double y1=msh->nodPos(msh->edgNodes(s,1),1);
            if(y0<ymin) ymin=y0; if(y1<ymin) ymin=y1;
            if(y0>ymax) ymax=y0; if(y1>ymax) ymax=y1;
        }
    }
    double pH = ymax-ymin;
    double wp_pow = (pH>1e-12) ? 4.0/pH : 1.0;
    // Combined power factor: geometry base x user-specified power
    double pfac = wp_pow * opt->power;

    for(size_t bcid=0; bcid<msh->facbc.size(); bcid++) {
        bc* bc = &msh->facbc[bcid];
        if(bc->type != bc::wave_port) continue;
        std::cout<<bc->name<<" ";
        logFile<<"\t"<<bc->name<<": ";

        std::vector<size_t> pn; std::set<size_t> pns;
        for(size_t s=0; s<msh->nEdges; s++) {
            if(msh->edgLab(s)!=bc->label) continue;
            size_t n0=msh->edgNodes(s,0), n1=msh->edgNodes(s,1);
            if(pns.insert(n0).second) pn.push_back(n0);
            if(pns.insert(n1).second) pn.push_back(n1);
            wpNodeSet.insert(n0); wpNodeSet.insert(n1);
            if(opt->p_ord>1) {
                for(size_t lev = 0; lev < p1; lev++) {
                    size_t mp = nV + lev * nE + s;
                    if(pns.insert(mp).second) pn.push_back(mp);
                }
            }
        }
        if(pn.empty()) { logFile<<"0 nodes\n"; continue; }

        size_t np = pn.size();
        std::unordered_map<size_t,size_t> li;
        for(size_t i=0;i<np;i++) li[pn[i]]=i;

        arma::mat St(np,np,arma::fill::zeros), Tt(np,np,arma::fill::zeros);
        for(size_t s=0; s<msh->nEdges; s++) {
            if(msh->edgLab(s)!=bc->label) continue;
            size_t n0=msh->edgNodes(s,0), n1=msh->edgNodes(s,1);
            double dx=msh->nodPos(n1,0)-msh->nodPos(n0,0);
            double dy=msh->nodPos(n1,1)-msh->nodPos(n0,1);
            double L=std::sqrt(dx*dx+dy*dy), iL=1.0/L;
            size_t i0=li[n0], i1=li[n1];
            St(i0,i0)+=iL; St(i0,i1)-=iL; St(i1,i0)-=iL; St(i1,i1)+=iL;
            Tt(i0,i0)+=L/3; Tt(i0,i1)+=L/6; Tt(i1,i0)+=L/6; Tt(i1,i1)+=L/3;
            if(p1 > 0) {
                size_t mid = nV + s;
                auto mi=li.find(mid);
                if(mi!=li.end()) {
                    size_t im=mi->second;
                    static const double gx[3]={-0.7745966692414834,0,0.7745966692414834};
                    static const double gw[3]={0.5555555555555556,0.8888888888888888,0.5555555555555556};
                    for(int q=0;q<3;q++) {
                        double t=gx[q], N0=t*(t-1)/2, N1=t*(t+1)/2, N2=(1-t)*(1+t);
                        double d0=(2*t-1)/2, d1=(2*t+1)/2, d2=-2*t;
                        double w=gw[q]*L/2, f=4/(L*L);
                        St(i0,i0)+=d0*d0*f*w; St(i0,i1)+=d0*d1*f*w; St(i0,im)+=d0*d2*f*w;
                        St(i1,i0)+=d1*d0*f*w; St(i1,i1)+=d1*d1*f*w; St(i1,im)+=d1*d2*f*w;
                        St(im,i0)+=d2*d0*f*w; St(im,i1)+=d2*d1*f*w; St(im,im)+=d2*d2*f*w;
                        Tt(i0,i0)+=N0*N0*w; Tt(i0,i1)+=N0*N1*w; Tt(i0,im)+=N0*N2*w;
                        Tt(i1,i0)+=N1*N0*w; Tt(i1,i1)+=N1*N1*w; Tt(i1,im)+=N1*N2*w;
                        Tt(im,i0)+=N2*N0*w; Tt(im,i1)+=N2*N1*w; Tt(im,im)+=N2*N2*w;
                    }
                }
            }
        }

        std::vector<size_t> fl;
        for(size_t i=0;i<np;i++) if(dirNodeSet.find(pn[i])==dirNodeSet.end()) fl.push_back(i);
        size_t nf=fl.size();
        if(nf<2) {
            logFile<<"Warning: "<<nf<<" free DOF(s) on port \""<<bc->name<<"\" — fewer than 2 required for any mode. Increase mesh density (h-refine or +p) for more port DOFs.\n";
            bc->mode_beta.zeros(); bc->mode_vec.reset(); continue;
        }

        arma::mat Sf(nf,nf), Tf(nf,nf);
        for(size_t i=0;i<nf;i++) for(size_t j=0;j<nf;j++) { Sf(i,j)=St(fl[i],fl[j]); Tf(i,j)=Tt(fl[i],fl[j]); }

        arma::cx_vec ev; arma::cx_mat evec;
        if(!arma::eig_pair(ev,evec,Sf,Tf)) {
            logFile<<"eig_pair fail\n"; bc->mode_beta.zeros(); bc->mode_vec.reset(); continue;
        }
        arma::uvec ord = arma::sort_index(arma::real(ev));
        ev=ev(ord); evec=evec.cols(ord);

        int nm=bc->num_modes; if(nm>(int)ev.n_elem) nm=(int)ev.n_elem;
        if(nm < bc->num_modes)
            logFile<<"Warning: requested "<<bc->num_modes<<" mode(s) but only "<<nm<<" computable from "<<nf<<" free DOF(s). Increase mesh density or reduce num_modes.\n";

        arma::cx_vec be(nm); arma::cx_mat mv(nf,nm);
        for(int m=0; m<nm; m++) {
            double lam=std::real(ev(m));
            be(m)=std::sqrt(std::complex<double>(k0sq-lam,0));
            mv.col(m)=evec.col(m);
            size_t br=0; double bv=-1;
            for(size_t r=0; r<nf; r++) { double av=std::abs(std::real(mv(r,m)));
                if(av>bv){bv=av;br=r;} }
            if(std::real(mv(br,m))<0) mv.col(m)*=-1.0;
        }
        for(int m=0; m<nm; m++) {
            double nrm=std::sqrt(std::abs(arma::as_scalar(mv.col(m).t()*Tf*mv.col(m))));
            if(nrm>1e-15) mv.col(m)*=(1.0/nrm);
        }

        logFile<<nm<<" modes, ";
        bc->mode_beta=be; bc->num_modes=nm;
        bc->mode_vec.set_size(np,nm); bc->mode_vec.zeros();
        for(int m=0;m<nm;m++) for(size_t k=0;k<nf;k++) bc->mode_vec(fl[k],m)=mv(k,m);
        bc->mode_vecdof.set_size(np);
        for(size_t k=0;k<np;k++) bc->mode_vecdof(k)=pn[k];
        sys.set_wave_ports_num(sys.get_wave_ports_num() + nm);
        sys.wave_portIds_vec()=arma::join_cols(sys.wave_portIds_vec(),bc->mode_vecdof);
    }
    logFile<<lt.toc()<<" s\n";

    // ── TFE block system ──
    if(sys.get_wave_ports_num()>0) {
        std::cout<<"\nTFE ("<<sys.get_wave_ports_num()<<" modes)\n";
        lt.tic();
        std::vector<size_t> dirV;
        for(size_t i=0;i<sys.Dirdofs_vec().n_elem;i++) dirV.push_back(sys.Dirdofs_vec()(i));
        std::set<size_t> dirS(dirV.begin(),dirV.end());

        std::vector<size_t> pWP; std::set<size_t> pWS;
        for(size_t i=0;i<sys.wave_portIds_vec().n_rows;i++) {
            size_t nid=sys.wave_portIds_vec()(i);
            if(nid>=sys.get_dofnum()) continue;
            if(dirS.find(nid)==dirS.end() && pWS.insert(nid).second) pWP.push_back(nid);
        }
        std::vector<size_t> nnV;
        for(size_t i=0;i<sys.get_dofnum();i++)
            if(dirS.find(i)==dirS.end() && pWS.find(i)==pWS.end()) nnV.push_back(i);

        size_t nW=pWP.size(), nN=nnV.size();
        std::unordered_map<size_t,size_t> wL;
        for(size_t i=0;i<nW;i++) wL[pWP[i]]=i;

        arma::cx_mat Mg(sys.get_wave_ports_num(),nW,arma::fill::zeros);
        arma::cx_mat MC(sys.get_wave_ports_num(),sys.get_wave_ports_num(),arma::fill::zeros);
        arma::cx_mat BX(sys.get_wave_ports_num(),sys.get_wave_ports_num(),arma::fill::zeros);
        size_t mi=0;
        for(auto& bc : msh->facbc) {
            if(bc.type!=bc::wave_port || bc.mode_beta.n_elem==0) continue;
            for(int m=0; m<(int)bc.mode_beta.n_elem; m++) {
                double ba=std::abs(bc.mode_beta(m));
                double sc=std::sqrt(k0*consts::z0*pfac/ba);
                MC(mi,mi)=std::complex<double>(0,k0*consts::z0*pfac);
                BX(mi,mi)=std::complex<double>(0,2*k0*consts::z0*pfac);
                for(size_t j=0;j<bc.mode_vecdof.n_elem;j++) {
                    auto wi=wL.find(bc.mode_vecdof(j));
                    if(wi!=wL.end() && wi->second<nW) Mg(mi,wi->second)=sc*bc.mode_vec(j,m);
                }
                mi++;
            }
        }

        arma::cx_mat Aw(nW,nW,arma::fill::zeros), Awn(nW,nN,arma::fill::zeros), Ann(nN,nN,arma::fill::zeros);
        for(size_t i=0;i<nW;i++) for(size_t j=0;j<nW;j++)
            if(pWP[i]<sys.get_dofnum() && pWP[j]<sys.get_dofnum()) Aw(i,j)=sys.A_mat()(pWP[i],pWP[j]);
        for(size_t i=0;i<nW;i++) for(size_t j=0;j<nN;j++)
            if(pWP[i]<sys.get_dofnum() && nnV[j]<sys.get_dofnum()) Awn(i,j)=sys.A_mat()(pWP[i],nnV[j]);
        for(size_t i=0;i<nN;i++) for(size_t j=0;j<nN;j++)
            if(nnV[i]<sys.get_dofnum() && nnV[j]<sys.get_dofnum()) Ann(i,j)=sys.A_mat()(nnV[i],nnV[j]);

        arma::cx_mat TL = (nW>0) ? (Mg*Aw*Mg.t()+MC) : MC;
        arma::cx_mat PI = (nW>0) ? (Mg*Awn) : arma::cx_mat(sys.get_wave_ports_num(),nN,arma::fill::zeros);

        size_t sD=sys.get_wave_ports_num()+nN;
        sys.set_dofreal(sD);
        std::vector<arma::uword> aR,aC; std::vector<std::complex<double>> aV;
        for(size_t i=0;i<sys.get_wave_ports_num();i++) for(size_t j=0;j<sys.get_wave_ports_num();j++)
            if(std::abs(TL(i,j))>0){aR.push_back(i);aC.push_back(j);aV.push_back(TL(i,j));}
        for(size_t i=0;i<sys.get_wave_ports_num();i++) for(size_t j=0;j<nN;j++)
            if(std::abs(PI(i,j))>0){aR.push_back(i);aC.push_back(sys.get_wave_ports_num()+j);aV.push_back(PI(i,j));}
        for(size_t i=0;i<nN;i++) for(size_t j=0;j<sys.get_wave_ports_num();j++)
            if(std::abs(PI(j,i))>0){aR.push_back(sys.get_wave_ports_num()+i);aC.push_back(j);aV.push_back(std::conj(PI(j,i)));}
        for(size_t i=0;i<nN;i++) for(size_t j=0;j<nN;j++)
            if(std::abs(Ann(i,j))>0){aR.push_back(sys.get_wave_ports_num()+i);aC.push_back(sys.get_wave_ports_num()+j);aV.push_back(Ann(i,j));}

        if(!aV.empty()) {
            arma::umat lo(2,aR.size()); arma::cx_vec cv(aV.data(),aV.size());
            for(size_t k=0;k<aR.size();k++){lo(0,k)=aR[k];lo(1,k)=aC[k];}
            eq_sys::mat_row_type An(lo,cv,sD,sD,true,false); std::swap(sys.A_mat(),An);
        }
        sys.B_mat().set_size(sD,sys.get_wave_ports_num()); sys.B_mat().zeros();
        for(size_t i=0;i<sys.get_wave_ports_num();i++) for(size_t j=0;j<sys.get_wave_ports_num();j++)
            if(std::abs(BX(i,j))>0) sys.B_mat()(i,j)=BX(i,j);
        sys.Nonwave_portIds_vec().set_size(nN);
        for(size_t i=0;i<nN;i++) sys.Nonwave_portIds_vec()(i)=nnV[i];
        logFile<<"\tTFE: "<<sD<<" DOF "<<sys.get_wave_ports_num()<<" modes "<<lt.toc()<<" s\n";
    }
    // ── END WAVEPORT ──
    } else {
        logFile << "On boundaries:\n"; lt.tic();
        if(sys.Dirdofs_vec().n_elem > 0) {
            arma::cx_vec Bcol0 = arma::cx_vec(sys.B_mat().col(0));
            arma::cx_vec Bnew = sys.A_mat() * Bcol0;
            for(size_t i = 0; i < sys.Dirdofs_vec().n_elem; i++) {
                size_t id = sys.Dirdofs_vec()(i); Bnew(id) = 0;
                std::vector<size_t> rc, cr;
                for(auto it=sys.A_mat().begin_row(id); it!=sys.A_mat().end_row(id); ++it) rc.push_back(it.col());
                for(size_t c : rc) sys.A_mat()(id,c)=0;
                for(auto it=sys.A_mat().begin_col(id); it!=sys.A_mat().end_col(id); ++it) cr.push_back(it.row());
                for(size_t r : cr) sys.A_mat()(r,id)=0;
            }
            for(size_t i = 0; i < sys.Dirdofs_vec().n_elem; i++) {
                size_t id = sys.Dirdofs_vec()(i);
                Bcol0(id) -= Bnew(id); sys.A_mat()(id,id) = 1.0;
            }
            for(size_t i = 0; i < sys.get_dofnum(); i++)
                if(Bcol0(i) != std::complex<double>(0,0)) sys.B_mat()(i,0) = Bcol0(i);
        }
        logFile << " " << lt.toc() << " s\n";
    }
    logFile << "+" << tt.toc() << " s\n";
}
