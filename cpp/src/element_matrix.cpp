#include "element_matrix.h"
#include <stdexcept>
#include "coupling.h"
#include "constants.h"

ele_mat::ele_mat(size_t p_ord, size_t cDim, arma::mat cGeo, quad* quad, mtrl* cmtrl, shape::s_type sType)
{
    size_t nums, numv;
    switch(cDim)
    {
    case 3:
        cJac = new jacobian(3, cGeo);
        switch(p_ord)
        {
        case 1:
            nums = 4;
            numv = 6;
            break;
        case 2:
            nums = 10;
            numv = 20;
            break;
        case 3:
            nums = 20;
            numv = 45;
            break;
        case 4:
            nums = 35;
            numv = 84;
            break;
        default:
            throw std::runtime_error("3D ele_mat order not yet implemented");
        }
        switch(sType)
        {
        case shape::hcurl:
            S.resize(numv,numv);
            T.resize(numv,numv);
            for(size_t iq=0; iq<quad->wq3.n_rows; iq++)
            {
                shape shphcurl(p_ord, 3, shape::hcurl, quad->xq3.row(iq), cJac);
                S += arma::conv_to<arma::cx_mat>::from(shphcurl.dNv.t()*shphcurl.dNv)*quad->wq3(iq);
                T += arma::conv_to<arma::cx_mat>::from(shphcurl.Nv.t()*shphcurl.Nv)*quad->wq3(iq);
            }
            Z = cJac->detJ * T * std::complex<double>(0.0,consts::z0*cmtrl->sigma);
            S *= cJac->detJ / cmtrl->mur;
            T *= cJac->detJ * std::complex<double>(cmtrl->epsr,cmtrl->epsr2);
            break;
        case shape::hgrad:
            S.resize(nums,nums);
            for(size_t iq=0; iq<quad->wq3.n_rows; iq++)
            {
                shape shphgrad(p_ord, 3, shape::hgrad, quad->xq3.row(iq), cJac);
                S += arma::conv_to<arma::cx_mat>::from(shphgrad.dNs.t()*shphgrad.dNs)*quad->wq3(iq);
            }
            S *= cJac->detJ * std::complex<double>(cmtrl->epsr,0.0);
            break;
        case shape::hdiv:
        {
            // H(div) element matrix: grad-div form + mass
            // S = div(v)*div(u),  T = v·u
            // Override numv for hdiv (different dimension from hcurl/nédélec)
            size_t hdiv_nv = 0;
            switch(p_ord) {
                case 1: hdiv_nv = 4; break;  // RT0: 4 face functions (3D tetrahedron)
                default: throw std::runtime_error("hdiv 3D order not yet implemented in ele_mat");
            }
            S.resize(hdiv_nv, hdiv_nv);
            T.resize(hdiv_nv, hdiv_nv);
            for(size_t iq=0; iq<quad->wq3.n_rows; iq++)
            {
                shape shphdiv(p_ord, 3, shape::hdiv, quad->xq3.row(iq), cJac);
                S += arma::conv_to<arma::cx_mat>::from(shphdiv.divNv.t()*shphdiv.divNv)*quad->wq3(iq);
                T += arma::conv_to<arma::cx_mat>::from(shphdiv.Nv.t()*shphdiv.Nv)*quad->wq3(iq);
            }
            S *= cJac->detJ;
            T *= cJac->detJ;
            break;
        }
        }
        delete cJac;
        break;
    case 2:
        cJac = new jacobian(2, cGeo);
        switch(p_ord)
        {
        case 1:
            St.resize(3,3);
            Tt.resize(3,3);
            Sz.resize(3,3);
            Tz.resize(3,3);
            G.resize(3,3);
            break;
        case 2:
            St.resize(8,8);
            Tt.resize(8,8);
            Sz.resize(6,6);
            Tz.resize(6,6);
            G.resize(8,6);
            break;
        case 3:
            St.resize(15,15);
            Tt.resize(15,15);
            Sz.resize(10,10);
            Tz.resize(10,10);
            G.resize(15,10);
            break;
        case 4:
            St.resize(24,24);
            Tt.resize(24,24);
            Sz.resize(15,15);
            Tz.resize(15,15);
            G.resize(24,15);
            break;
        default:
            throw std::runtime_error("2D ele_mat order not yet implemented");
        }
        for(size_t iq=0; iq<quad->wq2.n_rows; iq++)
        {
            shape shp(p_ord, 2, shape::hcurl, quad->xq2.row(iq), cJac);
            St += arma::conv_to<arma::cx_mat>::from(shp.dNv.t()*shp.dNv)*quad->wq2(iq);
            Tt += arma::conv_to<arma::cx_mat>::from(shp.Nv.t()*shp.Nv)*quad->wq2(iq);
            Sz += arma::conv_to<arma::cx_mat>::from(shp.dNs.t()*shp.dNs)*quad->wq2(iq);
            Tz += arma::conv_to<arma::cx_mat>::from(shp.Ns.t()*shp.Ns)*quad->wq2(iq);
            G += arma::conv_to<arma::cx_mat>::from(shp.Nv.t()*shp.dNs)*quad->wq2(iq);
        }
        St *= cJac->detJ;
        Tt *= cJac->detJ;
        Sz *= cJac->detJ;
        Tz *= cJac->detJ;
        G *= cJac->detJ;
        delete cJac;
        break;
    }
}


ele_mat::ele_mat(size_t p_ord, size_t cDim, arma::mat cGeo, quad* quad, mtrl* cmtrl, arma::vec int_node)
{
    if(cDim != 2)
    {
        throw std::runtime_error("ele_mat requested only for faces");
    }
    arma::vec v0 = cGeo.row(0).t();
    arma::vec v1 = cGeo.row(1).t();
    arma::vec v2 = cGeo.row(2).t();
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
    cJac = new jacobian(2, cGeo2);
    switch(p_ord)
    {
    case 1:
        St.resize(3,3);
        Tt.resize(3,3);
        Sz.resize(3,3);
        Tz.resize(3,3);
        G.resize(3,3);
        STt.resize(3,3);
        SSt.resize(3,3);
        break;
    case 2:
        St.resize(8,8);
        Tt.resize(8,8);
        Sz.resize(6,6);
        Tz.resize(6,6);
        G.resize(8,6);
        STt.resize(8,8);
        break;
    case 3:
        St.resize(15,15);
        Tt.resize(15,15);
        Sz.resize(10,10);
        Tz.resize(10,10);
        G.resize(15,10);
        STt.resize(15,15);
        break;
    case 4:
        St.resize(24,24);
        Tt.resize(24,24);
        Sz.resize(15,15);
        Tz.resize(15,15);
        G.resize(24,15);
        STt.resize(24,24);
        break;
    default:
        throw std::runtime_error("2D ele_mat order not yet implemented");
    }
    for(size_t iq=0; iq<quad->wq2.n_rows; iq++)
    {
        shape shp(p_ord, 2, shape::hcurl, quad->xq2.row(iq), cJac);
        St += arma::conv_to<arma::cx_mat>::from(shp.dNv.st()*shp.dNv)*quad->wq2(iq);
        Tt += arma::conv_to<arma::cx_mat>::from(shp.Nv.t()*shp.Nv)*quad->wq2(iq);
        Sz += arma::conv_to<arma::cx_mat>::from(shp.dNs.t()*shp.dNs)*quad->wq2(iq);
        Tz += arma::conv_to<arma::cx_mat>::from(shp.Ns.t()*shp.Ns)*quad->wq2(iq);
        G += arma::conv_to<arma::cx_mat>::from(shp.Nv.t()*shp.dNs)*quad->wq2(iq);
        STt += arma::conv_to<arma::cx_mat>::from(arma::diagmat(shp.Nv.t()*shp.Nv))*quad->wq2(iq);
    }
    St *= cJac->detJ;
    Tt *= cJac->detJ;
    Sz *= cJac->detJ;
    Tz *= cJac->detJ;
    G *= cJac->detJ;
    STt *= cJac->detJ;
    delete cJac;
}


ele_mat::ele_mat(size_t p_ord, size_t cDim, arma::mat cGeo, quad* quad, mtrl* cmtrl,
               arma::vec int_node, arma::vec kEinc, arma::vec polEinc)
{
    if(cDim != 2)
    {
        throw std::runtime_error("ele_mat requested only for faces");
    }
    arma::vec v0 = cGeo(0,arma::span::all).t();
    arma::vec v1 = cGeo(1,arma::span::all).t();
    arma::vec v2 = cGeo(2,arma::span::all).t();
    arma::vec vIn = ((v0+v1+v2)/3) - int_node;
    v1 -= v0;
    v2 -= v0;
    arma::vec u = v1 / arma::norm(v1,2);
    arma::vec n = arma::cross(v1,v2);
    n *= arma::dot(n,vIn);
    n /= arma::norm(n,2);
    arma::vec v = arma::cross(n,u);
    arma::vec k = kEinc/arma::norm(kEinc,2);
    arma::mat cGeo2(3,2);
    cGeo2.fill(0);
    cGeo2(1,0) = arma::dot((cGeo.row(1)-cGeo.row(0)).t(), u);
    cGeo2(2,0) = arma::dot((cGeo.row(2)-cGeo.row(0)).t(), u);
    cGeo2(2,1) = arma::dot((cGeo.row(2)-cGeo.row(0)).t(), v);
    cJac = new jacobian(2, cGeo2);
    switch(p_ord)
    {
    case 1:
        f.resize(3,1);
        break;
    case 2:
        f.resize(8,1);
        break;
    case 3:
        f.resize(15,1);
        break;
    case 4:
        f.resize(24,1);
        break;
    default:
        throw std::runtime_error("fEinc ele_mat order not yet implemented");
    }
    for(size_t iq=0; iq<quad->wq2.n_rows; iq++)
    {
        shape shp(p_ord, 2, shape::hcurl, quad->xq2.row(iq), cJac);
        arma::vec rho = v0 + quad->xq2(iq,0)*v1 + quad->xq2(iq,1)*v2 ;
        arma::vec incPol = polEinc - arma::cross(-n,arma::cross(k,polEinc));
        arma::cx_vec vEinc(2);
        vEinc.fill(std::exp(std::complex<double>(0.0,-arma::dot(kEinc,rho))));
        vEinc(0) *= arma::dot(incPol,u);
        vEinc(1) *= arma::dot(incPol,v);
        f += arma::conv_to<arma::cx_mat>::from(shp.Nv.t()*vEinc)*quad->wq2(iq);
    }
    f *= cJac->detJ;
    delete cJac;
}

ele_mat::~ele_mat()
{
    S.clear();
    T.clear();
    Z.clear();
    St.clear();
    Tt.clear();
    Sz.clear();
    Tz.clear();
    G.clear();
    f.clear();
}

// eps(E) = eps(0) + eps(0) kerr E^2
ele_mat::ele_mat(size_t p_ord, size_t cDim, arma::mat cGeo, quad* quad, mtrl* cmtrl,
               dof* cdof, arma::cx_vec cSol, size_t n_harm, size_t dofnumv, double mFreq)
{
    size_t numv;
    coupl* cCpl;
    shape* cShp;
    arma::mat centroid(1,3);
    centroid.fill(0.25);
    arma::vec normE(n_harm);
    std::complex<double> epsr(cmtrl->epsr,cmtrl->epsr2);
    std::complex<double> kerr(cmtrl->epsr,0.0);
    kerr *= cmtrl->kerr;
    switch(cDim)
    {
    case 3:
        cJac = new jacobian(3, cGeo);
        switch(p_ord)
        {
        case 1:
            numv = 6;
            break;
        case 2:
            numv = 20;
            break;
        case 3:
            numv = 45;
            break;
        case 4:
            numv = 84;
            break;
        default:
            throw std::runtime_error("3D ele_mat order not yet implemented");
        }
        S.resize(numv,numv);
        T.resize(numv,numv);
        for(size_t iq=0; iq<quad->wq3.n_rows; iq++)
        {
            shape shp(p_ord, 3, shape::hcurl, quad->xq3.row(iq), cJac);
            S += arma::conv_to<arma::cx_mat>::from(shp.dNv.t()*shp.dNv)*quad->wq3(iq);
            T += arma::conv_to<arma::cx_mat>::from(shp.Nv.t()*shp.Nv)*quad->wq3(iq);
        }
        Z = T * cJac->detJ * std::complex<double>(0.0,consts::z0*cmtrl->sigma);
        S *= cJac->detJ / cmtrl->mur;
        T *= cJac->detJ;
        S = repmat(S, n_harm, n_harm);
        T = repmat(T, n_harm, n_harm);
        Z = repmat(Z, n_harm, n_harm);
        cShp = new shape(p_ord, 3, shape::hcurl, centroid.row(0), cJac);
        for(size_t ih=0; ih < n_harm; ih++)
        {
            arma::cx_vec tSol = cSol.elem(cdof->v.rows(ih*numv,(ih+1)*numv-1));
            normE(ih) = arma::norm(cShp->Nv*tSol,2);
        }
        cCpl = new coupl(n_harm, epsr, kerr, normE);
        for(size_t ih=0; ih < n_harm; ih++)
        {
            for(size_t jh=0; jh < n_harm; jh++)
            {
                S.submat(ih*numv, jh*numv, (ih+1)*numv-1, (jh+1)*numv-1) *= cCpl->N(ih,jh);
                T.submat(ih*numv, jh*numv, (ih+1)*numv-1, (jh+1)*numv-1) *= cCpl->D(ih,jh);// +
                //cCpl->N(ih,jh) * std::complex<double>(0.0,cmtrl->epsr2);
                Z.submat(ih* numv, jh* numv, (ih+1)*numv-1, (jh+1)*numv-1) *= cCpl->N(ih,jh);
            }
        }
        delete cShp;
        delete cCpl;
        delete cJac;
        break;
    case 2:
        cJac = new jacobian(2, cGeo);
        switch(p_ord)
        {
        case 1:
            St.resize(3,3);
            Tt.resize(3,3);
            Sz.resize(3,3);
            Tz.resize(3,3);
            G.resize(3,3);
            break;
        case 2:
            St.resize(8,8);
            Tt.resize(8,8);
            Sz.resize(6,6);
            Tz.resize(6,6);
            G.resize(8,6);
            break;
        case 3:
            St.resize(15,15);
            Tt.resize(15,15);
            Sz.resize(10,10);
            Tz.resize(10,10);
            G.resize(15,10);
            break;
        case 4:
            St.resize(24,24);
            Tt.resize(24,24);
            Sz.resize(15,15);
            Tz.resize(15,15);
            G.resize(24,15);
            break;
        default:
            throw std::runtime_error("2D ele_mat order not yet implemented");
        }
        St.fill(0);
        Tt.fill(0);
        Sz.fill(0);
        Tz.fill(0);
        G.fill(0);
        for(size_t iq=0; iq<quad->wq2.n_rows; iq++)
        {
            shape shp(p_ord, 2, shape::hcurl, quad->xq2.row(iq), cJac);
            St += arma::conv_to<arma::cx_mat>::from(shp.dNv.t()*shp.dNv)*quad->wq2(iq);
            Tt += arma::conv_to<arma::cx_mat>::from(shp.Nv.t()*shp.Nv)*quad->wq2(iq);
            Sz += arma::conv_to<arma::cx_mat>::from(shp.dNs.t()*shp.dNs)*quad->wq2(iq);
            Tz += arma::conv_to<arma::cx_mat>::from(shp.Ns.t()*shp.Ns)*quad->wq2(iq);
            G += arma::conv_to<arma::cx_mat>::from(shp.Nv.t()*shp.dNs)*quad->wq2(iq);
        }
        St *= cJac->detJ;
        Tt *= cJac->detJ;
        Sz *= cJac->detJ;
        Tz *= cJac->detJ;
        G *= cJac->detJ;
        delete cJac;
        break;
    }
}
