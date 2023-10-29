#include "EleMat.h"
#include "Coupl.h"
#include "Const.h"

EleMat::EleMat(size_t pOrd, size_t cDim, arma::mat cGeo, Quad* quad, Mtrl* cMtrl, Shape::STYPE sType)
{
    size_t nums, numv;
    switch(cDim)
    {
    case 3:
        cJac = new Jacobian(3, cGeo);
        switch(pOrd)
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
        default:
            throw std::string("3D EleMat order not yet implemented");
        }
        switch(sType)
        {
        case Shape::Hcurl:
            S.resize(numv,numv);
            T.resize(numv,numv);
            for(size_t iq=0; iq<quad->wq3.n_rows; iq++)
            {
                Shape shpHcurl(pOrd, 3, Shape::Hcurl, quad->xq3.row(iq), cJac);
                S += arma::conv_to<arma::cx_mat>::from(shpHcurl.dNv.t()*shpHcurl.dNv)*quad->wq3(iq);
                T += arma::conv_to<arma::cx_mat>::from(shpHcurl.Nv.t()*shpHcurl.Nv)*quad->wq3(iq);
            }
            Z = cJac->detJ * T * std::complex<double>(0.0,Const::z0*cMtrl->sigma);
            S *= cJac->detJ / cMtrl->mur;
            T *= cJac->detJ * std::complex<double>(cMtrl->epsr,cMtrl->epsr2);
            break;
        case Shape::Hgrad:
            S.resize(nums,nums);
            for(size_t iq=0; iq<quad->wq3.n_rows; iq++)
            {
                Shape shpHgrad(pOrd, 3, Shape::Hgrad, quad->xq3.row(iq), cJac);
                S += arma::conv_to<arma::cx_mat>::from(shpHgrad.dNs.t()*shpHgrad.dNs)*quad->wq3(iq);
            }
            S *= cJac->detJ * std::complex<double>(cMtrl->epsr,0.0);
            break;
        }
        delete cJac;
        break;
    case 2:
        cJac = new Jacobian(2, cGeo);
        switch(pOrd)
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
        default:
            throw std::string("2D EleMat order not yet implemented");
        }
        for(size_t iq=0; iq<quad->wq2.n_rows; iq++)
        {
            Shape shp(pOrd, 2, Shape::Hcurl, quad->xq2.row(iq), cJac);
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


EleMat::EleMat(size_t pOrd, size_t cDim, arma::mat cGeo, Quad* quad, Mtrl* cMtrl, arma::vec intNode)
{
    if(cDim != 2)
    {
        throw std::string("EleMat requested only for faces");
    }
    arma::vec v0 = cGeo.row(0).t();
    arma::vec v1 = cGeo.row(1).t();
    arma::vec v2 = cGeo.row(2).t();
    arma::vec vIn = ((v0+v1+v2)/3.0) - intNode;
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
    cJac = new Jacobian(2, cGeo2);
    switch(pOrd)
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
    default:
        throw std::string("2D EleMat order not yet implemented");
    }
    for(size_t iq=0; iq<quad->wq2.n_rows; iq++)
    {
        Shape shp(pOrd, 2, Shape::Hcurl, quad->xq2.row(iq), cJac);
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


EleMat::EleMat(size_t pOrd, size_t cDim, arma::mat cGeo, Quad* quad, Mtrl* cMtrl,
               arma::vec intNode, arma::vec kEinc, arma::vec polEinc)
{
    if(cDim != 2)
    {
        throw std::string("EleMat requested only for faces");
    }
    arma::vec v0 = cGeo(0,arma::span::all).t();
    arma::vec v1 = cGeo(1,arma::span::all).t();
    arma::vec v2 = cGeo(2,arma::span::all).t();
    arma::vec vIn = ((v0+v1+v2)/3) - intNode;
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
    cJac = new Jacobian(2, cGeo2);
    switch(pOrd)
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
    default:
        throw std::string("fEinc EleMat order not yet implemented");
    }
//    f.fill(0);
    for(size_t iq=0; iq<quad->wq2.n_rows; iq++)
    {
        Shape shp(pOrd, 2, Shape::Hcurl, quad->xq2.row(iq), cJac);
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

EleMat::~EleMat()
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
EleMat::EleMat(size_t pOrd, size_t cDim, arma::mat cGeo, Quad* quad, Mtrl* cMtrl,
               DoF* cDoF, arma::cx_vec cSol, size_t nHarm, size_t DoFnumv, double mFreq)
{
    size_t numv;
    Coupl* cCpl;
    Shape* cShp;
    arma::mat centroid(1,3);
    centroid.fill(0.25);
    arma::vec normE(nHarm);
    std::complex<double> epsr(cMtrl->epsr,cMtrl->epsr2);
    std::complex<double> kerr(cMtrl->epsr,0.0);
    kerr *= cMtrl->kerr;
    switch(cDim)
    {
    case 3:
        cJac = new Jacobian(3, cGeo);
        switch(pOrd)
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
        default:
            throw std::string("3D EleMat order not yet implemented");
        }
        S.resize(numv,numv);
        T.resize(numv,numv);
        for(size_t iq=0; iq<quad->wq3.n_rows; iq++)
        {
            Shape shp(pOrd, 3, Shape::Hcurl, quad->xq3.row(iq), cJac);
            S += arma::conv_to<arma::cx_mat>::from(shp.dNv.t()*shp.dNv)*quad->wq3(iq);
            T += arma::conv_to<arma::cx_mat>::from(shp.Nv.t()*shp.Nv)*quad->wq3(iq);
        }
        Z = T * cJac->detJ * std::complex<double>(0.0,Const::z0*cMtrl->sigma);
        S *= cJac->detJ / cMtrl->mur;
        T *= cJac->detJ;
        S = repmat(S, nHarm, nHarm);
        T = repmat(T, nHarm, nHarm);
        Z = repmat(Z, nHarm, nHarm);
        cShp = new Shape(pOrd, 3, Shape::Hcurl, centroid.row(0), cJac);
        for(size_t ih=0; ih < nHarm; ih++)
        {
            arma::cx_vec tSol = cSol.elem(cDoF->v.rows(ih*numv,(ih+1)*numv-1));
            normE(ih) = arma::norm(cShp->Nv*tSol,2);
        }
        cCpl = new Coupl(nHarm, epsr, kerr, normE);
        for(size_t ih=0; ih < nHarm; ih++)
        {
            for(size_t jh=0; jh < nHarm; jh++)
            {
                S.submat(ih*numv, jh*numv, (ih+1)*numv-1, (jh+1)*numv-1) *= cCpl->N(ih,jh);
                T.submat(ih*numv, jh*numv, (ih+1)*numv-1, (jh+1)*numv-1) *= cCpl->D(ih,jh);// +
                //cCpl->N(ih,jh) * std::complex<double>(0.0,cMtrl->epsr2);
                Z.submat(ih* numv, jh* numv, (ih+1)*numv-1, (jh+1)*numv-1) *= cCpl->N(ih,jh);
            }
        }
        delete cShp;
        delete cCpl;
        delete cJac;
        break;
    case 2:
        cJac = new Jacobian(2, cGeo);
        switch(pOrd)
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
        default:
            throw std::string("2D EleMat order not yet implemented");
        }
        St.fill(0);
        Tt.fill(0);
        Sz.fill(0);
        Tz.fill(0);
        G.fill(0);
        for(size_t iq=0; iq<quad->wq2.n_rows; iq++)
        {
            Shape shp(pOrd, 2, Shape::Hcurl, quad->xq2.row(iq), cJac);
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
