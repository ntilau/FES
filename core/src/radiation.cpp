#include "radiation.h"
#include "degree_of_freedom.h"
#include "shape.h"
#include "constants.h"


rad::rad(project* prj, arma::cx_mat& sol, double freq, double& Pacc) : prj(prj), pfreq(freq), quadr(new quad(prj->opt->p_ord)),
    n_theta(prj->opt->n_theta), n_phi(prj->opt->n_phi), Pacc(Pacc), Prad(0.0), nDirOrGain(false)
{
    size_t nDirs = n_theta*n_phi;
    arma::cx_vec fullSol = arma::sum(sol,1) * std::sqrt(2);
    arma::vec thetas = arma::linspace<arma::vec>(0.0, 180.0, n_theta);
    arma::vec phis = arma::linspace<arma::vec>(0.0, 360.0, n_phi);
    Ef.resize(n_theta*n_phi,3);
    Ef.fill(0);
    Dir.resize(n_theta*n_phi,3);
    Dir.fill(0);
    std::complex<double> Hconst(0.0, -1.0 / (2.0*consts::pi*freq*consts::mu0));
    double kv = 2.0*consts::pi*freq/consts::c0;
    //arma::cx_vec Efar(3);
    //Efar.fill(0);
    for(size_t bcid = 0; bcid < prj->msh->facbc.size(); bcid++)
    {
        bc* bc = &(prj->msh->facbc[bcid]);
        if(bc->type == bc::radiation)
        {
                std::cout << bc->name;
            for(size_t fid = 0; fid < bc->Faces.size(); fid++)
            {
                arma::uvec adjTet = prj->msh->facAdjTet(bc->Faces(fid));
                mtrl* cmtrl = &(prj->msh->tetmtrl[prj->msh->tetLab(adjTet(0))]);
                std::complex<double> epsr(cmtrl->epsr, cmtrl->calc_epsr2(freq));
                double mur = cmtrl->mur;
                dof cdof(prj, 3, adjTet(0));
                jacobian* cJac3 = new jacobian(3, prj->msh->tet_geo(adjTet(0)));
                arma::cx_vec tSol = fullSol.elem(cdof.v);
                arma::mat cGeo = prj->msh->fac_geo(bc->Faces(fid));
                size_t RefFace = 0;
                arma::mat int_node = prj->msh->int_node(bc->Faces(fid), RefFace);
                arma::mat RefGeo(3,3);
                arma::cx_vec RefNorm(3);
                RefGeo.fill(0);
                RefNorm.fill(0);
                switch(RefFace)
                {
                case 0:
                    RefGeo(0,0) = 1.0;
                    RefGeo(1,0) = -1.0;
                    RefGeo(1,1) = 1.0;
                    RefGeo(2,0) = -1.0;
                    RefGeo(2,2) = 1.0;
                    break;
                case 1:
                    RefGeo(1,1) = 1.0;
                    RefGeo(2,2) = 1.0;
                    break;
                case 2:
                    RefGeo(1,0) = 1.0;
                    RefGeo(2,2) = 1.0;
                    break;
                case 3:
                    RefGeo(1,0) = 1.0;
                    RefGeo(2,1) = 1.0;
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
                RefNorm(0) = -n(0);
                RefNorm(1) = -n(1);
                RefNorm(2) = -n(2);
                for(size_t iq=0; iq< quadr->wq2.n_rows; iq++)
                {
                    arma::rowvec LocPnt =  RefGeo.row(0) +
                                           RefGeo.row(1)*quadr->xq2(iq,0) +
                                           RefGeo.row(2)*quadr->xq2(iq,1);
                    arma::vec GlobPnt =  v0 + v1*quadr->xq2(iq,0) + v2*quadr->xq2(iq,1);
                    shape shp(prj->opt->p_ord, 3, shape::hcurl, LocPnt, cJac3);
                    arma::cx_vec Jm = arma::cross(RefNorm, shp.Nv*tSol);
                    arma::cx_vec Je = arma::cross(RefNorm,Hconst*shp.dNv*tSol);
                    Prad += 0.5 * (std::real(arma::dot(arma::cross(Jm,arma::conj(Je)), RefNorm))) *
                            quadr->wq2(iq) * cJac2->detJ;
                    for(size_t it=0; it<thetas.n_rows; it++)
                    {
                        double theta = thetas(it);
                        for(size_t ip=0; ip<phis.n_rows; ip++)
                        {
                            double phi = phis(ip);
                            arma::vec rv(3);
                            rv(0) = std::sin(theta/180.0*consts::pi)*std::cos(phi/180.0*consts::pi);
                            rv(1) = std::sin(theta/180.0*consts::pi)*std::sin(phi/180.0*consts::pi);
                            rv(2) = std::cos(theta/180.0*consts::pi);
                            Dir(it*n_phi+ip, 0) = rv(0);
                            Dir(it*n_phi+ip, 1) = rv(1);
                            Dir(it*n_phi+ip, 2) = rv(2);
                            arma::cx_vec rDir = Dir.row(it*n_phi+ip).st();
                            std::complex<double> G = std::exp(std::complex<double>(0.0,1.0)*
                                                              std::sqrt(mur*epsr)*kv*arma::dot(rv, GlobPnt));
                            G *= std::complex<double>(0.0,1.0)*std::sqrt(mur*epsr)*kv/4.0/consts::pi;
                            Ef.row(it*n_phi+ip) -= (G*(consts::z0*std::sqrt(mur/epsr)*(arma::dot(Je,rDir)*rDir - Je) -
                                                      arma::cross(Jm,rDir)) * quadr->wq2(iq) * cJac2->detJ).st();
                        }
                    }
                }
                delete cJac2;
                delete cJac3;
            }
        }
            std::cout << " ";
    }
        std::cout << "\n";
    Savefield();
}

void rad::Savefield()
{
    Prad = std::abs(Prad);
    arma::vec magE(Ef.n_rows);
    for(size_t i=0; i<Ef.n_rows; i++)
    {
        magE(i) = arma::norm(Ef.row(i),2);
    }
    double delta = 20.0*std::log10(magE.max() - magE.min());
    double GainCoeff, DirCoeff;
    if(nDirOrGain)
    {
        GainCoeff = 2.0*consts::pi/Pacc/consts::z0;
        DirCoeff = 2.0*consts::pi/Prad/consts::z0;
            std::cout << "Pacc = " << Pacc << " W\n";
            std::cout << "Prad = " << Prad << " W\n";
            std::cout << "maxGain = " << 10.0*std::log10(GainCoeff * std::pow(magE.max(),2)) << " dB\n";
            std::cout << "maxDir = " << 10.0*std::log10(DirCoeff * std::pow(magE.max(),2)) << " dB\n";
    }
    else
    {
        if(prj->opt->einc)
        {
            arma::vec kEinc(3), polEinc(3);
            kEinc(0) = prj->opt->k[0];
            kEinc(1) = prj->opt->k[1];
            kEinc(2) = prj->opt->k[2];
            polEinc(0) = prj->opt->E[0];
            polEinc(1) = prj->opt->E[1];
            polEinc(2) = prj->opt->E[2];
            DirCoeff = std::pow(2.0*std::sqrt(consts::pi)/arma::norm(polEinc,2),2);
            delta = -10.0*std::log10(DirCoeff * std::pow(magE.min(),2));
        }
        else
        {
            DirCoeff = 2.0*consts::pi/Prad/consts::z0;
        }
            std::cout << "Pinc = " << Pacc << " W\n";
            std::cout << "Prad = " << Prad << " W\n";
            std::cout << "maxDir = " << 10.0*std::log10(DirCoeff * std::pow(magE.max(),2)) << " dB\n";
    }
    std::stringstream tmp;
    tmp << pfreq;
    std::ofstream outfield(std::string(prj->opt->name + "_" + tmp.str() + "_rad.vtk").data());
    outfield << "# vtk DataFile Version 2.0\n";
    outfield << "radiation data\n";
    outfield << "ASCII\n";
    outfield << "DATASET UNSTRUCTURED_GRID\n";
    outfield << "POINTS " << Dir.n_rows << " float \n";
    for(size_t i= 0; i < Dir.n_rows; i++)
    {
        double rMag = 10.0*std::log10(DirCoeff*std::pow(arma::norm(Ef.row(i),2),2)) + delta;
        outfield << (float) rMag* std::real(Dir(i,0)) << " ";
        outfield << (float) rMag* std::real(Dir(i,1)) << " ";
        outfield << (float) rMag* std::real(Dir(i,2)) << "\n";
    }
    outfield << "CELLS " << (n_theta-1)*(n_phi-1) << " " << (n_theta-1)*(n_phi-1)*5 << "\n";
    for(size_t i = 0; i < n_theta-1; i++)
    {
        for(size_t j = 0; j < n_phi-1; j++)
        {
            outfield << 4 << " ";
            outfield << (i)*n_phi+j << " " << (i)*n_phi+j+1 << " ";
            outfield << (i+1)*n_phi+j+1 << " " << (i+1)*n_phi+j << " ";
        }
    }
    outfield << "CELL_TYPES " << (n_theta-1)*(n_phi-1) << "\n";
    for(size_t i = 0; i < (n_theta-1)*(n_phi-1); i++)
    {
        outfield << 7 << "\n";
    }
    outfield << "POINT_DATA " << Ef.n_rows  << "\n";
    if(prj->opt->einc)
    {
        outfield << "SCALARS RCS_[dB] float 1\n";
    }
    else
    {
        outfield << "SCALARS Dir_[dB] float 1\n";
    }
    outfield << "LOOKUP_TABLE jet\n";
    for(size_t i = 0; i < Ef.n_rows; i++)
    {
        outfield << (float) 10.0*std::log10(DirCoeff * std::pow(arma::norm(Ef.row(i),2),2)) << "\n";
    }
    if(nDirOrGain)
    {
        //outfield << "POINT_DATA " << Ef.n_rows  << "\n";
        outfield << "SCALARS Gain_[dB] float 1\n";
        outfield << "LOOKUP_TABLE jet\n";
        for(size_t i = 0; i < Ef.n_rows; i++)
        {
            outfield << (float) 10.0*std::log10(GainCoeff * std::pow(arma::norm(Ef.row(i),2),2)) << "\n";
        }
    }
    outfield << "SCALARS E_far_norm_[V/m] float 1\n";
    outfield << "LOOKUP_TABLE jet\n";
    for(size_t i = 0; i < Ef.n_rows; i++)
    {
        outfield << (float) arma::norm(arma::abs(Ef.row(i)),2) << "\n";
    }
    outfield << "VECTORS E_far_abs_[V/m] float\n";
    for(size_t i = 0; i < Ef.n_rows; i++)
    {
        outfield << (float) std::abs(Ef(i,0)) << " ";
        outfield << (float) std::abs(Ef(i,1)) << " ";
        outfield << (float) std::abs(Ef(i,2)) << "\n";
    }
    outfield << "VECTORS E_far_real_[V/m] float\n";
    for(size_t i = 0; i < Ef.n_rows; i++)
    {
        outfield << (float) std::real(Ef(i,0)) << " ";
        outfield << (float) std::real(Ef(i,1)) << " ";
        outfield << (float) std::real(Ef(i,2)) << "\n";
    }
    outfield << "VECTORS E_far_imag_[V/m] float\n";
    for(size_t i = 0; i < Ef.n_rows; i++)
    {
        outfield << (float) std::imag(Ef(i,0)) << " ";
        outfield << (float) std::imag(Ef(i,1)) << " ";
        outfield << (float) std::imag(Ef(i,2)) << "\n";
    }
}


rad::~rad()
{
}
