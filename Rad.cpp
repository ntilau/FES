#include "Rad.h"
#include "DoF.h"
#include "Shape.h"
#include "Const.h"


Rad::Rad(Project* prj, arma::cx_mat& sol, double& freq, double& Pacc) : prj(prj), pfreq(freq), quad(new Quad(prj->opt->pOrd)),
    nTheta(prj->opt->nTheta), nPhi(prj->opt->nPhi), Pacc(Pacc), Prad(0.0), nDirOrGain(prj->opt->sparam)
{
    size_t nDirs = nTheta*nPhi;
    arma::cx_vec fullSol = arma::sum(sol,1) * std::sqrt(2);
    arma::vec thetas = arma::linspace<arma::vec>(0.0, 180.0, nTheta);
    arma::vec phis = arma::linspace<arma::vec>(0.0, 360.0, nPhi);
    Ef.resize(nTheta*nPhi,3);
    Ef.fill(0);
    Dir.resize(nTheta*nPhi,3);
    Dir.fill(0);
    std::complex<double> Hconst(0.0, -1.0 / (2.0*Const::pi*freq*Const::mu0));
    double kv = 2.0*Const::pi*freq/Const::c0;
    //arma::cx_vec Efar(3);
    //Efar.fill(0);
    for(size_t bcid = 0; bcid < prj->msh->facBC.size(); bcid++)
    {
        BC* bc = &(prj->msh->facBC[bcid]);
        if(bc->type == BC::Radiation)
        {
            if(prj->opt->verbose)
            {
                std::cout << bc->name;
            }
            #pragma omp parallel for
            for(size_t fid = 0; fid < bc->Faces.size(); fid++)
            {
                arma::uvec adjTet = prj->msh->facAdjTet(bc->Faces(fid));
                Mtrl* cMtrl = &(prj->msh->tetMtrl[prj->msh->tetLab(adjTet(0))]);
                std::complex<double> epsr(cMtrl->epsr, cMtrl->CalcEpsr2(freq));
                double mur = cMtrl->mur;
                DoF cDoF(prj, 3, adjTet(0));
                Jacobian* cJac3 = new Jacobian(3, prj->msh->tetGeo(adjTet(0)));
                arma::cx_vec tSol = fullSol.elem(cDoF.v);
                arma::mat cGeo = prj->msh->facGeo(bc->Faces(fid));
                size_t RefFace = 0;
                arma::mat intNode = prj->msh->intNode(bc->Faces(fid), RefFace);
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
                Jacobian* cJac2 = new Jacobian(2, cGeo2);
                RefNorm(0) = -n(0);
                RefNorm(1) = -n(1);
                RefNorm(2) = -n(2);
                for(size_t iq=0; iq< quad->wq2.n_rows; iq++)
                {
                    arma::rowvec LocPnt =  RefGeo.row(0) +
                                           RefGeo.row(1)*quad->xq2(iq,0) +
                                           RefGeo.row(2)*quad->xq2(iq,1);
                    arma::vec GlobPnt =  v0 + v1*quad->xq2(iq,0) + v2*quad->xq2(iq,1);
                    Shape shp(prj->opt->pOrd, 3, Shape::Hcurl, LocPnt, cJac3);
                    arma::cx_vec Jm = arma::cross(RefNorm, shp.Nv*tSol);
                    arma::cx_vec Je = arma::cross(RefNorm,Hconst*shp.dNv*tSol);
                    Prad += 0.5 * (std::real(arma::dot(arma::cross(Jm,arma::conj(Je)), RefNorm))) *
                            quad->wq2(iq) * cJac2->detJ;
                    for(size_t it=0; it<thetas.n_rows; it++)
                    {
                        double theta = thetas(it);
                        for(size_t ip=0; ip<phis.n_rows; ip++)
                        {
                            double phi = phis(ip);
                            arma::vec rv(3);
                            rv(0) = std::sin(theta/180.0*Const::pi)*std::cos(phi/180.0*Const::pi);
                            rv(1) = std::sin(theta/180.0*Const::pi)*std::sin(phi/180.0*Const::pi);
                            rv(2) = std::cos(theta/180.0*Const::pi);
                            Dir(it*nPhi+ip, 0) = rv(0);
                            Dir(it*nPhi+ip, 1) = rv(1);
                            Dir(it*nPhi+ip, 2) = rv(2);
                            arma::cx_vec rDir = Dir.row(it*nPhi+ip).st();
                            std::complex<double> G = std::exp(std::complex<double>(0.0,1.0)*
                                                              std::sqrt(mur*epsr)*kv*arma::dot(rv, GlobPnt));
                            G *= std::complex<double>(0.0,1.0)*std::sqrt(mur*epsr)*kv/4.0/Const::pi;
                            Ef.row(it*nPhi+ip) -= (G*(Const::z0*std::sqrt(mur/epsr)*(arma::dot(Je,rDir)*rDir - Je) -
                                                      arma::cross(Jm,rDir)) * quad->wq2(iq) * cJac2->detJ).st();
                        }
                    }
                }
                delete cJac2;
                delete cJac3;
            }
        }
        if(prj->opt->verbose)
        {
            std::cout << " ";
        }
    }
    if(prj->opt->verbose)
    {
        std::cout << "\n";
    }
    SaveField();
}

void Rad::SaveField()
{
    Prad = std::abs(Prad);
    arma::vec magE(Ef.n_rows);
    #pragma omp parallel for
    for(size_t i=0; i<Ef.n_rows; i++)
    {
        magE(i) = arma::norm(Ef.row(i),2);
    }
    double delta = 20.0*std::log10(magE.max() - magE.min());
    double GainCoeff, DirCoeff;
    if(nDirOrGain)
    {
        GainCoeff = 2.0*Const::pi/Pacc/Const::z0;
        DirCoeff = 2.0*Const::pi/Prad/Const::z0;
        if(prj->opt->verbose)
        {
            std::cout << "Pacc = " << Pacc << " W\n";
            std::cout << "Prad = " << Prad << " W\n";
            std::cout << "maxGain = " << 10.0*std::log10(GainCoeff * std::pow(magE.max(),2)) << " dB\n";
            std::cout << "maxDir = " << 10.0*std::log10(DirCoeff * std::pow(magE.max(),2)) << " dB\n";
        }
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
            DirCoeff = std::pow(2.0*std::sqrt(Const::pi)/arma::norm(polEinc,2),2);
            delta = -10.0*std::log10(DirCoeff * std::pow(magE.min(),2));
        }
        else
        {
            DirCoeff = 2.0*Const::pi/Prad/Const::z0;
        }
        if(prj->opt->verbose)
        {
            std::cout << "Pinc = " << Pacc << " W\n";
            std::cout << "Prad = " << Prad << " W\n";
            std::cout << "maxDir = " << 10.0*std::log10(DirCoeff * std::pow(magE.max(),2)) << " dB\n";
        }
    }
    std::stringstream tmp;
    tmp << pfreq;
    std::ofstream outField(std::string(prj->opt->name + "_" + tmp.str() + "_Rad.vtk").data());
    outField << "# vtk DataFile Version 2.0\n";
    outField << "Radiation data\n";
    outField << "ASCII\n";
    outField << "DATASET UNSTRUCTURED_GRID\n";
    outField << "POINTS " << Dir.n_rows << " float \n";
    for(size_t i= 0; i < Dir.n_rows; i++)
    {
        double rMag = 10.0*std::log10(DirCoeff*std::pow(arma::norm(Ef.row(i),2),2)) + delta;
        outField << (float) rMag* std::real(Dir(i,0)) << " ";
        outField << (float) rMag* std::real(Dir(i,1)) << " ";
        outField << (float) rMag* std::real(Dir(i,2)) << "\n";
    }
    outField << "CELLS " << (nTheta-1)*(nPhi-1) << " " << (nTheta-1)*(nPhi-1)*5 << "\n";
    for(size_t i = 0; i < nTheta-1; i++)
    {
        for(size_t j = 0; j < nPhi-1; j++)
        {
            outField << 4 << " ";
            outField << (i)*nPhi+j << " " << (i)*nPhi+j+1 << " ";
            outField << (i+1)*nPhi+j+1 << " " << (i+1)*nPhi+j << " ";
        }
    }
    outField << "CELL_TYPES " << (nTheta-1)*(nPhi-1) << "\n";
    for(size_t i = 0; i < (nTheta-1)*(nPhi-1); i++)
    {
        outField << 7 << "\n";
    }
    outField << "POINT_DATA " << Ef.n_rows  << "\n";
    if(prj->opt->einc)
    {
        outField << "SCALARS RCS_[dB] float 1\n";
    }
    else
    {
        outField << "SCALARS Dir_[dB] float 1\n";
    }
    outField << "LOOKUP_TABLE jet\n";
    for(size_t i = 0; i < Ef.n_rows; i++)
    {
        outField << (float) 10.0*std::log10(DirCoeff * std::pow(arma::norm(Ef.row(i),2),2)) << "\n";
    }
    if(nDirOrGain)
    {
        //outField << "POINT_DATA " << Ef.n_rows  << "\n";
        outField << "SCALARS Gain_[dB] float 1\n";
        outField << "LOOKUP_TABLE jet\n";
        for(size_t i = 0; i < Ef.n_rows; i++)
        {
            outField << (float) 10.0*std::log10(GainCoeff * std::pow(arma::norm(Ef.row(i),2),2)) << "\n";
        }
    }
    outField << "SCALARS E_far_norm_[V/m] float 1\n";
    outField << "LOOKUP_TABLE jet\n";
    for(size_t i = 0; i < Ef.n_rows; i++)
    {
        outField << (float) arma::norm(arma::abs(Ef.row(i)),2) << "\n";
    }
    outField << "VECTORS E_far_abs_[V/m] float\n";
    for(size_t i = 0; i < Ef.n_rows; i++)
    {
        outField << (float) std::abs(Ef(i,0)) << " ";
        outField << (float) std::abs(Ef(i,1)) << " ";
        outField << (float) std::abs(Ef(i,2)) << "\n";
    }
    outField << "VECTORS E_far_real_[V/m] float\n";
    for(size_t i = 0; i < Ef.n_rows; i++)
    {
        outField << (float) std::real(Ef(i,0)) << " ";
        outField << (float) std::real(Ef(i,1)) << " ";
        outField << (float) std::real(Ef(i,2)) << "\n";
    }
    outField << "VECTORS E_far_imag_[V/m] float\n";
    for(size_t i = 0; i < Ef.n_rows; i++)
    {
        outField << (float) std::imag(Ef(i,0)) << " ";
        outField << (float) std::imag(Ef(i,1)) << " ";
        outField << (float) std::imag(Ef(i,2)) << "\n";
    }
}


Rad::~Rad()
{
}
