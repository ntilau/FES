#ifndef GMRES_H
#define GMRES_H

#include <gmm.h>
#include <cmumps_c.h>
#include <zmumps_c.h>
#include "Option.h"

template <typename Matrix, typename VecI, typename V1, typename V2> inline
void JacobiDbl(const Matrix& A, const VecI& Blocks, const V1& v1, V2& v2)
{
    typedef gmm::row_matrix<gmm::rsvector<std::complex<double> > > MatType;
    typedef gmm::rsvector<std::complex<double> > VecType;
    gmm::copy(v1, v2);
    for(size_t did=0; did < Blocks.size()-1; did++)
    {
        ZMUMPS_STRUC_C* idz = new ZMUMPS_STRUC_C;
        idz->job=-1; // init
        idz->par=1;
        idz->sym=(MUMPS_INT) 1;
        idz->comm_fortran=-987654; // use_comm_world
        zmumps_c(idz);
        idz->n = (MUMPS_INT)(Blocks[did+1]-Blocks[did]);
        idz->nz = (MUMPS_INT) gmm::nnz(gmm::sub_matrix(A, gmm::sub_interval(Blocks[did], Blocks[did+1]-Blocks[did])));
        idz->irn = new MUMPS_INT [idz->nz];
        idz->jcn = new MUMPS_INT [idz->nz];
        idz->a = new ZMUMPS_COMPLEX [idz->nz];
        typename gmm::linalg_traits<MatType>::const_row_iterator it = mat_row_const_begin(A);
        typename gmm::linalg_traits<MatType>::const_row_iterator ite = mat_row_const_end(A);
        size_t i = 0,  idx = 0;
        for(; it != ite; ++it)
        {
            if((i >= Blocks[did]) && (i < Blocks[did+1]))
            {
                typename gmm::linalg_traits<MatType>::const_sub_row_type row = gmm::linalg_traits<MatType>::row(it);
                typename gmm::linalg_traits<VecType>::const_iterator itr = vect_const_begin(row);
                typename gmm::linalg_traits<VecType>::const_iterator itre = vect_const_end(row);
                for(; itr != itre; ++itr)
                {
                    if((itr.index() >= Blocks[did]) && (itr.index() < Blocks[did+1]) && (itr.index() >= i))
                    {
                        idz->a[idx].r = (ZMUMPS_REAL) itr->real();
                        idz->a[idx].i = (ZMUMPS_REAL) itr->imag();
                        idz->irn[idx] = (MUMPS_INT)(i-Blocks[did])+1;
                        idz->jcn[idx++] = (MUMPS_INT)(itr.index()-Blocks[did])+1;
                    }
                }
            }
            i++;
        }
        idz->icntl[0]=-1; //-1 6
        idz->icntl[1]=-1; //-1 6
        idz->icntl[2]=-1; //-1 1
        idz->icntl[3]=0; //0 6
        idz->rhs = new ZMUMPS_COMPLEX [idz->n];
        #pragma omp parallel for
        for(size_t row = 0; row < (size_t) idz->n; row++)
        {
            idz->rhs[row].r = (ZMUMPS_REAL) gmm::real(v2[Blocks[did]+row]);
            idz->rhs[row].i = (ZMUMPS_REAL) gmm::imag(v2[Blocks[did]+row]);
        }
        idz->job=6; // factor and solve
        zmumps_c(idz);
        #pragma omp parallel for
        for(size_t row = 0; row < (size_t) idz->n; row++)
        {
            v2[Blocks[did]+row] = std::complex<double>(idz->rhs[row].r, idz->rhs[row].i);
        }
        idz->job=-2; // end
        zmumps_c(idz);
        delete idz->a;
        delete idz->rhs;
        delete idz->irn;
        delete idz->jcn;
        delete idz;
    }
}


template <typename Matrix, typename VecI, typename V1, typename V2> inline
void GaussSeidelDbl(const Matrix& A, const Matrix& PR, const VecI& Blocks, const V1& v1, V2& v2)
{
    typedef gmm::row_matrix<gmm::rsvector<std::complex<double> > > MatType;
    typedef gmm::rsvector<std::complex<double> > VecType;
    gmm::copy(v1, v2);
    for(size_t did=0; did < Blocks.size()-1; did++)
    {
        ZMUMPS_STRUC_C* idz = new ZMUMPS_STRUC_C;
        idz->job=-1; // init
        idz->par=1;
        idz->sym=(MUMPS_INT) 1;
        idz->comm_fortran=-987654; // use_comm_world
        zmumps_c(idz);
        idz->n = (MUMPS_INT)(Blocks[did+1]-Blocks[did]);
        idz->nz = (MUMPS_INT) gmm::nnz(gmm::sub_matrix(A, gmm::sub_interval(Blocks[did], Blocks[did+1]-Blocks[did])));
        idz->irn = new MUMPS_INT [idz->nz];
        idz->jcn = new MUMPS_INT [idz->nz];
        idz->a = new ZMUMPS_COMPLEX [idz->nz];
        typename gmm::linalg_traits<MatType>::const_row_iterator it = mat_row_const_begin(A);
        typename gmm::linalg_traits<MatType>::const_row_iterator ite = mat_row_const_end(A);
        size_t i = 0,  idx = 0;
        for(; it != ite; ++it)
        {
            if((i >= Blocks[did]) && (i < Blocks[did+1]))
            {
                typename gmm::linalg_traits<MatType>::const_sub_row_type row = gmm::linalg_traits<MatType>::row(it);
                typename gmm::linalg_traits<VecType>::const_iterator itr = vect_const_begin(row);
                typename gmm::linalg_traits<VecType>::const_iterator itre = vect_const_end(row);
                for(; itr != itre; ++itr)
                {
                    if((itr.index() >= Blocks[did]) && (itr.index() < Blocks[did+1]) && (itr.index() >= i))
                    {
                        idz->a[idx].r = (ZMUMPS_REAL) itr->real();
                        idz->a[idx].i = (ZMUMPS_REAL) itr->imag();
                        idz->irn[idx] = (MUMPS_INT)(i-Blocks[did])+1;
                        idz->jcn[idx++] = (MUMPS_INT)(itr.index()-Blocks[did])+1;
                    }
                }
            }
            i++;
        }
        idz->icntl[0]=-1; //-1 6
        idz->icntl[1]=-1; //-1 6
        idz->icntl[2]=-1; //-1 1
        idz->icntl[3]=0; //0 6
        //
        V2 tmp(idz->n);
        for(size_t lil = 0; lil < did; lil++)
        {
            gmm::mult_add(gmm::sub_matrix(PR, gmm::sub_interval(Blocks[did], Blocks[did+1]-Blocks[did]),
                                          gmm::sub_interval(Blocks[lil], Blocks[lil+1]-Blocks[lil])),
                          gmm::sub_vector(v2, gmm::sub_interval(Blocks[lil], Blocks[lil+1]-Blocks[lil])),
                          tmp);
        }
        gmm::add(gmm::scaled(tmp,-1.0), gmm::sub_vector(v2, gmm::sub_interval(Blocks[did], Blocks[did+1]-Blocks[did])));
        //
        idz->rhs = new ZMUMPS_COMPLEX [idz->n];
        #pragma omp parallel for
        for(size_t row = 0; row < (size_t) idz->n; row++)
        {
            idz->rhs[row].r = (ZMUMPS_REAL) gmm::real(v2[Blocks[did]+row]);
            idz->rhs[row].i = (ZMUMPS_REAL) gmm::imag(v2[Blocks[did]+row]);
        }
        idz->job=6; // factor and solve
        zmumps_c(idz);
        #pragma omp parallel for
        for(size_t row = 0; row < (size_t) idz->n; row++)
        {
            v2[Blocks[did]+row] = std::complex<double>(idz->rhs[row].r, idz->rhs[row].i);
        }
        idz->job=-2; // end
        zmumps_c(idz);
        delete idz->a;
        delete idz->rhs;
        delete idz->irn;
        delete idz->jcn;
        delete idz;
    }
}

template <typename Matrix, typename VecI, typename V1, typename V2> inline
void GaussSeidelDblSchur(const Matrix& A, const Matrix& PR, const VecI& Blocks, const V1& v1, V2& v2)
{
    typedef gmm::row_matrix<gmm::rsvector<std::complex<double> > > MatType;
    typedef gmm::rsvector<std::complex<double> > VecType;
    gmm::copy(v1, v2);
    for(size_t did = Blocks.size()-2; did >= 0; did--)
    {
        //std::cout << "\n" << did;
        ZMUMPS_STRUC_C* idz = new ZMUMPS_STRUC_C;
        idz->job=-1; // init
        idz->par=1;
        idz->sym=(MUMPS_INT) 1;
        idz->comm_fortran=-987654; // use_comm_world
        zmumps_c(idz);
        idz->n = (MUMPS_INT)(Blocks[did+1]-Blocks[did]);
//        if(did>0) {
        idz->nz = (MUMPS_INT) gmm::nnz(gmm::sub_matrix(A, gmm::sub_interval(Blocks[did], Blocks[did+1]-Blocks[did])));
        idz->irn = new MUMPS_INT [idz->nz];
        idz->jcn = new MUMPS_INT [idz->nz];
        idz->a = new ZMUMPS_COMPLEX [idz->nz];
        typename gmm::linalg_traits<MatType>::const_row_iterator it = mat_row_const_begin(A);
        typename gmm::linalg_traits<MatType>::const_row_iterator ite = mat_row_const_end(A);
        size_t i = 0,  idx = 0;
        for(; it != ite; ++it)
        {
            if((i >= Blocks[did]) && (i < Blocks[did+1]))
            {
                typename gmm::linalg_traits<MatType>::const_sub_row_type row = gmm::linalg_traits<MatType>::row(it);
                typename gmm::linalg_traits<VecType>::const_iterator itr = vect_const_begin(row);
                typename gmm::linalg_traits<VecType>::const_iterator itre = vect_const_end(row);
                for(; itr != itre; ++itr)
                {
                    if((itr.index() >= Blocks[did]) && (itr.index() < Blocks[did+1]) && (itr.index() >= i))
                    {
                        idz->a[idx].r = (ZMUMPS_REAL) itr->real();
                        idz->a[idx].i = (ZMUMPS_REAL) itr->imag();
                        idz->irn[idx] = (MUMPS_INT)(i-Blocks[did])+1;
                        idz->jcn[idx++] = (MUMPS_INT)(itr.index()-Blocks[did])+1;
                    }
                }
            }
            i++;
        }
//        } else {
//
//        }
        idz->icntl[0]=-1; //-1 6
        idz->icntl[1]=-1; //-1 6
        idz->icntl[2]=-1; //-1 1
        idz->icntl[3]=0; //0 6
        //
        V2 tmp(idz->n);
        for(size_t lil = Blocks.size()-2; lil > did; lil--)
        {
            //std::cout << "+";
            gmm::mult_add(gmm::sub_matrix(PR, gmm::sub_interval(Blocks[did], Blocks[did+1]-Blocks[did]),
                                          gmm::sub_interval(Blocks[lil], Blocks[lil+1]-Blocks[lil])),
                          gmm::sub_vector(v2, gmm::sub_interval(Blocks[lil], Blocks[lil+1]-Blocks[lil])),
                          tmp);
        }
        gmm::add(gmm::scaled(tmp,-1.0), gmm::sub_vector(v2, gmm::sub_interval(Blocks[did], Blocks[did+1]-Blocks[did])));
        //
        idz->rhs = new ZMUMPS_COMPLEX [idz->n];
        #pragma omp parallel for
        for(size_t row = 0; row < (size_t) idz->n; row++)
        {
            idz->rhs[row].r = (ZMUMPS_REAL) gmm::real(v2[Blocks[did]+row]);
            idz->rhs[row].i = (ZMUMPS_REAL) gmm::imag(v2[Blocks[did]+row]);
        }
        idz->job=6; // factor and solve
        zmumps_c(idz);
        #pragma omp parallel for
        for(size_t row = 0; row < (size_t) idz->n; row++)
        {
            v2[Blocks[did]+row] = std::complex<double>(idz->rhs[row].r, idz->rhs[row].i);
        }
        idz->job=-2; // end
        zmumps_c(idz);
        delete idz->a;
        delete idz->rhs;
        delete idz->irn;
        delete idz->jcn;
        delete idz;
        if(did == 0)
        {
            break;
        }
    }
    //std::cout << "\n";
}


template <typename Matrix, typename VecI, typename V1, typename V2> inline
void JAcobiSgl(const Matrix& A, const VecI& Blocks, const V1& v1, V2& v2)
{
    typedef gmm::row_matrix<gmm::rsvector<std::complex<double> > > MatType;
    typedef gmm::rsvector<std::complex<double> > VecType;
    gmm::copy(v1, v2);
    for(size_t did=0; did < Blocks.size()-1; did++)
    {
        CMUMPS_STRUC_C* idz = new CMUMPS_STRUC_C;
        idz->job=-1; // init
        idz->par=1;
        idz->sym=(MUMPS_INT) 1;
        idz->comm_fortran=-987654; // use_comm_world
        cmumps_c(idz);
        idz->n = (MUMPS_INT)(Blocks[did+1]-Blocks[did]);
        idz->nz = (MUMPS_INT) gmm::nnz(gmm::sub_matrix(A, gmm::sub_interval(Blocks[did], Blocks[did+1]-Blocks[did])));
        idz->irn = new MUMPS_INT [idz->nz];
        idz->jcn = new MUMPS_INT [idz->nz];
        idz->a = new CMUMPS_COMPLEX [idz->nz];
        typename gmm::linalg_traits<MatType>::const_row_iterator it = mat_row_const_begin(A);
        typename gmm::linalg_traits<MatType>::const_row_iterator ite = mat_row_const_end(A);
        size_t i = 0,  idx = 0;
        for(; it != ite; ++it)
        {
            if((i >= Blocks[did]) && (i < Blocks[did+1]))
            {
                typename gmm::linalg_traits<MatType>::const_sub_row_type row = gmm::linalg_traits<MatType>::row(it);
                typename gmm::linalg_traits<VecType>::const_iterator itr = vect_const_begin(row);
                typename gmm::linalg_traits<VecType>::const_iterator itre = vect_const_end(row);
                for(; itr != itre; ++itr)
                {
                    if((itr.index() >= Blocks[did]) && (itr.index() < Blocks[did+1]) && (itr.index() >= i))
                    {
                        idz->a[idx].r = (CMUMPS_REAL) itr->real();
                        idz->a[idx].i = (CMUMPS_REAL) itr->imag();
                        idz->irn[idx] = (MUMPS_INT)(i-Blocks[did])+1;
                        idz->jcn[idx++] = (MUMPS_INT)(itr.index()-Blocks[did])+1;
                    }
                }
            }
            i++;
        }
        idz->icntl[0]=-1; //-1 6
        idz->icntl[1]=-1; //-1 6
        idz->icntl[2]=-1; //-1 1
        idz->icntl[3]=0; //0 6
        idz->rhs = new CMUMPS_COMPLEX [idz->n];
        #pragma omp parallel for
        for(size_t row = 0; row < (size_t) idz->n; row++)
        {
            idz->rhs[row].r = (CMUMPS_REAL) gmm::real(v2[Blocks[did]+row]);
            idz->rhs[row].i = (CMUMPS_REAL) gmm::imag(v2[Blocks[did]+row]);
        }
        idz->job=6; // factor and solve
        cmumps_c(idz);
        #pragma omp parallel for
        for(size_t row = 0; row < (size_t) idz->n; row++)
        {
            v2[Blocks[did]+row] = std::complex<double>(idz->rhs[row].r, idz->rhs[row].i);
        }
        idz->job=-2; // end
        cmumps_c(idz);
        delete idz->a;
        delete idz->rhs;
        delete idz->irn;
        delete idz->jcn;
        delete idz;
    }
}


template <typename Matrix, typename VecI, typename V1, typename V2> inline
void GaussSeidelSgl(const Matrix& A, const Matrix& PR, const VecI& Blocks, const V1& v1, V2& v2)
{
    typedef gmm::row_matrix<gmm::rsvector<std::complex<double> > > MatType;
    typedef gmm::rsvector<std::complex<double> > VecType;
    gmm::copy(v1, v2);
    for(size_t did=0; did < Blocks.size()-1; did++)
    {
        CMUMPS_STRUC_C* idz = new CMUMPS_STRUC_C;
        idz->job=-1; // init
        idz->par=1;
        idz->sym=(MUMPS_INT) 1;
        idz->comm_fortran=-987654; // use_comm_world
        cmumps_c(idz);
        idz->n = (MUMPS_INT)(Blocks[did+1]-Blocks[did]);
        idz->nz = (MUMPS_INT) gmm::nnz(gmm::sub_matrix(A, gmm::sub_interval(Blocks[did], Blocks[did+1]-Blocks[did])));
        idz->irn = new MUMPS_INT [idz->nz];
        idz->jcn = new MUMPS_INT [idz->nz];
        idz->a = new CMUMPS_COMPLEX [idz->nz];
        typename gmm::linalg_traits<MatType>::const_row_iterator it = mat_row_const_begin(A);
        typename gmm::linalg_traits<MatType>::const_row_iterator ite = mat_row_const_end(A);
        size_t i = 0,  idx = 0;
        for(; it != ite; ++it)
        {
            if((i >= Blocks[did]) && (i < Blocks[did+1]))
            {
                typename gmm::linalg_traits<MatType>::const_sub_row_type row = gmm::linalg_traits<MatType>::row(it);
                typename gmm::linalg_traits<VecType>::const_iterator itr = vect_const_begin(row);
                typename gmm::linalg_traits<VecType>::const_iterator itre = vect_const_end(row);
                for(; itr != itre; ++itr)
                {
                    if((itr.index() >= Blocks[did]) && (itr.index() < Blocks[did+1]) && (itr.index() >= i))
                    {
                        idz->a[idx].r = (CMUMPS_REAL) itr->real();
                        idz->a[idx].i = (CMUMPS_REAL) itr->imag();
                        idz->irn[idx] = (MUMPS_INT)(i-Blocks[did])+1;
                        idz->jcn[idx++] = (MUMPS_INT)(itr.index()-Blocks[did])+1;
                    }
                }
            }
            i++;
        }
        idz->icntl[0]=-1; //-1 6
        idz->icntl[1]=-1; //-1 6
        idz->icntl[2]=-1; //-1 1
        idz->icntl[3]=0; //0 6
        //
        V2 tmp(idz->n);
        for(size_t lil = 0; lil < did; lil++)
        {
            gmm::mult_add(gmm::sub_matrix(PR, gmm::sub_interval(Blocks[did], Blocks[did+1]-Blocks[did]),
                                          gmm::sub_interval(Blocks[lil], Blocks[lil+1]-Blocks[lil])),
                          gmm::sub_vector(v2, gmm::sub_interval(Blocks[lil], Blocks[lil+1]-Blocks[lil])),
                          tmp);
        }
        gmm::add(gmm::scaled(tmp,-1.0), gmm::sub_vector(v2, gmm::sub_interval(Blocks[did], Blocks[did+1]-Blocks[did])));
        //
        idz->rhs = new CMUMPS_COMPLEX [idz->n];
        #pragma omp parallel for
        for(size_t row = 0; row < (size_t) idz->n; row++)
        {
            idz->rhs[row].r = (CMUMPS_REAL) gmm::real(v2[Blocks[did]+row]);
            idz->rhs[row].i = (CMUMPS_REAL) gmm::imag(v2[Blocks[did]+row]);
        }
        idz->job=6; // factor and solve
        cmumps_c(idz);
        #pragma omp parallel for
        for(size_t row = 0; row < (size_t) idz->n; row++)
        {
            v2[Blocks[did]+row] = std::complex<double>(idz->rhs[row].r, idz->rhs[row].i);
        }
        idz->job=-2; // end
        cmumps_c(idz);
        delete idz->a;
        delete idz->rhs;
        delete idz->irn;
        delete idz->jcn;
        delete idz;
    }
}

template <typename Mat, typename Vec, typename VecB, typename VecI, typename Log, typename Opt>
void GMRES(const Mat& A, const Mat& PR, Vec& x, const VecB& b, const VecI& Blocks,
           int restart, gmm::iteration& outer, Log& logFile, Opt& opt)
{
    if(opt->nJorGS)
    {
        if(opt->dbl)
        {
            logFile << "Gauss-Seidel double prec. preconditioner\n";
            std::cout << "Gauss-Seidel double prec. preconditioner\n";
        }
        else
        {
            logFile << "Gauss-Seidel single prec. preconditioner\n";
            std::cout << "Gauss-Seidel single prec. preconditioner\n";
        }
    }
    else
    {
        if(opt->dbl)
        {
            logFile << "Jacobi double prec. preconditioner\n";
            std::cout << "Jacobi double prec. preconditioner\n";
        }
        else
        {
            logFile << "Jacobi single prec. preconditioner\n";
            std::cout << "Jacobi single prec. preconditioner\n";
        }
    }
    typedef typename gmm::linalg_traits<Vec>::value_type T;
    typedef typename gmm::number_traits<T>::magnitude_type R;
    gmm::modified_gram_schmidt<T> KS(restart, gmm::vect_size(x));
    std::vector<T> w(gmm::vect_size(x)), r(gmm::vect_size(x)), u(gmm::vect_size(x)), tmp(gmm::vect_size(x));
    std::vector<T> c_rot(restart+1), s_rot(restart+1), s(restart+1);
    gmm::dense_matrix<T> H(restart+1, restart);
    Mat Adiag(gmm::vect_size(x),gmm::vect_size(x));
    #pragma omp parallel for
    for(size_t id=0; id < gmm::vect_size(x); id++)
    {
        Adiag(id,id) = -A(id,id);
    }
    if(opt->nJorGS)
    {
        if(opt->dbl)
        {
            if(opt->dds)
            {
                GaussSeidelDbl(A,PR,Blocks,b,r);
            }
            else
            {
                GaussSeidelDbl(A,PR,Blocks,b,r);
            }
        }
        else
        {
            GaussSeidelSgl(A,PR,Blocks,b,r);
        }
    }
    else
    {
        if(opt->dbl)
        {
            JacobiDbl(A,Blocks,b,r);
        }
        else
        {
            JAcobiSgl(A,Blocks,b,r);
        }
    }
    outer.set_rhsnorm(gmm::vect_norm2(r));
    if(outer.get_rhsnorm() == 0.0)
    {
        gmm::clear(x);
        return;
    }
    gmm::mult(A, gmm::scaled(x, T(-1)), b, w);
    gmm::mult_add(gmm::transposed(A), gmm::scaled(x, T(-1)), w);
    gmm::mult_add(Adiag, gmm::scaled(x, T(-1)), w);
    gmm::mult_add(PR, gmm::scaled(x, T(-1)), w);
    if(opt->nJorGS)
    {
        if(opt->dbl)
        {
            if(opt->dds)
            {
                GaussSeidelDbl(A,PR,Blocks,w,r);
            }
            else
            {
                GaussSeidelDbl(A, PR, Blocks, w, r);
            }
        }
        else
        {
            GaussSeidelSgl(A, PR, Blocks, w, r);
        }
    }
    else
    {
        if(opt->dbl)
        {
            JacobiDbl(A,Blocks, w, r);
        }
        else
        {
            JAcobiSgl(A,Blocks, w, r);
        }
    }
    R beta = gmm::vect_norm2(r), beta_old = beta;
    int blocked = 0;
    gmm::iteration inner = outer;
    inner.reduce_noisy();
    inner.set_maxiter(restart);
    inner.set_name("GMRes inner");
    int itnum = 0;
    while(! outer.finished(beta))
    {
        gmm::copy(gmm::scaled(r, R(1)/beta), KS[0]);
        gmm::clear(s);
        s[0] = beta;
        gmm::size_type i = 0;
        inner.init();
        do
        {
            logFile << itnum << " " << (inner.get_res()/inner.get_rhsnorm() > 0 ?
                                        inner.get_res()/inner.get_rhsnorm() :
                                        outer.get_res()/outer.get_rhsnorm()) << "\n";
            std::cout << itnum++ << ": " << (inner.get_res()/inner.get_rhsnorm() > 0 ?
                                             inner.get_res()/inner.get_rhsnorm() :
                                             outer.get_res()/outer.get_rhsnorm()) << "\n";
            gmm::mult(A, KS[i], u);
            gmm::mult_add(gmm::transposed(A), KS[i], u);
            gmm::mult_add(Adiag, KS[i], u);
            gmm::mult_add(PR, KS[i], u);
            if(opt->nJorGS)
            {
                if(opt->dbl)
                {
                    if(opt->dds)
                    {
                        GaussSeidelDbl(A,PR,Blocks, u, tmp);
                    }
                    else
                    {
                        GaussSeidelDbl(A,PR,Blocks, u, tmp);
                    }
                }
                else
                {
                    GaussSeidelSgl(A,PR,Blocks, u, tmp);
                }
            }
            else
            {
                if(opt->dbl)
                {
                    JacobiDbl(A,Blocks, u, tmp);
                }
                else
                {
                    JAcobiSgl(A,Blocks, u, tmp);
                }
            }
            gmm::copy(tmp, KS[i+1]);
            gmm::orthogonalize(KS, gmm::mat_col(H, i), i);
            R a = gmm::vect_norm2(KS[i+1]);
            H(i+1, i) = T(a);
            gmm::scale(KS[i+1], T(1) / a);
            for(gmm::size_type k = 0; k < i; k++)
            {
                gmm::Apply_Givens_rotation_left(H(k,i), H(k+1,i), c_rot[k], s_rot[k]);
            }
            gmm::Givens_rotation(H(i,i), H(i+1,i), c_rot[i], s_rot[i]);
            gmm::Apply_Givens_rotation_left(H(i,i), H(i+1,i), c_rot[i], s_rot[i]);
            gmm::Apply_Givens_rotation_left(s[i], s[i+1], c_rot[i], s_rot[i]);
            ++inner, ++outer, ++i;
        }
        while(! inner.finished(gmm::abs(s[i])));
        gmm::upper_tri_solve(H, s, i, false);
        gmm::combine(KS, s, x, i);
        gmm::mult(A, gmm::scaled(x, T(-1)), b, w);
        gmm::mult_add(gmm::transposed(A), gmm::scaled(x, T(-1)), w);
        gmm::mult_add(Adiag, gmm::scaled(x, T(-1)), w);
        gmm::mult_add(PR, gmm::scaled(x, T(-1)), w);
        if(opt->nJorGS)
        {
            if(opt->dbl)
            {
                if(opt->dds)
                {
                    GaussSeidelDbl(A,PR,Blocks,w,r);
                }
                else
                {
                    GaussSeidelDbl(A,PR,Blocks,w,r);
                }
            }
            else
            {
                GaussSeidelSgl(A,PR,Blocks,w,r);
            }
        }
        else
        {
            if(opt->dbl)
            {
                JacobiDbl(A,Blocks,w,r);
            }
            else
            {
                JAcobiSgl(A,Blocks,w,r);
            }
        }
        beta_old = std::min(beta, beta_old);
        beta = gmm::vect_norm2(r);
        if(int(inner.get_iteration()) < restart -1 || beta_old <= beta)
        {
            ++blocked;
        }
        else
        {
            blocked = 0;
        }
        if(blocked > 10)
        {
            if(outer.get_noisy())
            {
                std::cout << "Gmres is blocked, exiting\n";
            }
            break;
        }
    }
}
#endif // GMRES_H
