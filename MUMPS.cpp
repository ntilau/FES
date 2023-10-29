#include "MUMPS.h"
#include <zmumps_c.h>
#include <cfloat>

MUMPS::MUMPS(arma::cx_mat& Af, arma::cx_mat& Bf)
{
    //copy
    size_t SymmFlag = 2;
    bool sparse = false;
//    arma::wall_clock tt;
    gmm::resize(A,Af.n_rows,Af.n_cols);
    gmm::resize(B,Bf.n_rows,Bf.n_cols);
    for(size_t i = 0; i< Af.n_cols; i++)
    {
        for(size_t j = 0; j< Af.n_rows; j++)
        {
            if(std::abs(Af(i,j)) > DBL_EPSILON)
            {
                A(i,j) = Af(i,j);
            }
        }
        for(size_t j = 0; j< Bf.n_rows; j++)
        {
            if(std::abs(Bf(i,j)) > DBL_EPSILON)
            {
                B(i,j) = Bf(i,j);
            }
        }
    }
    // solve
    ZMUMPS_STRUC_C* idz = new ZMUMPS_STRUC_C;
    idz->job=-1; // init
    idz->par=1;
    idz->sym=(MUMPS_INT) SymmFlag;
    idz->comm_fortran=-987654; // use_comm_world
    zmumps_c(idz);
    idz->n = (MUMPS_INT) Af.n_rows;
    if(SymmFlag == 0)
    {
        idz->nz = (MUMPS_INT) gmm::nnz(A);
    }
    else
    {
        idz->nz = (MUMPS_INT)(gmm::nnz(A) - gmm::mat_nrows(A))/2 + gmm::mat_nrows(A);
    }
    idz->irn = new MUMPS_INT [idz->nz];
    idz->jcn = new MUMPS_INT [idz->nz];
    idz->a = new ZMUMPS_COMPLEX [idz->nz];
    typename gmm::linalg_traits<MatRowType>::const_row_iterator it = mat_row_const_begin(A);
    typename gmm::linalg_traits<MatRowType>::const_row_iterator ite = mat_row_const_end(A);
    size_t i = 0,  idx = 0;
    for(; it != ite; ++it)
    {
        typename gmm::linalg_traits<MatRowType>::const_sub_row_type row = gmm::linalg_traits<MatRowType>::row(it);
        typename gmm::linalg_traits<VecType>::const_iterator itr = vect_const_begin(row);
        typename gmm::linalg_traits<VecType>::const_iterator itre = vect_const_end(row);
        for(; itr != itre; ++itr)
        {
            if(itr.index() >= i || SymmFlag == 0)
            {
                idz->a[idx].r = (ZMUMPS_REAL) itr->real();
                idz->a[idx].i = (ZMUMPS_REAL) itr->imag();
                idz->irn[idx] = (MUMPS_INT) i+1;
                idz->jcn[idx++] = (MUMPS_INT) itr.index()+1;
            }
        }
        i++;
    }
    idz->icntl[0]=-1; //-1 6
    idz->icntl[1]=-1; //-1 6
    idz->icntl[2]=-1; //-1 1
    idz->icntl[3]=0; //0 6
//    idz->icntl[5]=7;
    //idz->icntl[6]=7;
//    idz->icntl[7]=7;
    idz->cntl[0] = 1e-3;
    idz->job=1; // analysis
    zmumps_c(idz);
    //std::cout << " -" << idz->info[14] << "- ";
    idz->job=2; // factorization
//    tt.tic();
    //idz->icntl[13] = 100; /// work array increase
    zmumps_c(idz);
//    std::cout << "Factor: " << tt.toc() << "\n";
    //std::cout << "Factored - ";
    idz->nrhs = (MUMPS_INT) Bf.n_cols;
    idz->lrhs = idz->n;
    idz->rhs = new ZMUMPS_COMPLEX [idz->lrhs*idz->nrhs];
    if(sparse)
    {
        idz->icntl[19] = 1; // sparse right hand side
        idz->nz_rhs = (MUMPS_INT) gmm::nnz(B) ;
        idz->rhs_sparse = new ZMUMPS_COMPLEX [idz->nz_rhs];
        idz->irhs_sparse = new MUMPS_INT [idz->nz_rhs];
        idz->irhs_ptr = new MUMPS_INT [idz->nrhs+1];
        it = mat_row_const_begin(B);
        ite = mat_row_const_end(B);
        i = 0,  idx = 0;
        for(; it != ite; ++it)
        {
            typename gmm::linalg_traits<MatRowType>::const_sub_row_type row = gmm::linalg_traits<MatRowType>::row(it);
            typename gmm::linalg_traits<VecType>::const_iterator itr = vect_const_begin(row);
            typename gmm::linalg_traits<VecType>::const_iterator itre = vect_const_end(row);
            idz->irhs_ptr[i++] = (MUMPS_INT) idx+1;
            for(; itr != itre; ++itr)
            {
                if(itr.index() >= i || SymmFlag == 0 || true)
                {
                    idz->irhs_sparse[idx] = (MUMPS_INT) itr.index()+1;
                    idz->rhs_sparse[idx].r = (ZMUMPS_REAL) itr->real();
                    idz->rhs_sparse[idx++].i = (ZMUMPS_REAL) itr->imag();
                }
            }
        }
        idz->irhs_ptr[idz->nrhs] = (MUMPS_INT)(idz->nz_rhs+1);
    }
    else
    {
        size_t shift;
        for(size_t col = 0; col < idz->nrhs; col++)
        {
            shift = col*idz->lrhs;
            for(size_t row=0; row < idz->lrhs; row++)
            {
                idz->rhs[shift+row].r = std::real(Bf(row,col));
                idz->rhs[shift+row].i = std::imag(Bf(row,col));
            }
        }
    }
    idz->job = 3;
//    tt.tic();
    zmumps_c(idz);
//    std::cout << "Solve: " << tt.toc() << "\n";
    size_t shift;
    Sol.resize(idz->lrhs, idz->nrhs);
    for(size_t col = 0; col < idz->nrhs; col++)
    {
        shift = col*idz->lrhs;
        for(size_t row=0; row < idz->lrhs; row++)
        {
            Sol(row,col) = std::complex<double>((double)idz->rhs[shift+row].r,(double)idz->rhs[shift+row].i);
        }
    }
    idz->job=-2; // end
    zmumps_c(idz);
    delete idz->a;
    delete idz->rhs;
    delete idz->irn;
    delete idz->jcn;
    delete idz->irhs_sparse;
    delete idz->irhs_ptr;
    delete idz->rhs_sparse;
    delete idz;
}


MUMPS::MUMPS(MatRowType& AII, MatRowType& AFI, MatRowType& S)
{
    ZMUMPS_STRUC_C* idz = new ZMUMPS_STRUC_C;
    idz->job=-1; // init
    idz->par=1;
    idz->sym=(MUMPS_INT) 1;
    idz->comm_fortran=-987654; // use_comm_world
    zmumps_c(idz);
    idz->n = (MUMPS_INT) gmm::mat_nrows(AII);
    idz->nz = (MUMPS_INT) gmm::nnz(AII);
    idz->irn = new MUMPS_INT [idz->nz];
    idz->jcn = new MUMPS_INT [idz->nz];
    idz->a = new ZMUMPS_COMPLEX [idz->nz];
    typename gmm::linalg_traits<MatRowType>::const_row_iterator it = mat_row_const_begin(AII);
    typename gmm::linalg_traits<MatRowType>::const_row_iterator ite = mat_row_const_end(AII);
    size_t i = 0,  idx = 0;
    for(; it != ite; ++it)
    {
        typename gmm::linalg_traits<MatRowType>::const_sub_row_type row = gmm::linalg_traits<MatRowType>::row(it);
        typename gmm::linalg_traits<VecType>::const_iterator itr = vect_const_begin(row);
        typename gmm::linalg_traits<VecType>::const_iterator itre = vect_const_end(row);
        for(; itr != itre; ++itr)
        {
            idz->a[idx].r = (ZMUMPS_REAL) itr->real();
            idz->a[idx].i = (ZMUMPS_REAL) itr->imag();
            idz->irn[idx] = (MUMPS_INT) i+1;
            idz->jcn[idx++] = (MUMPS_INT) itr.index()+1;
        }
        i++;
    }
    //AII.clear_mat();
    idz->nrhs = (MUMPS_INT) gmm::mat_nrows(AFI);
    idz->lrhs = (MUMPS_INT) gmm::mat_ncols(AFI);
    idz->rhs = new ZMUMPS_COMPLEX [idz->lrhs*idz->nrhs];
    idz->nz_rhs = (MUMPS_INT) gmm::nnz(AFI) ;
    idz->rhs_sparse = new ZMUMPS_COMPLEX [idz->nz_rhs];
    idz->irhs_sparse = new MUMPS_INT [idz->nz_rhs];
    idz->irhs_ptr = new MUMPS_INT [idz->nrhs+1];
    it = mat_row_const_begin(AFI);
    ite = mat_row_const_end(AFI);
    i = 0,  idx = 0;
    for(; it != ite; ++it)
    {
        typename gmm::linalg_traits<MatRowType>::const_sub_row_type row = gmm::linalg_traits<MatRowType>::row(it);
        typename gmm::linalg_traits<VecType>::const_iterator itr = vect_const_begin(row);
        typename gmm::linalg_traits<VecType>::const_iterator itre = vect_const_end(row);
        idz->irhs_ptr[i++] = (MUMPS_INT) idx+1;
        for(; itr != itre; ++itr)
        {
            idz->irhs_sparse[idx] = (MUMPS_INT) itr.index()+1;
            idz->rhs_sparse[idx].r = (ZMUMPS_REAL) itr->real();
            idz->rhs_sparse[idx++].i = (ZMUMPS_REAL) itr->imag();
        }
    }
    idz->irhs_ptr[idz->nrhs] = (MUMPS_INT)(idz->nz_rhs+1);
    idz->icntl[0]=-1; //-1 6
    idz->icntl[1]=-1; //-1 6
    idz->icntl[2]=-1; //-1 1
    idz->icntl[3]=0; //0 6
    idz->icntl[13] = 0; //20
    idz->icntl[19] = 1; // sparse right hand side
    idz->cntl[0] = 0.0; //thresh
    idz->job=4; // analysis
    std::cout << "Analyzing ";
    zmumps_c(idz);
    std::cout << "+ " << idz->info[14] << " MB, ";
    idz->job=2; // factorization
    std::cout << "Factor, ";
    zmumps_c(idz);
    idz->job = 3;
    std::cout << "Solve ";
    zmumps_c(idz);
    size_t shift;
    gmm::resize(invAIIAFIt, idz->lrhs, idz->nrhs);
    for(size_t col = 0; col < idz->nrhs; col++)
    {
        shift = col*idz->lrhs;
        for(size_t row=0; row < idz->lrhs; row++)
        {
            std::complex<double> tmp((double)idz->rhs[shift+row].r,(double)idz->rhs[shift+row].i);
            if(std::abs(tmp)>1.0e-12)
            {
                invAIIAFIt(row,col) = tmp;
            }
        }
    }
    idz->job=-2; // end
    zmumps_c(idz);
    delete idz->a;
    delete idz->rhs;
    delete idz->irn;
    delete idz->jcn;
    delete idz->irhs_sparse;
    delete idz->irhs_ptr;
    delete idz->rhs_sparse;
    delete idz;
    gmm::resize(SII,gmm::mat_nrows(S),gmm::mat_ncols(S));
    gmm::mult(gmm::scaled(AFI,-1.0),invAIIAFIt,SII);
    AFI.clear_mat();
    invAIIAFIt.clear_mat();
    gmm::add(SII,S);
    SII.clear_mat();
}

MUMPS::MUMPS(MatRowType& A, std::vector<size_t>& RangeAII, size_t& RangeAFF, MatRowType& IO, bool Mode)
{
    ZMUMPS_STRUC_C* idz = new ZMUMPS_STRUC_C;
    idz->job=-1; // init
    idz->par=1;
    idz->sym=(MUMPS_INT) 2;
    idz->comm_fortran=-987654; // use_comm_world
    zmumps_c(idz);
    idz->n = (MUMPS_INT)(RangeAII[1]-RangeAII[0]);
    idz->nz = (MUMPS_INT) RangeAII[2];
    idz->irn = new MUMPS_INT [idz->nz];
    idz->jcn = new MUMPS_INT [idz->nz];
    idz->a = new ZMUMPS_COMPLEX [idz->nz];
    typename gmm::linalg_traits<MatRowType>::const_row_iterator it = mat_row_const_begin(A);
    typename gmm::linalg_traits<MatRowType>::const_row_iterator ite = mat_row_const_end(A);
    size_t i = 0,  idx = 0;
    for(; it != ite; ++it)
    {
        if((i >= RangeAII[0]) && (i < RangeAII[1]))
        {
            typename gmm::linalg_traits<MatRowType>::const_sub_row_type row = gmm::linalg_traits<MatRowType>::row(it);
            typename gmm::linalg_traits<VecType>::const_iterator itr = vect_const_begin(row);
            typename gmm::linalg_traits<VecType>::const_iterator itre = vect_const_end(row);
            for(; itr != itre; ++itr)
            {
                if((itr.index() >= RangeAII[0]) && (itr.index() < RangeAII[1]))
                {
                    idz->a[idx].r = (ZMUMPS_REAL) itr->real();
                    idz->a[idx].i = (ZMUMPS_REAL) itr->imag();
                    idz->irn[idx] = (MUMPS_INT)(i-RangeAII[0])+1;
                    idz->jcn[idx++] = (MUMPS_INT)(itr.index()-RangeAII[0])+1;
                }
            }
        }
        i++;
    }
    idz->icntl[0]=-1; //-1 6
    idz->icntl[1]=-1; //-1 6
    idz->icntl[2]=-1; //-1 1
    idz->icntl[3]=0; //0 6
    idz->icntl[19] = 1; // sparse right hand side
    //idz->icntl[13] = 0; //20
    //idz->cntl[0] = 0.0; //thresh
    idz->job=4; // analysis
    std::cout << "Analyzing ";
    zmumps_c(idz);
    std::cout << "+ " << idz->info[14] << " MB, ";
    idz->job=2; // factorization
    std::cout << "Factor, ";
    zmumps_c(idz);
    idz->nrhs = (MUMPS_INT) RangeAFF ;
    idz->lrhs = (MUMPS_INT)(RangeAII[1]-RangeAII[0]);
    idz->rhs = new ZMUMPS_COMPLEX [idz->lrhs*idz->nrhs];
    idz->nz_rhs = (MUMPS_INT) RangeAII[3];
    idz->rhs_sparse = new ZMUMPS_COMPLEX [idz->nz_rhs];
    idz->irhs_sparse = new MUMPS_INT [idz->nz_rhs];
    idz->irhs_ptr = new MUMPS_INT [idz->nrhs+1];
    it = mat_row_const_begin(A);
    ite = mat_row_const_end(A);
    i = 0,  idx = 0;
    for(; it != ite; ++it)
    {
        if(i < RangeAFF)
        {
            typename gmm::linalg_traits<MatRowType>::const_sub_row_type row = gmm::linalg_traits<MatRowType>::row(it);
            typename gmm::linalg_traits<VecType>::const_iterator itr = vect_const_begin(row);
            typename gmm::linalg_traits<VecType>::const_iterator itre = vect_const_end(row);
            idz->irhs_ptr[i] = (MUMPS_INT) idx+1;
            for(; itr != itre; ++itr)
            {
                if((itr.index() >= RangeAII[0]) && (itr.index() < RangeAII[1]))
                {
                    idz->irhs_sparse[idx] = (MUMPS_INT)(itr.index()-RangeAII[0])+1;
                    idz->rhs_sparse[idx].r = (ZMUMPS_REAL) itr->real();
                    idz->rhs_sparse[idx++].i = (ZMUMPS_REAL) itr->imag();
                }
            }
        }
        i++;
    }
    idz->irhs_ptr[idz->nrhs] = (MUMPS_INT)(idz->nz_rhs+1);
//        #pragma omp parallel for
//        for(j=0; j< (RangeAII[1]-RangeAII[0]); j++) {
//            idz->rhs[j].r = (ZMUMPS_REAL) gmm::real(A(is,RangeAII[0]+j));
//            idz->rhs[j].i = (ZMUMPS_REAL) gmm::imag(A(is,RangeAII[0]+j));
//        }
    idz->job = 3;
    std::cout << "Solve ";
    zmumps_c(idz);
//    std::cout << "+";
//        #pragma omp parallel for private(ii, j)
//        for(ii=0; ii< RangeAFF; ii++) {
//            for(j=0; j< (RangeAII[1]-RangeAII[0]); j++) {
//                IO(ii,is) -= A(ii,RangeAII[0]+j)*std::complex<double>(idz->rhs[j].r, idz->rhs[j].i);
//            }
//        }
    /*
        it = mat_row_const_begin(A);
        ite = mat_row_const_end(A);
        size_t i = 0;
        for(; it != ite; ++it) {
            if((i < RangeAFF) && (i <= is)) {
                typename gmm::linalg_traits<MatRowType>::const_sub_row_type row = gmm::linalg_traits<MatRowType>::row(it);
                typename gmm::linalg_traits<VecType>::const_iterator itr = vect_const_begin(row);
                typename gmm::linalg_traits<VecType>::const_iterator itre = vect_const_end(row);
                for(; itr != itre; ++itr) {
                    if((itr.index() >= RangeAII[0]) && (itr.index() < RangeAII[1])) {
                        IO(i,is) -= std::complex<double>(itr->real(), itr->imag())*
                                    std::complex<double>(idz->rhs[itr.index()-RangeAII[0]].r,
                                                         idz->rhs[itr.index()-RangeAII[0]].i);
                        if(i != is) {
                            IO(is,i) -= std::complex<double>(itr->real(), itr->imag())*
                                        std::complex<double>(idz->rhs[itr.index()-RangeAII[0]].r,
                                                             idz->rhs[itr.index()-RangeAII[0]].i);
                        }
                    }
                }
            } else {
                break;
            }
            i++;
        }
        */
    idz->job=-2; // end
    zmumps_c(idz);
    delete idz->a;
    delete idz->rhs;
    delete idz->irn;
    delete idz->jcn;
    delete idz->irhs_sparse;
    delete idz->irhs_ptr;
    delete idz->rhs_sparse;
    delete idz;
}


//MUMPS::MUMPS(MatRowType& A, std::vector<size_t>& RangeAII, size_t& RangeAFF, MatRowType& IO, bool Mode) {
//    ZMUMPS_STRUC_C* idz = new ZMUMPS_STRUC_C;
//    idz->job=-1; // init
//    idz->par=1;
//    idz->sym=(MUMPS_INT) 2;
//    idz->comm_fortran=-987654; // use_comm_world
//    zmumps_c(idz);
//    idz->n = (MUMPS_INT)(RangeAII[1]-RangeAII[0]);
//    idz->nz = (MUMPS_INT) RangeAII[2];
//    idz->irn = new MUMPS_INT [idz->nz];
//    idz->jcn = new MUMPS_INT [idz->nz];
//    idz->a = new ZMUMPS_COMPLEX [idz->nz];
//    typename gmm::linalg_traits<MatRowType>::const_row_iterator it = mat_row_const_begin(A);
//    typename gmm::linalg_traits<MatRowType>::const_row_iterator ite = mat_row_const_end(A);
//    size_t i = 0,  idx = 0;
//    for(; it != ite; ++it) {
//        if((i >= RangeAII[0]) && (i < RangeAII[1])) {
//            typename gmm::linalg_traits<MatRowType>::const_sub_row_type row = gmm::linalg_traits<MatRowType>::row(it);
//            typename gmm::linalg_traits<VecType>::const_iterator itr = vect_const_begin(row);
//            typename gmm::linalg_traits<VecType>::const_iterator itre = vect_const_end(row);
//            for(; itr != itre; ++itr) {
//                if((itr.index() >= RangeAII[0]) && (itr.index() < RangeAII[1])) {
//                    idz->a[idx].r = (ZMUMPS_REAL) itr->real();
//                    idz->a[idx].i = (ZMUMPS_REAL) itr->imag();
//                    idz->irn[idx] = (MUMPS_INT)(i-RangeAII[0])+1;
//                    idz->jcn[idx++] = (MUMPS_INT)(itr.index()-RangeAII[0])+1;
//                }
//            }
//        }
//        i++;
//    }
//    idz->icntl[0]=-1; //-1 6
//    idz->icntl[1]=-1; //-1 6
//    idz->icntl[2]=-1; //-1 1
//    idz->icntl[3]=0; //0 6
//    idz->icntl[19] = 1; // sparse right hand side
//    //idz->icntl[13] = 0; //20
//    //idz->cntl[0] = 0.0; //thresh
//    idz->job=4; // analysis
//    std::cout << "Analyzing ";
//    zmumps_c(idz);
//    std::cout << "+ " << idz->info[14] << " MB, ";
//    idz->job=2; // factorization
//    std::cout << "Factor, ";
//    zmumps_c(idz);
//    idz->nrhs = (MUMPS_INT) 1 ;
//    idz->lrhs = (MUMPS_INT)(RangeAII[1]-RangeAII[0]);
//    idz->rhs = new ZMUMPS_COMPLEX [idz->lrhs*idz->nrhs];
//    idz->job = 3;
//    std::cout << "Solve ";
//    size_t j, ii, colcnt;
//    for(size_t is=0; is< RangeAFF; is++) {
//        //std::cout << RangeAII[3+is] << "\n";
//        idz->nz_rhs = (MUMPS_INT) RangeAII[4+is];
//        idz->rhs_sparse = new ZMUMPS_COMPLEX [idz->nz_rhs];
//        idz->irhs_sparse = new MUMPS_INT [idz->nz_rhs];
//        idz->irhs_ptr = new MUMPS_INT [idz->nrhs+1];
//        it = mat_row_const_begin(A);
//        ite = mat_row_const_end(A);
//        i = 0,  idx = 0, colcnt = 0;
//        for(; it != ite; ++it) {
//            if(colcnt++ == is) {
//                typename gmm::linalg_traits<MatRowType>::const_sub_row_type row = gmm::linalg_traits<MatRowType>::row(it);
//                typename gmm::linalg_traits<VecType>::const_iterator itr = vect_const_begin(row);
//                typename gmm::linalg_traits<VecType>::const_iterator itre = vect_const_end(row);
//                idz->irhs_ptr[i++] = (MUMPS_INT) idx+1;
//                for(; itr != itre; ++itr) {
//                    if((itr.index() >= RangeAII[0]) && (itr.index() < RangeAII[1])) {
//                        //std::cout << "(" << is+1  << "," << itr.index()+1 << ")=" << itr->real() << "+ " << itr->imag() << "i\n";
//                        idz->irhs_sparse[idx] = (MUMPS_INT)(itr.index()-RangeAII[0])+1;
//                        idz->rhs_sparse[idx].r = (ZMUMPS_REAL) itr->real();
//                        idz->rhs_sparse[idx++].i = (ZMUMPS_REAL) itr->imag();
//                    }
//                }
//                break;
//            }
//        }
//        idz->irhs_ptr[idz->nrhs] = (MUMPS_INT)(idz->nz_rhs+1);
////        #pragma omp parallel for
////        for(j=0; j< (RangeAII[1]-RangeAII[0]); j++) {
////            idz->rhs[j].r = (ZMUMPS_REAL) gmm::real(A(is,RangeAII[0]+j));
////            idz->rhs[j].i = (ZMUMPS_REAL) gmm::imag(A(is,RangeAII[0]+j));
////        }
//        zmumps_c(idz);
//        delete idz->irhs_sparse;
//        delete idz->irhs_ptr;
//        delete idz->rhs_sparse;
//        std::cout << "+";
////        #pragma omp parallel for private(ii, j)
////        for(ii=0; ii< RangeAFF; ii++) {
////            for(j=0; j< (RangeAII[1]-RangeAII[0]); j++) {
////                IO(ii,is) -= A(ii,RangeAII[0]+j)*std::complex<double>(idz->rhs[j].r, idz->rhs[j].i);
////            }
////        }
//        it = mat_row_const_begin(A);
//        ite = mat_row_const_end(A);
//        size_t i = 0;
//        for(; it != ite; ++it) {
//            if((i < RangeAFF) && (i <= is)) {
//                typename gmm::linalg_traits<MatRowType>::const_sub_row_type row = gmm::linalg_traits<MatRowType>::row(it);
//                typename gmm::linalg_traits<VecType>::const_iterator itr = vect_const_begin(row);
//                typename gmm::linalg_traits<VecType>::const_iterator itre = vect_const_end(row);
//                for(; itr != itre; ++itr) {
//                    if((itr.index() >= RangeAII[0]) && (itr.index() < RangeAII[1])) {
//                        IO(i,is) -= std::complex<double>(itr->real(), itr->imag())*
//                                    std::complex<double>(idz->rhs[itr.index()-RangeAII[0]].r,
//                                                         idz->rhs[itr.index()-RangeAII[0]].i);
//                        if(i != is) {
//                            IO(is,i) -= std::complex<double>(itr->real(), itr->imag())*
//                                        std::complex<double>(idz->rhs[itr.index()-RangeAII[0]].r,
//                                                             idz->rhs[itr.index()-RangeAII[0]].i);
//                        }
//                    }
//                }
//            } else {
//                break;
//            }
//            i++;
//        }
//    }
//    idz->job=-2; // end
//    zmumps_c(idz);
//    delete idz->a;
//    delete idz->rhs;
//    delete idz->irn;
//    delete idz->jcn;
//    delete idz;
//}



MUMPS::MUMPS(MatRowType& SF, MatRowType& gF)
{
    ZMUMPS_STRUC_C* idz = new ZMUMPS_STRUC_C;
    idz->job=-1; // init
    idz->par=1;
    idz->sym=(MUMPS_INT) 2;
    idz->comm_fortran=-987654; // use_comm_world
    zmumps_c(idz);
    idz->n = (MUMPS_INT) gmm::mat_nrows(SF);
    idz->nz = (MUMPS_INT)(MUMPS_INT)(gmm::nnz(SF) - gmm::mat_nrows(SF))/2 + gmm::mat_nrows(SF);
    idz->irn = new MUMPS_INT [idz->nz];
    idz->jcn = new MUMPS_INT [idz->nz];
    idz->a = new ZMUMPS_COMPLEX [idz->nz];
    typename gmm::linalg_traits<MatRowType>::const_row_iterator it = mat_row_const_begin(SF);
    typename gmm::linalg_traits<MatRowType>::const_row_iterator ite = mat_row_const_end(SF);
    size_t i = 0,  idx = 0;
    for(; it != ite; ++it)
    {
        typename gmm::linalg_traits<MatRowType>::const_sub_row_type row = gmm::linalg_traits<MatRowType>::row(it);
        typename gmm::linalg_traits<VecType>::const_iterator itr = vect_const_begin(row);
        typename gmm::linalg_traits<VecType>::const_iterator itre = vect_const_end(row);
        for(; itr != itre; ++itr)
        {
            if(itr.index() >= i)
            {
                idz->a[idx].r = (ZMUMPS_REAL) itr->real();
                idz->a[idx].i = (ZMUMPS_REAL) itr->imag();
                idz->irn[idx] = (MUMPS_INT) i+1;
                idz->jcn[idx++] = (MUMPS_INT) itr.index()+1;
            }
        }
        i++;
    }
    idz->nrhs = (MUMPS_INT) gmm::mat_nrows(gF);
    idz->lrhs = idz->n;
    idz->rhs = new ZMUMPS_COMPLEX [idz->lrhs*idz->nrhs];
    idz->nz_rhs = (MUMPS_INT) gmm::nnz(gF) ;
    idz->rhs_sparse = new ZMUMPS_COMPLEX [idz->nz_rhs];
    idz->irhs_sparse = new MUMPS_INT [idz->nz_rhs];
    idz->irhs_ptr = new MUMPS_INT [idz->nrhs+1];
    it = mat_row_const_begin(gF);
    ite = mat_row_const_end(gF);
    i = 0,  idx = 0;
    for(; it != ite; ++it)
    {
        typename gmm::linalg_traits<MatRowType>::const_sub_row_type row = gmm::linalg_traits<MatRowType>::row(it);
        typename gmm::linalg_traits<VecType>::const_iterator itr = vect_const_begin(row);
        typename gmm::linalg_traits<VecType>::const_iterator itre = vect_const_end(row);
        idz->irhs_ptr[i++] = (MUMPS_INT) idx+1;
        for(; itr != itre; ++itr)
        {
            idz->irhs_sparse[idx] = (MUMPS_INT) itr.index()+1;
            idz->rhs_sparse[idx].r = (ZMUMPS_REAL) itr->real();
            idz->rhs_sparse[idx++].i = (ZMUMPS_REAL) itr->imag();
        }
    }
    idz->irhs_ptr[idz->nrhs] = (MUMPS_INT)(idz->nz_rhs+1);
    idz->icntl[0]=-1; //-1 6
    idz->icntl[1]=-1; //-1 6
    idz->icntl[2]=-1; //-1 1
    idz->icntl[3]=0; //0 6
    idz->icntl[13] = 0; // work array increase
    idz->icntl[19] = 1; // sparse right hand side
    idz->job = 6;
    zmumps_c(idz);
    size_t shift;
    Sol.resize(idz->lrhs, idz->nrhs);
    for(size_t col = 0; col < idz->nrhs; col++)
    {
        shift = col*idz->lrhs;
        for(size_t row=0; row < idz->lrhs; row++)
        {
            Sol(row,col) = std::complex<double>((double)idz->rhs[shift+row].r,(double)idz->rhs[shift+row].i);
        }
    }
    idz->job=-2; // end
    zmumps_c(idz);
    delete idz->a;
    delete idz->rhs;
    delete idz->irn;
    delete idz->jcn;
    delete idz->irhs_sparse;
    delete idz->irhs_ptr;
    delete idz->rhs_sparse;
    delete idz;
}

MUMPS::~MUMPS()
{
    Sol.clear();
    A.clear_mat();
    B.clear_mat();
    invAIIAFIt.clear_mat();
    SII.clear_mat();
}
