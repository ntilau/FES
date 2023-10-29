#ifndef QUAD_H
#define QUAD_H

#include <armadillo>

class Quad
{
public:
    Quad(size_t p);
    virtual ~Quad();
    void setQuad(size_t p);
    arma::mat xq3;
    arma::mat xq2;
    arma::vec wq3;
    arma::vec wq2;
};

#endif // QUAD_H
