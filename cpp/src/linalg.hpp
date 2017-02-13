#include "uni10.hpp"
#include "primme.h"
using namespace std;
using namespace uni10;
extern "C"{
    void dgemv_(char* TRANS, const int* M, const int* N, double* alpha, double* A, const int* LDA, double* X, const int* INCX, double* beta, double* Y, const int* INCY);
}

class linalg{
    public:
        void eigs(void &eigsMatvec, double *evals, double *evecs);
        void svds(Matrix &A, const double chi, double *svals, double *svecs);
    private:
        Matrix A;
        size_t m;
        size_t n;
        void svdsMatvec(void *x, PRIMME_INT *ldx, void *y, PRIMME_INT *ldy, int *blockSize, int *transpose, primme_svds_params *primme_svds, int *err);
};
