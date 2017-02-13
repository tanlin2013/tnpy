
void linalg::eigs(void &eigsMatvec, const int n, double *evals, double *evecs){
    double *resNorms;
    primme_paras primme;
    primme_initialize(&primme);
    primme.matrixMatvec = eigsMatvec;
    primme.n = n;
    primme.numEvals = 1;
    primme_set_method(PRIMME_JDQR,&primme);
    evals = (double*)malloc(primme.numEvals*sizeof(double));
    evecs = (double*)malloc(primme.n*primme.numEvals*sizeof(double));
    resNorms = (double*)malloc(primme.numEvals*sizeof(double));
    int ret = dprimme(evals,evecs,resNorms,&primme);
    if (ret != 0) {
        fprintf(primme.outputFile, "Error: primme returned with nonzero exit status: %d \n",ret);
    }
    primme_free(&primme); free(resNorms);
}

void linalg::svds(Matrix &A, const double chi, double *svals, double *svecs){
    this->A = A; this->m = A.col(); this->n = A.row();
    double *resNorms;
    primme_svds_params primme_svds;
    primme_svds_initialize(&primme_svds);
    primme_svds.matrixMatvec = svdsMatvec;
    primme_svds.m = m;
    primme_svds.n = n;
    primme_svds.numSvals = chi;
    primmesvds_set_method(primme_svds_hybrid, PRIMME_DEFAULT_METHOD, PRIMME_DEFAULT_METHOD, &primme_svds);
    svals = (double*)malloc(primme_svds.numSvals*sizeof(double));
    svecs = (double*)malloc((primme_svds.n+primme_svds.m)*primme_svds.numSvals*sizeof(double));
    resNorms = (double*)malloc(primme_svds.numSvals*sizeof(double));
    int ret = dprimme_svds(svals, svecs, resNorms, &primme_svds);
    if (ret!=0){
        fprintf(primme_svds.outputFile, "Error: primme_svds returned with nonzero exit status: %d \n",ret);
    }
    primme_svds_free(&primme_svds); free(resNorms);
}

void linalg::svdsMatvec(void *x, PRIMME_INT *ldx, void *y, PRIMME_INT *ldy, int *blockSize, int *transpose, primme_svds_params *primme_svds, int *err){
    double *xvec; double *yvec;
    if (*transpose==0){
        char TRANS="N";
    }
    else{
        char TRANS="T";
    }
    for (i=0; i<*blockSize; i++){
        xvec = (double *)x + (*ldx)*i;
        yvec = (double *)y + (*ldy)*i;
    }
    double alpha = 1.0, beta = 0.0;
    dgemv(&TRANS, &m, &n, &alpha, &A, &m, &xvec, &m, &beta, &yvec, &m));
}
