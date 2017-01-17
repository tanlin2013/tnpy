/**************************************************************************
This file contains the fundamental operations for Tensor Network algorithm.
**************************************************************************/

tuple <double,double> MPS::initialize_iMPS(){
    vector<UniTensor> Gs;
    vector<Matrix> SVMs;
    vsctor<bonds> bonds;
    
    for (int site=0; site<2; site++){
            Gs.push_back(UniTensor G(bonds).randomize());
            SVMs.push_back(Matrix SVM(chi,chi).randomize());
        }
    return make_tuple(Gs,SVMs);
}

double MPS::initialize_fMPS(string canonical_form=NULL, int N=NULL){
    vector<UniTensor> Gs;
    vector<Matrix> SVMs;
    vector<Bond> bonds;
    bonds.push_back(Bond bdi(BD_IN,chi));
    bonds.push_back(Bond bdo1(BD_OUT,d));
    bonds.push_back(Bond bdo2(BD_OUT,chi));
    
        for (int site=0; site<N; site++){
            if (site==0){
                Gs.push_back(UniTensor G().randomize());
            }
            else if (site==N-1){
                Gs.push_back(UniTensor G().randomize())
            }
            else{
                Gs.push_back(UniTensor G(bonds).randomize());
            }
            SVMs.push_back(Matrix SVM(chi,chi).randomize());
        }
   
    return make_tuple(Gs,SVMs);
}

void normalize_fMPS(){

}

void to_GL_rep(){

}
double eigensolver(){

}

double Trotter_Suzuki_Decomposition(){

}
