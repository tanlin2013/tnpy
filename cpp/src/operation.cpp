/**************************************************************************
This file contains the fundamental operations for Tensor Network algorithm.
**************************************************************************/

#include <iostream>
#include <stdexcept>
#include <tuple>
#include <uni10.hpp>
#include <primme.h>
using namespace std;
using namespace uni10;

class MPS{
    public:
        string whichMPS;
        int d;
        int chi;
};

tuple <double,double> MPS::initialize_MPS(string canonical_form=NULL, int N=NULL){
    vector<UniTensor> Gs;
    vector<Matrix> SVMs;
    vector<Bond> bonds;
    bonds.push_back(Bond bdi(BD_IN,chi));
    bonds.push_back(Bond bdo1(BD_OUT,d));
    bonds.push_back(Bond bdo2(BD_OUT,chi));
    if (whichMPS=='i'){
        for (int site=0; site<2; site++){
            Gs.push_back(UniTensor G(bonds).randomize());
            SVMs.push_back(Matrix SVM(chi,chi).randomize());
        }
    }
    else if (whichMPS=='f'){
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
    }
    else{
        throw invalid_argument("MPS must be either iMPS or fMPS.");
    }
    return make_tuple(Gs,SVMs);
}

double MPS::initialize_EnvLs(){



}

double MPS::initialize_EnvRs(){



}



double eigensolver(){

}

double Trotter_Suzuki_Decomposition(){

}
