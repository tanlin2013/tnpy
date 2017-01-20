/**************************************************************************
This file contains the fundamental operations for Tensor Network algorithm.
**************************************************************************/

MPS::MPS(int d, int chi){
    this->d=d;
    this->chi=chi;
}

tuple<vector<UniTensor>,vector<Matrix>> MPS::initialize_iMPS(){
    vector<UniTensor> Gs;
    vector<Matrix> SVMs;
    vector<Bond> bonds;
    Bond vbdi(BD_IN,chi); Bond pbdo(BD_OUT,d); Bond vbdo(BD_OUT,chi);
    bonds.push_back(vbdi); bonds.push_back(pbdo); bonds.push_back(vbdo);
    for(int site=0; site<2; site++){
        UniTensor G(bonds); G.randomize(); Gs.push_back(G);
        Matrix SVM(chi,chi); SVM.randomize(); SVMs.push_back(SVM);
        }
    return make_tuple(Gs,SVMs);
}

vector<UniTensor> MPS::initialize_fMPS(const int N, const string canonical_form){
    vector<UniTensor> Gs;
    vector<Bond> bonds;
    Bond pbdi(BD_IN,this->d); Bond vbdi(BD_IN,this->chi); Bond pbdo(BD_OUT,this->d); Bond vbdo(BD_OUT,this->chi);
    for(int site=0; site<N; site++){
        if(site==0){
            bonds.push_back(vbdi); bonds.push_back(pbdo);
        }
        else if(site==N-1){
            bonds.push_back(pbdi); bonds.push_back(vbdo);
        }
        else{
            bonds.push_back(vbdi); bonds.push_back(pbdo); bonds.push_back(vbdo);
        }
        UniTensor G(bonds); G.randomize(); Gs.push_back(G); bonds.clear();
    }
    return Gs;
}

