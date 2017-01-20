/************************************************************
* File: MPS.hpp
* Descrption - Definitions of MPS.cpp.
************************************************************/
#ifndef MPS
#define MPS
#include <vector>
#include <tuple>
#include <uni10.hpp>
using namespace uni10;
  
class MPS{
    public:
        MPS(int d, int chi);
        tuple<vector<UniTensor>,vector<Matrix>> initialize_iMPS();
        vector<UniTensor> initialize_fMPS(const int N, const string canonical_form);
        //void normalize_fMPS(vector<UniTensor>& Gs, string canonical_form);
        //double to_GL_rep(vector<UniTensor>& Gs);
    private:
        int d;
        int chi;
};

#endif
