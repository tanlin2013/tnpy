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
      tuple <double,double> initialize_iMPS();
      double initialize_fMPS(int N; string canonical_form);
      void normalize_fMPS(vector<UniTensor>& Gs; string canonical_form);
      double to_GL_rep(vector<UniTensor>& Gs);
  private:
      static int d;
      static int chi;
};

#endif
