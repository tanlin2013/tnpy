from sympy import symbols, Matrix, pprint, tensorproduct, tensorcontraction


if __name__ == "__main__":

    Sp, Sm, Sz, I, O = symbols('S^+ S^- Sz I O')
    mpo = Matrix(
        [[I, Sp, Sm, Sz, Sz+I],
         [O, O, O, O, Sm],
         [O, O, O, O, Sp],
         [O, O, O, O, Sz],
         [O, O, O, O, I]]
    )

    Sp_, Sm_, Sz_, I_, O_ = symbols('S^+_ S^-_ Sz_ I_ O_')
    mpo_ = Matrix(
        [[I_, Sp_, Sm_, Sz_, Sz_ + I_],
         [O_, O_, O_, O_, Sm_],
         [O_, O_, O_, O_, Sp_],
         [O_, O_, O_, O_, Sz_],
         [O_, O_, O_, O_, I_]]
    )

    bi_mpo = tensorproduct(mpo, mpo_)
    pprint(mpo)
    pprint(mpo_)
    pprint(bi_mpo)
