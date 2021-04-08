//
// Created by vintick on 02.04.21.
//

#ifndef CHPROJECT_INVERSEMATRIX_H
#define CHPROJECT_INVERSEMATRIX_H

#include <deal.II/lac/sparse_ilu.h>

class InverseMatrix {
public:
    InverseMatrix(const SparseMatrix<double> &m);

    void vmult(Vector<double> &dst, const Vector<double> &src) const;

private:
    const SparseMatrix<double> &matrix;
    SparseILU<double> preconditioner;
};

inline InverseMatrix::InverseMatrix(const SparseMatrix<double> &m)
        : matrix(m) {
    preconditioner.initialize(matrix);
}


inline void InverseMatrix::vmult(Vector<double> &dst, const Vector<double> &src) const {
    SolverControl solver_control(src.size(), 1e-8 * src.l2_norm());
    SolverCG<Vector<double>> cg(solver_control);
    dst = 0;


    //std::cout<< &src << std::endl;
    //std::cout << &dst << std::endl;
    //std::cout<< &matrix << std::endl;
    try {
        cg.solve(matrix, dst, src, preconditioner);
    }
    catch (std::exception &e) {
        Assert(false, ExcMessage(e.what()));
    }
}


#endif //CHPROJECT_INVERSEMATRIX_H