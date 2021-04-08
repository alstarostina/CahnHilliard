//
// Created by vintick on 02.04.21.
//

#ifndef CHPROJECT_BLOCKDIAGONALPRECONDITIONER_H
#define CHPROJECT_BLOCKDIAGONALPRECONDITIONER_H

#include "InverseMatrix.h"

class BlockDiagonalPreconditioner {
public:
    BlockDiagonalPreconditioner(const InverseMatrix &Block_00,
                                const InverseMatrix &Block_11);

    void vmult(BlockVector<double> &dst,
               const BlockVector<double> &src) const;

private:
    const InverseMatrix &Block_00;
    const InverseMatrix &Block_11;
};

inline BlockDiagonalPreconditioner::BlockDiagonalPreconditioner(const InverseMatrix &Block_00,
                                                                const InverseMatrix &Block_11)
        : Block_00(Block_00), Block_11(Block_11) {}

inline void BlockDiagonalPreconditioner::vmult(BlockVector<double> &dst, const BlockVector<double> &src) const {
    assert( dst.n_blocks() == 2);
    assert( src.n_blocks() == 2);
    //std::cout << "solving block 0" << std::endl;
    Block_00.vmult(dst.block(0), src.block(0));
    //std::cout << "solving block 1" << std::endl;
    Block_11.vmult(dst.block(1), src.block(1));
}



#endif //CHPROJECT_BLOCKDIAGONALPRECONDITIONER_H