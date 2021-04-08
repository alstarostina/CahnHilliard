//
// Created by vintick on 21.02.21.
//

#ifndef CHPROJECT_CAHNHILLIARD_H
#define CHPROJECT_CAHNHILLIARD_H

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_minres.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <fstream>
#include <iostream>
#include <random>

using namespace dealii;
class CahnHilliard
{
public:
    CahnHilliard();
    void run();
private:
    void make_grid();
    void setup_system_CH();
    void assemble_system_CH();
    void solve_CH();
    void output_results_CH(int step) const;
    void generate_initial_value();
    void assemble_rhs_CH();
    void update_and_clean_up_CH();
    const double tau = 1.25e-4/2;
    const double epsilon  = 2e-2;

    Triangulation<2> triangulation;
    FE_Q<2, 2>           fe;
    DoFHandler<2>    dof_handler;

    BlockSparsityPattern      sparsity_pattern_CH;
    BlockSparseMatrix<double> system_matrix_CH;
    BlockSparseMatrix<double> preconditioner_CH;
    BlockVector <double>          solution_CH;
    BlockVector <double>          system_rhs_CH;
    Vector <double>     initial_value;
};

#endif //CHPROJECT_CAHNHILLIARD_H