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
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_minres.h>
#include <deal.II/lac/sparse_matrix.h>
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
  void setup_system();
  void assemble_system();
  void solve();
  void output_results() const;
  void setup_system_CH();
  void assemble_system_CH();
  void solve_CH();
  void output_results_CH(int step) const;
  void generate_initial_value();
  void assemble_rhs_CH();
  void update_and_clean_up_CH();


  Triangulation<2> triangulation;
  FE_Q<2, 2>           fe;
  DoFHandler<2>    dof_handler;
  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;
  Vector<double> solution;
  Vector<double> system_rhs;

  BlockSparsityPattern      sparsity_pattern_CH;
  BlockSparseMatrix<double> system_matrix_CH;
  BlockVector <double>          solution_CH;
  BlockVector <double>          system_rhs_CH;
  Vector <double>     initial_value;
};
CahnHilliard::CahnHilliard()
  : fe(1)
    , dof_handler(triangulation)
{}
void CahnHilliard::make_grid()
{
  GridGenerator::hyper_cube(triangulation, 0, 1);
  triangulation.refine_global(6);
  std::cout << "Number of active cells: " << triangulation.n_active_cells()
            << std::endl;
}

void CahnHilliard::setup_system_CH()
{
   dof_handler.distribute_dofs(fe);
   std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs()
             << std::endl;
   const unsigned int dofs_number =  dof_handler.n_dofs();
   BlockDynamicSparsityPattern dsp_CH(2, 2);
   std::cout << dofs_number << std::endl;
   dsp_CH.block(0, 0).reinit(dofs_number, dofs_number);
   dsp_CH.block(1, 0).reinit(dofs_number, dofs_number);
   dsp_CH.block(0, 1).reinit(dofs_number, dofs_number);
   dsp_CH.block(1, 1).reinit(dofs_number, dofs_number);
   dsp_CH.collect_sizes();
   DoFTools::make_sparsity_pattern(dof_handler, dsp_CH.block(0, 0));
   DoFTools::make_sparsity_pattern(dof_handler, dsp_CH.block(1, 0));
   DoFTools::make_sparsity_pattern(dof_handler, dsp_CH.block(0, 1));
   DoFTools::make_sparsity_pattern(dof_handler, dsp_CH.block(1, 1));
   sparsity_pattern_CH.copy_from(dsp_CH);
   system_matrix_CH.reinit(sparsity_pattern_CH);
   solution_CH.reinit(3);
   solution_CH.block(0).reinit(dofs_number);
   solution_CH.block(1).reinit(dofs_number);
   solution_CH.collect_sizes();
   system_rhs_CH.reinit(2);
   system_rhs_CH.block(0).reinit(dofs_number);
   system_rhs_CH.block(1).reinit(dofs_number);
   system_rhs_CH.collect_sizes();
   initial_value.reinit(dofs_number);
}

void CahnHilliard::generate_initial_value()
{
   double lower_bound = - 0.1;
   double upper_bound = 0.1;
   std::default_random_engine re;
   std::uniform_real_distribution<double> unif(lower_bound,upper_bound);
   const unsigned int number = initial_value.size();
   for (unsigned int i = 0; i < number; ++i)
   {
      initial_value(i) = unif(re);
   }
}

void CahnHilliard::assemble_system_CH()
{
   double tau = 1.25e-4/2;
   double epsilon  = 2e-2;
   QGauss<2> quadrature_formula(fe.degree + 1);
   FEValues<2> fe_values(fe,
                         quadrature_formula,
                         update_values | update_gradients | update_JxW_values);
   const unsigned int dofs_per_cell = fe.dofs_per_cell;

   FullMatrix<double> cell_K(dofs_per_cell, dofs_per_cell);
   FullMatrix<double> cell_M(dofs_per_cell, dofs_per_cell);

   std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
   for (const auto &cell : dof_handler.active_cell_iterators())
   {
      fe_values.reinit(cell);
      cell_K = 0;
      cell_M = 0;
      cell->get_dof_indices(local_dof_indices);
      for (const unsigned int q_index : fe_values.quadrature_point_indices())
      {
         for (const unsigned int i : fe_values.dof_indices())
         {
            for ( const unsigned int j : fe_values.dof_indices() )
            {
               cell_K( i, j ) += ( fe_values.shape_grad( i, q_index ) * // grad phi_i(x_q)
                                   fe_values.shape_grad( j, q_index ) * // grad phi_j(x_q)
                                   fe_values.JxW( q_index ) );          // dx

               cell_M( i, j ) += ( fe_values.shape_value( i, q_index ) * // phi_i(x_q)
                                   fe_values.shape_value( j, q_index ) * // f(x_q)
                                   fe_values.JxW( q_index ) );           // dx
            }
         }
      }

      for (const unsigned int i : fe_values.dof_indices())
      {
         for ( const unsigned int j : fe_values.dof_indices() )
         {
            system_matrix_CH.block( 0, 0 ).add(
                local_dof_indices[i], local_dof_indices[j], -epsilon * cell_K( i, j ) - ( 8 / epsilon ) * cell_M( i, j ) );
            system_matrix_CH.block( 1, 1 ).add( local_dof_indices[i], local_dof_indices[j], tau * epsilon * cell_K( i, j ) );
            system_matrix_CH.block( 0, 1 ).add( local_dof_indices[i], local_dof_indices[j], cell_M( i, j ) );
            system_matrix_CH.block(1, 0).add(local_dof_indices[i],
                                             local_dof_indices[j],
                                             cell_M(i, j));
         }
      }

     // system_matrix_CH.print_formatted(std::cout);
   }
}


void CahnHilliard::solve_CH()
{
   SolverControl solver_control(1000, 1e-12);
   SolverMinRes<BlockVector<double>> solver(solver_control);
   solver.solve(system_matrix_CH, solution_CH, system_rhs_CH, PreconditionIdentity());
}

void CahnHilliard::assemble_rhs_CH()
{
   constexpr double epsilon  = 2e-2;
   QGauss<2> quadrature_formula(fe.degree + 1);
   FEValues<2> fe_values(fe,
                         quadrature_formula,
                         update_values | update_gradients | update_JxW_values);
   const unsigned int dofs_per_cell = fe.dofs_per_cell;

   Vector<double>     cell_rhs_psi(dofs_per_cell);
   Vector<double>     cell_rhs_mu(dofs_per_cell);
   std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
   for (const auto &cell : dof_handler.active_cell_iterators())
   {
      fe_values.reinit(cell);
      cell_rhs_psi = 0;
      cell_rhs_mu = 0;
      cell->get_dof_indices(local_dof_indices);

      std::vector<double> initial_values_at_quad_points(fe_values.quadrature_point_indices().size());
      fe_values.get_function_values(initial_value, initial_values_at_quad_points);
      for (const unsigned int q_index : fe_values.quadrature_point_indices())
      {
         double a = initial_values_at_quad_points[q_index];//initial_value(local_dof_indices[i]);
         for (const unsigned int i : fe_values.dof_indices())
         {
            cell_rhs_psi( i ) += ( fe_values.shape_value( i, q_index ) * // phi_i(x_q)
                                   (-(12/epsilon)* a + (4/epsilon) * pow(a,3)) * // f(x_q)
                                   fe_values.JxW( q_index ) );           // dx

            cell_rhs_mu( i ) += ( fe_values.shape_value( i, q_index ) * // phi_i(x_q)
                                  (a) *                                   // f(x_q)
                                  fe_values.JxW( q_index ) );           // dx
         }

      }
      system_rhs_CH.block(0).add(local_dof_indices, cell_rhs_psi);
      system_rhs_CH.block(1).add(local_dof_indices, cell_rhs_mu);
   }
}

void CahnHilliard::update_and_clean_up_CH()
{
   initial_value = solution_CH.block(0);
   solution_CH.block(0) = 0;
   solution_CH.block(1) = 0;
   system_rhs_CH.block(0) = 0;
   system_rhs_CH.block(1) = 0;
}

void CahnHilliard::output_results_CH(int step) const
{
   DataOut<2> data_out;
   data_out.attach_dof_handler(dof_handler);
   data_out.add_data_vector(solution_CH.block(0), "solution_CH");

   data_out.build_patches();
   const std::string filename =
       "output/solution_CH" + Utilities::int_to_string(step, 3) + ".vtk";
   std::ofstream output_init(filename);
   data_out.write_vtk(output_init);
}

void CahnHilliard::run()
{
  make_grid();
  setup_system_CH();
  generate_initial_value();
  assemble_system_CH();
  assemble_rhs_CH();
  for (unsigned int i = 0; i < 40; ++i)
   {
      solve_CH();
      output_results_CH(i);
      update_and_clean_up_CH();
      assemble_rhs_CH();
   }

}
int main()
{
  deallog.depth_console(2);
  CahnHilliard problem;
  problem.run();
  return 0;
}
