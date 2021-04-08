//
// Created by vintick on 21.02.21.
//
#include <deal.II/base/logstream.h>
#include "CahnHilliard.h"

int main() {
    deallog.depth_console(2);
    CahnHilliard problem;
    problem.run();
    return 0;
}

