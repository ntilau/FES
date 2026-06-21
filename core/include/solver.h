#ifndef SOLVER_H
#define SOLVER_H

#include <fstream>
#include <memory>
#include <armadillo>

class eq_sys;
class option;

class solver
{
public:
    virtual ~solver() = default;
    virtual void solve(eq_sys& sys, std::ofstream& log) = 0;
    static std::unique_ptr<solver> create(const option& opt);

protected:
    static void extract_solution_full(eq_sys& sys, int n, void* rhs, int col, bool isDouble);
};

// ── MUMPS direct solver (single/double complex) ──

class mumps_solver : public solver
{
public:
    void solve(eq_sys& sys, std::ofstream& log) override;
private:
    template<typename MUMPS_STRUC, typename MUMPS_COMPLEX, typename MUMPS_REAL, void (*MUMPS_FUNC)(MUMPS_STRUC*)>
    void solveDispatch(eq_sys& sys, std::ofstream& log, const char* label);
};

// ── gmres iterative solver (Jacobi or DD-preconditioned) ──

class gmres_solver : public solver
{
public:
    void solve(eq_sys& sys, std::ofstream& log) override;
};

#endif // SOLVER_H
