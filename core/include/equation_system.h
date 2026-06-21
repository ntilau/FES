#ifndef EQSYS_H
#define EQSYS_H

#include <armadillo>

class mesh;
class option;
class project;
class quad;

class eq_sys
{
public:
    using mat_row_type = arma::SpMat<std::complex<double>>;
    using mat_col_type = arma::SpMat<std::complex<double>>;
    using vec_type = arma::cx_vec;
    eq_sys(std::ofstream&, project*);
    virtual ~eq_sys();
    friend class post_processor;

    // ── Simulation context (set by constructor) ──
    project* prj;
    mesh* msh;
    option* opt;
    quad* quadr;

    // ── Scalar accessors ──
    size_t get_dofnum() const { return dofnum; }
    void set_dofnum(size_t n) { dofnum = n; }
    size_t get_dofreal() const { return dofreal; }
    void set_dofreal(size_t n) { dofreal = n; }
    size_t get_nonzero() const { return NonZero; }
    void set_nonzero(size_t n) { NonZero = n; }
    size_t get_wave_ports_num() const { return wave_portsNum; }
    void set_wave_ports_num(size_t n) { wave_portsNum = n; }
    size_t get_wave_ports_dofnum() const { return wave_portsdofnum; }
    void set_wave_ports_dofnum(size_t n) { wave_portsdofnum = n; }
    double get_freq() const { return freq; }
    void set_freq(double f) { freq = f; }
    double get_k0() const { return k0; }
    void set_k0(double v) { k0 = v; }
    double get_kk() const { return kk; }
    void set_kk(double v) { kk = v; }
    double get_error() const { return error; }
    void set_error(double e) { error = e; }
    size_t get_iter() const { return iter; }
    void set_iter(size_t i) { iter = i; }
    int get_symm_flag() const { return symm_flag; }
    void set_symm_flag(int f) { symm_flag = f; }

    // ── Matrix / vector accessors (mutable references for now) ──
    mat_row_type& A_mat() { return A; }
    const mat_row_type& A_mat() const { return A; }
    mat_row_type& PR_mat() { return PR; }
    const mat_row_type& PR_mat() const { return PR; }
    mat_row_type& Adiag_mat() { return Adiag; }
    const mat_row_type& Adiag_mat() const { return Adiag; }
    mat_col_type& B_mat() { return B; }
    const mat_col_type& B_mat() const { return B; }
    arma::cx_mat& Sp_mat() { return Sp; }
    const arma::cx_mat& Sp_mat() const { return Sp; }
    arma::cx_mat& sp_prev_mat() { return sp_prev; }
    arma::cx_mat& Sol_mat() { return Sol; }
    const arma::cx_mat& Sol_mat() const { return Sol; }
    arma::cx_mat& sol_prev_mat() { return sol_prev; }

    // ── DOF/boundary accessors ──
    arma::uvec& Dirdofs_vec() { return Dirdofs; }
    const arma::uvec& Dirdofs_vec() const { return Dirdofs; }
    arma::uvec& Dirdofv_vec() { return Dirdofv; }
    const arma::uvec& Dirdofv_vec() const { return Dirdofv; }
    std::vector<size_t>& non_dir_ids_vec() { return non_dir_ids; }
    arma::uvec& dofmapv_vec() { return dofmapv; }
    arma::uvec& Invdofmapv_vec() { return Invdofmapv; }
    std::vector<size_t>& doflevel_vec() { return doflevel; }
    std::vector<mat_row_type>& AFF_vec() { return AFF; }
    arma::uvec& wave_portIds_vec() { return wave_portIds; }
    const arma::uvec& wave_portIds_vec() const { return wave_portIds; }
    arma::uvec& Nonwave_portIds_vec() { return Nonwave_portIds; }
    const arma::uvec& Nonwave_portIds_vec() const { return Nonwave_portIds; }
    std::vector<std::complex<double>>& port_ampl_vec() { return port_ampl; }

    // ── Timing ──
    arma::wall_clock& tt_clock() { return tt; }
    arma::wall_clock& lt_clock() { return lt; }

private:
    //
    arma::uvec Dirdofs, Dirdofv;
    std::vector<size_t> non_dir_ids;
    arma::uvec dofmapv, Invdofmapv; // useful for DD mapping or reordering
    std::vector<size_t> doflevel; // dof level for Schur decomposition
    std::vector<mat_row_type> AFF; // for each domain internal AFF
    arma::uvec wave_portIds, Nonwave_portIds;
    //
    size_t dofnum, dofreal, NonZero;
    size_t wave_portsNum;
    size_t wave_portsdofnum;
    std::vector<std::complex<double> > port_ampl;
    double freq, k0, kk, error;
    size_t iter;
    mat_row_type A, PR, Adiag;
    mat_col_type B;
    arma::cx_mat Sp, sp_prev, Sol, sol_prev;
    int symm_flag; // A matrix symmetry
    arma::wall_clock tt, lt;
};

#endif // EQSYS_H
