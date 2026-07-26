#ifndef ASSEMBLER_H
#define ASSEMBLER_H

#include <armadillo>
#include <fstream>
#include <map>
#include <memory>

#include "equation_system.h"
class mesh;
class option;
class project;
class quad;

class assembler
{
public:
    virtual ~assembler() = default;
    virtual void assemble(std::ofstream& log, eq_sys& sys) = 0;

    enum Type { em_e_fd, em_e_fd_dd, em_e_fd_dd_schur, em_e_fd_nl, em_e_qs, em_ez_fd, em_e_tl_eig };

    static std::unique_ptr<assembler> create(Type type);

protected:
    static void read_port_amplitudes(std::ofstream& log, eq_sys& sys);
    static eq_sys::mat_row_type build_sparse(
        const std::map<std::pair<arma::uword, arma::uword>, std::complex<double>>& map,
        size_t n);

    // Shared waveport eigenmode computation (for em_e_fd and em_e_fd_dd).
    // DD variant passes optional domdofmap/shareddofv for shared DOF fixup.
    // Returns the number of waveport modes accumulated.
    static size_t compute_waveport_modes(
        std::ofstream& log, eq_sys& sys,
        mesh* msh, option* opt, project* prj, quad* quadr,
        double k0, double kk,
        std::vector<bool>& doftoLeave,
        arma::wall_clock& lt,
        const arma::field<arma::uvec>* domdofmap = nullptr,
        const arma::umat* shareddofv = nullptr);

    // Rebuild sparse matrix from coordinate list for deterministic CSC structure.
    static void rebuild_sparse(eq_sys::mat_row_type& mat, size_t n);
};

// ── Linear frequency-domain assembly (curl-curl + mass + waveport/TFE/Abc) ──

class assembler_em_e_fd : public assembler
{
public:
    void assemble(std::ofstream& log, eq_sys& sys) override;
};

// ── Domain decomposition assembly (additive Schwarz) ──

class assembler_em_e_fd_dd : public assembler
{
public:
    void assemble(std::ofstream& log, eq_sys& sys) override;
};

// ── Schur complement domain decomposition assembly (dd_schur) ──

class assembler_em_e_fd_schur : public assembler
{
public:
    void assemble(std::ofstream& log, eq_sys& sys) override;
};

// ── Nonlinear Kerr assembly (harmonic balance) ──

class assembler_em_e_fd_nl : public assembler
{
public:
    void assemble(std::ofstream& log, eq_sys& sys) override;
};

// ── Electrostatic assembly (grad-grad) ──

class assembler_em_e_qs : public assembler
{
public:
    void assemble(std::ofstream& log, eq_sys& sys) override;
};

// ── 2D TMz assembly ──

class assembler_em_ez_fd : public assembler
{
public:
    void assemble(std::ofstream& log, eq_sys& sys) override;
};

// ── 2D transmission-line eigenmode (hcurl + hgrad) ──

class assembler_em_e_tl_eig : public assembler
{
public:
    void assemble(std::ofstream& log, eq_sys& sys) override;
};

#endif // ASSEMBLER_H
