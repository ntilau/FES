#include "option.h"
#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <stdlib.h>
#include <string.h>

// ── Enum name helpers ──

const char* option::assemb_type_name(assemb_type t) {
    switch(t) {
        case em_e_fd:     return "em_e_fd";
        case em_e_fd_dd:  return "em_e_fd_dd";
        case em_e_fd_nl:  return "em_e_fd_nl";
        case em_e_qs:     return "em_e_qs";
        case em_h_qs:     return "em_h_qs";
        case em_ez_fd:    return "em_ez_fd";
        case em_e_tl_eig: return "em_e_tl_eig";
        default:          return "UNKNOWN";
    }
}

const char* option::solver_type_name(solverType t) {
    switch(t) {
        case direct:  return "direct";
        case gmres:   return "gmres";
        case matlab:  return "matlab";
        default:      return "UNKNOWN";
    }
}

option::assemb_type option::assemb_type_from_name(const std::string& s) {
    if(s == "em_e_fd")     return em_e_fd;
    if(s == "em_e_fd_dd")  return em_e_fd_dd;
    if(s == "em_e_fd_nl")  return em_e_fd_nl;
    if(s == "em_e_qs")     return em_e_qs;
    if(s == "em_h_qs")     return em_h_qs;
    if(s == "em_ez_fd")    return em_ez_fd;
    if(s == "em_e_tl_eig") return em_e_tl_eig;
    return em_e_fd;
}

option::assemb_type option::formula_type_from_name(const std::string& s) {
    if(s == "em_e_fd")     return em_e_fd;
    if(s == "em_e_fd_dd")  return em_e_fd_dd;
    if(s == "em_e_fd_nl")  return em_e_fd_nl;
    if(s == "em_e_qs")     return em_e_qs;
    if(s == "em_h_qs")     return em_h_qs;
    if(s == "em_ez_fd")    return em_ez_fd;
    if(s == "em_e_tl_eig") return em_e_tl_eig;
    return em_e_fd;
}

option::solverType option::solver_type_from_name(const std::string& s) {
    if(s == "gmres")   return gmres;
    if(s == "matlab")  return matlab;
    return direct;
}

// ── constsructor ──

option::option()
    : solver(direct)
    , assembly(em_e_fd)
    , dbg(false)
    , dbl(true)
    , niter(0)
    , toll(0)
    , h_ord(0)
    , p_ord(1)
    , freq(5e9)
    , l_freq(0)
    , h_freq(0)
    , n_freqs(1)
    , n_harm(0)
    , relax(0)
    , sol(false)
    , einc(false)
    , field(false)
    , rad(false)
    , n_theta(0)
    , n_phi(0)
    , poly(false)
    , dd(false)
    , ddn(false)
    , dds(false)
    , nl(false)
    , n_jor_gs(true)
    , n_dd(0)
    , power(1.0)
{
    E[0] = E[1] = E[2] = 0;
    k[0] = k[1] = k[2] = 0;
    // cli_override is default-initialized (empty map)
}
option::~option() {}

// ── Thin CLI wrapper ──
//
// Only special-cases:
//   <project-name>    position 0 (sets opt.name)
//   +f <freq>         frequency in Hz (+f is the most common interactive arg)
//
// Everything else is stored as generic key→value pairs in cli_override and
// applied after loading .fes XML, then re-saved.
//
// Known XML option keys (from serialize() / readXmlHeader()):
//   solver, assembly, dbg, dbl, niter, toll,
//   h_ord, p_ord, freq, l_freq, h_freq, n_freqs,
//   n_harm, nl_mtrl_name, kerr, relax, n_dd,
//   sweep_freq, sparam, sol, tfe, einc,
//   field, rad, n_theta, n_phi,
//   hfss, poly, unv, poly_cmd,
//   href, href_cmd, verbose, msh,
//   dd, ddn, dds, n_jor_gs, nl, high_p, power,
//   stat, estat, mstat, tmz,
//   Ex, Ey, Ez, kx, ky, kz

void option::set(const int argc, char* argv[])
{
    for(int cnt = 1; cnt < argc; ++cnt)
    {
        // ── +f <freq> (special-cased for convenience) ──
        if(strcmp(argv[cnt], "+f") == 0) {
            if(++cnt >= argc) { std::cout << "Missing frequency\n"; print_usage(std::cout); exit(1); }
            cli_override["freq"] = argv[cnt];
            continue;
        }

        // ── +formula <name> (select formulation directly) ──
        if(strcmp(argv[cnt], "+formula") == 0) {
            if(++cnt >= argc) { std::cout << "Missing formula name\n"; print_usage(std::cout); exit(1); }
            cli_override["formula"] = argv[cnt];
            continue;
        }

        // ── +volt <boundary> <voltage> (special-cased: takes two args) ──
        if(strcmp(argv[cnt], "+volt") == 0) {
            if(cnt + 2 >= argc) { std::cout << "Missing boundary name or voltage\n"; print_usage(std::cout); exit(1); }
            cli_override["volt:" + std::string(argv[cnt+1])] = argv[cnt+2];
            cnt += 2;
            continue;
        }

        // ── project name (first non-flag argument) ──
        if(argv[cnt][0] != '+' && argv[cnt][0] != '-') {
            if(name.empty()) name = argv[cnt];
            continue;
        }

        // ── Generic +flag <val> or -flag / +flag (boolean) ──
        std::string key = argv[cnt] + 1;  // strip leading '+' or '-'

        // Normalize shorthand flags to their serialized XML key names
        if(key == "p")       key = "p_ord";
        else if(key == "h")  key = "h_ord";

        if(cnt + 1 < argc && argv[cnt+1][0] != '+' && argv[cnt+1][0] != '-') {
            // Next token is a value (not another flag)
            cli_override[key] = argv[++cnt];
        } else {
            // Boolean: +flag → "1", -flag → "0"
            cli_override[key] = (argv[cnt][0] == '+') ? "1" : "0";
        }
    }
}

// ── Apply cli_override entries to member fields ──
// Mirrors the key→field mapping in readXmlHeader() in project.cpp.
// Used both pre-load (to set +poly/+hfss flags before project constructor)
// and post-load (to override values read from .fes XML).
void option::apply_cli()
{
    if(cli_override.empty()) return;

    auto ov = [&](const std::string& key) -> bool {
        return cli_override.find(key) != cli_override.end();
    };
    auto getS = [&](const std::string& key) -> const std::string& {
        return cli_override[key];
    };
    auto getB = [&](const std::string& key) -> bool {
        return cli_override[key] == "1";
    };
    auto getD = [&](const std::string& key) -> double {
        try { return std::stod(cli_override[key]); } catch(...) { return 0; }
    };
    auto getI = [&](const std::string& key) -> size_t {
        try { return (size_t)std::stoull(cli_override[key]); } catch(...) { return 0; }
    };

    if(ov("solver"))    solver    = solver_type_from_name(getS("solver"));
    if(ov("formula"))  assembly  = formula_type_from_name(getS("formula"));
    if(ov("assembly"))  assembly  = assemb_type_from_name(getS("assembly"));

    // Shorthand: +em_e_fd, +em_ez_fd, +em_e_tl_eig, etc.
    if(ov("em_e_fd"))     assembly = em_e_fd;
    if(ov("em_e_fd_dd"))  assembly = em_e_fd_dd;
    if(ov("em_e_fd_nl"))  assembly = em_e_fd_nl;
    if(ov("em_e_qs"))     assembly = em_e_qs;
    if(ov("em_h_qs"))     assembly = em_h_qs;
    if(ov("em_ez_fd"))    assembly = em_ez_fd;
    if(ov("em_e_tl_eig")) assembly = em_e_tl_eig;

    if(ov("dbg"))        dbg        = getB("dbg");
    if(ov("dbl"))        dbl        = getB("dbl");
    if(ov("sol"))        sol        = getB("sol");
    if(ov("einc"))       einc       = getB("einc");
    if(ov("field"))      field      = getB("field");
    if(ov("rad"))        rad        = getB("rad");
    // +poly <cmd>: if value is not "1"/"0", treat as boolean true and capture command
    if(ov("poly")) {
        const std::string& v = cli_override["poly"];
        if(v == "1" || v == "0") { poly = (v == "1"); }
        else { poly = true; poly_cmd = v; }
    }
    if(ov("dd"))         dd         = getB("dd");
    if(ov("ddn"))        ddn        = getB("ddn");
    if(ov("dds"))        dds        = getB("dds");
    if(ov("nl"))         nl         = getB("nl");
    if(ov("n_jor_gs"))   n_jor_gs     = getB("n_jor_gs");
    if(ov("n_dd"))       n_dd        = getI("n_dd");
    if(ov("niter"))      niter      = getI("niter");
    if(ov("h_ord"))      h_ord       = getI("h_ord");
    if(ov("p_ord"))      p_ord       = getI("p_ord");
    if(ov("n_freqs"))    n_freqs     = getI("n_freqs");
    if(ov("n_harm"))     n_harm      = getI("n_harm");

    if(ov("freq"))       freq       = getD("freq");
    if(ov("toll"))       toll       = getD("toll");
    if(ov("l_freq"))     l_freq      = getD("l_freq");
    if(ov("h_freq"))     h_freq      = getD("h_freq");
    if(ov("relax"))      relax      = getD("relax");
    if(ov("n_theta"))    n_theta     = getD("n_theta");
    if(ov("n_phi"))      n_phi       = getD("n_phi");
    if(ov("power"))      power      = getD("power");

    if(ov("poly_cmd"))   poly_cmd    = getS("poly_cmd");
    if(ov("href_cmd"))   href_cmd    = getS("href_cmd");

    if(ov("Ex")) { einc = true; E[0] = getD("Ex"); }
    if(ov("Ey")) { einc = true; E[1] = getD("Ey"); }
    if(ov("Ez")) { einc = true; E[2] = getD("Ez"); }
    if(ov("kx")) { einc = true; k[0] = getD("kx"); }
    if(ov("ky")) { einc = true; k[1] = getD("ky"); }
    if(ov("kz")) { einc = true; k[2] = getD("kz"); }

    // volt: prefix keys → Vbnd map (one per +volt <boundary> <voltage> CLI arg)
    for(const auto& kv : cli_override) {
        if(kv.first.compare(0, 5, "volt:") == 0) {
            try { Vbnd[kv.first.substr(5)] = std::stod(kv.second); }
            catch(...) { std::cerr << "Warning: invalid voltage for " << kv.first << "\n"; }
        }
    }
}

// ── XML Serialize ──
// Emits XML element lines (no indentation — caller wraps <options>)

void option::serialize(std::ostream& out) const
{
    auto esc = [&](const std::string& s) {
        for(char c : s) {
            switch(c) {
                case '&':  out << "&amp;";  break;
                case '<':  out << "&lt;";   break;
                case '>':  out << "&gt;";   break;
                case '"':  out << "&quot;"; break;
                case '\'': out << "&apos;"; break;
                default:   out << c;         break;
            }
        }
    };

    auto el = [&](const std::string& name, const std::string& val) {
        out << "  <" << name << ">"; esc(val); out << "</" << name << ">\n";
    };
    auto elBool = [&](const std::string& name, bool v) {
        out << "  <" << name << ">" << (v ? "1" : "0") << "</" << name << ">\n";
    };
    auto elSize  = [&](const std::string& name, size_t v) { out << "  <" << name << ">" << v << "</" << name << ">\n"; };
    auto elDbl   = [&](const std::string& name, double v) {
        out << "  <" << name << ">" << std::setprecision(17) << v << "</" << name << ">\n";
    };

    el("solver", solver_type_name(solver));
    el("assembly", assemb_type_name(assembly));
    el("name", name);
    elBool("dbg", dbg);
    elBool("dbl", dbl);
    elSize("niter", niter);
    elDbl("toll", toll);
    elSize("h_ord", h_ord);
    elSize("p_ord", p_ord);
    elDbl("freq", freq);
    elDbl("l_freq", l_freq);
    elDbl("h_freq", h_freq);
    elSize("n_freqs", n_freqs);
    elSize("n_harm", n_harm);
    elDbl("relax", relax);
    elBool("sol", sol);
    elBool("einc", einc);
    elBool("field", field);
    elBool("rad", rad);
    elDbl("n_theta", n_theta);
    elDbl("n_phi", n_phi);
    el("href_cmd", href_cmd);
    elBool("dd", dd);
    elBool("ddn", ddn);
    elBool("dds", dds);
    elBool("nl", nl);
    elBool("n_jor_gs", n_jor_gs);
    elSize("n_dd", n_dd);
    elDbl("power", power);
    // Arrays E and k
    elDbl("Ex", E[0]); elDbl("Ey", E[1]); elDbl("Ez", E[2]);
    elDbl("kx", k[0]); elDbl("ky", k[1]); elDbl("kz", k[2]);
    // Vbnd map
    for(const auto& kv : Vbnd) {
        out << "  <Vbnd:" << kv.first << ">" << std::setprecision(17) << kv.second
            << "</Vbnd:" << kv.first << ">\n";
    }
}

// ── Deserialize ──  (no-op — XML parsing is done in project.cpp)
void option::deserialize(std::istream& /*in*/)
{
    // XML header parsing is handled by readXmlHeader() in project.cpp,
    // which populates both option fields and mesh regions/boundaries.
}

void option::print_usage(std::ostream& ostr) const
{
    ostr << "Usage: FES\n";
    ostr << "  name                         project name\n";
    ostr << "  +f <freq>                    main frequency [Hz]\n";
    ostr << "        ==  0 Hz  -> Electrostatic formulation\n";
    ostr << "        >= 1 MHz  -> Electromagnetic formulation\n";
    ostr << "  [+fr $lf $hf $n]             perform discrete frequency sweep\n";
    ostr << "  [+pow $p]                    power scaling at ports (default = 1[W])\n";
    ostr << "  [+p $n]                      select polynomial order (1-3)\n";
    ostr << "  [+h $n]                      perform homogeneous mesh refinement\n";
    ostr << "  [+poly [q__a__]]              mesh .poly project (auto-detects 2D/3D); TetGen quality switches optional\n";
    ostr << "  [+href q__a__Y]              quality mesh refinement with TetGen\n";
    ostr << "  [+einc <label> = {Ex,Ey,Ez,kx,ky,kz}]  apply incident plane wave\n";
    ostr << "                               and disable +sparam\n";
    ostr << "  [+volt $bnd $pot]            apply voltage [V] to perfect_e boundary\n";
    ostr << "  [+matlab]                    dump to Matlab in MatrixMarket format\n";
    ostr << "  [+sgl]                       decrease solver precision to single\n";
    ostr << "  [+direct]                    solve direct\n";
    ostr << "  [+gmres $tol $restart]       solve with GMRes\n";
    ostr << "  [+nl $h $mtrl $kerr $relax]  solve direct Kerr materials with $h harm.\n";

    ostr << "  [+dd $n]                     apply partitioning in $n regions\n";
    ostr << "            [+gs]              Gauss-Seidel precond. (default)\n";
    ostr << "            [+jc]              Jacobi precond.\n";
    ostr << "  [+sol]                       write solution\n";
    ostr << "  [+solid]                     write VTK solids\n";
    ostr << "  [+field]                     write VTK fields\n";
    ostr << "  [+rad $n_theta $n_phi]         write VTK radiation solids\n";
    ostr << "Default options: +direct +p 1 (rad. conditions on ports)\n";
    ostr << "Example: FES MagicTee 1e11 +field\n";
}
