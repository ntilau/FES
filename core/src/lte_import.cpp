#include "lte.h"
#include <stdexcept>
#include "mesh.h"
#include "boundary_condition.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iterator>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <set>
#include <map>

// ── Path helpers ──

std::string lte::lte_path(const std::string& file) const {
    size_t slash = name.find_last_of("/\\");
    std::string dir = (slash == std::string::npos) ? "." : name.substr(0, slash);
    return dir + "/lte_fileset/" + file;
}

// Resolve filename: prefer <basename>.<ext> over 3d.<ext>
// Default is "3d" (CST convention). LTE models use <model-name>.<ext>.
std::string lte::resolve_file(const std::string& ext) const {
    std::string base = name;
    size_t slash = base.find_last_of("/\\");
    if(slash != std::string::npos) base = base.substr(slash + 1);
    // Try <basename>.<ext> first
    std::string p = lte_path(base + "." + ext);
    std::ifstream f(p.c_str());
    if(f.good()) { f.close(); return base + "." + ext; }
    // Fall back to 3d.<ext>
    return std::string("3d.") + ext;
}

// ── Constructor ──

lte::lte(project* prj)
    : name(prj->opt->name), msh(prj->msh), prj(prj), debug(prj->opt->dbg)
{
    std::cout << "lte_fileset project files:\n";
    msh->nNodes = 0; msh->nEdges = 0; msh->nFaces = 0; msh->nTetras = 0;
    read_mesh();
    read_mtr();
    read_mtrl_files();
    read_bc();
    Finalizemesh();
    store_plc();
    std::cout << "Nodes  = " << msh->nNodes << "\n"
              << "Edges  = " << msh->nEdges << "\n"
              << "Faces  = " << msh->nFaces << "\n"
              << "Tetras = " << msh->nTetras << "\n";
}

lte::~lte() {}

// ── Mesh reader ──
// Reads 3d.mesh (CST) or <name>.mesh (LTE) — same brace-delimited format.

void lte::read_mesh()
{
    std::string fn = resolve_file("mesh");
    std::string path = lte_path(fn);
    std::ifstream f(path);
    if(!f.is_open()) throw std::runtime_error("Cannot open " + path);
    std::cout << "  Reading " << path << "\n" << std::flush;
    std::string full((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    f.close();

    const char* secNames[] = {"POSITION","EDGE","FACE","ELEMENT","SOLID","POLY_FACES"};
    for(const char* secName : secNames) {
        std::string pat = std::string("{ ") + secName + " SINGLE_MESH";
        size_t hp = full.find(pat);
        if(hp == std::string::npos) continue;
        size_t ds = full.find('\n', hp); if(ds == std::string::npos) continue;
        ds++;
        int depth = 1; size_t scan = ds;
        while(scan < full.size() && depth > 0) {
            if(full[scan] == '{') depth++; else if(full[scan] == '}') depth--;
            scan++;
        }
        std::string block = full.substr(ds, scan - 1 - ds);
        std::string sn(secName);
        if(sn == "POSITION")    { std::istringstream in(block); int c; in>>c; msh->nNodes=(size_t)c; msh->nodPos.set_size(c,3); for(int i=0;i<c;i++){int idx;std::string t;double x,y,z;in>>idx>>t>>x>>y>>z;msh->nodPos(i,0)=x;msh->nodPos(i,1)=y;msh->nodPos(i,2)=z;} }
        else if(sn == "EDGE")   { std::istringstream in(block); int c; in>>c; msh->nEdges=(size_t)c; msh->edgNodes.set_size(c,2); for(int i=0;i<c;i++){int idx,v0,v1;std::string t;in>>idx>>t>>v0>>v1;msh->edgNodes(i,0)=v0;msh->edgNodes(i,1)=v1;} }
        else if(sn == "FACE")   { std::istringstream in(block); int c; in>>c; msh->nFaces=(size_t)c; msh->facNodes.set_size(c,3); for(int i=0;i<c;i++){int idx,v0,v1,v2;std::string t;in>>idx>>t>>v0>>v1>>v2;msh->facNodes(i,0)=v0;msh->facNodes(i,1)=v1;msh->facNodes(i,2)=v2;} }
        else if(sn == "ELEMENT"){ std::istringstream in(block); int c; in>>c; msh->nTetras=(size_t)c; msh->tetNodes.set_size(c,4); for(int i=0;i<c;i++){int idx,v0,v1,v2,v3;std::string t;in>>idx>>t>>v0>>v1>>v2>>v3;msh->tetNodes(i,0)=v0;msh->tetNodes(i,1)=v1;msh->tetNodes(i,2)=v2;msh->tetNodes(i,3)=v3;} }
        else if(sn == "SOLID")  { parse_solid_section(block, parts); }
        else if(sn == "POLY_FACES") { parse_polyface_section(block, faceMap); }
    }

    if(msh->nNodes > 0)  msh->nodPos.resize(msh->nNodes, 3);
    if(msh->nEdges > 0)  msh->edgNodes.resize(msh->nEdges, 2);
    if(msh->nFaces > 0)  msh->facNodes.resize(msh->nFaces, 3);
    if(msh->nTetras > 0) msh->tetNodes.resize(msh->nTetras, 4);
    msh->facLab.resize(msh->nFaces); msh->facLab.fill(msh->maxLab);
    msh->tetLab.resize(msh->nTetras); msh->tetLab.fill(msh->maxLab);
    msh->facAdjTet.set_size(msh->nFaces);
    std::cout << "  Mesh: " << msh->nNodes << " n, " << msh->nEdges
              << " e, " << msh->nFaces << " f, " << msh->nTetras << " t\n";
}

// ── SOLID section parser ──

void lte::parse_solid_section(const std::string& data, std::vector<lte_part>& parts) {
    std::istringstream in(data); int count; in >> count;
    parts.reserve((size_t)count);
    for(int i = 0; i < count; i++) {
        while(true) {
            int idx; std::string solidKw;
            if(in >> idx >> solidKw) {
                if(solidKw == "Solid") {
                    std::string pname; in >> pname;
                    std::string token; in >> token;
                    std::string aiKw; int tc;
                    if(token == "AI") { aiKw=token; in>>tc; }
                    else { in>>aiKw>>tc; }
                    if(aiKw=="AI") {
                        lte_part p; p.name=pname; p.tet_ids.reserve((size_t)tc);
                        for(int j=0;j<tc;j++){int tid;in>>tid;p.tet_ids.push_back((size_t)tid);}
                        parts.push_back(p);
                    }
                    break;
                }
            } else {
                if(in.eof()) return;
                in.clear(); std::string junk; in>>junk;
            }
        }
    }
}

// ── POLY_FACES section parser ──

void lte::parse_polyface_section(const std::string& data,
                                  std::map<std::string, std::vector<size_t>>& faceMap) {
    std::istringstream in(data); int count; in >> count;
    for(int i = 0; i < count; i++) {
        while(true) {
            int idx; std::string skw;
            if(in >> idx >> skw) {
                if(skw == "S") {
                    std::string fname; in >> fname;
                    std::string token; in >> token;
                    std::string aiKw; int fc;
                    if(token == "AI") { aiKw=token; in>>fc; }
                    else { in>>aiKw>>fc; }
                    if(aiKw=="AI") {
                        std::vector<size_t> ids; ids.reserve((size_t)fc);
                        for(int j=0;j<fc;j++){size_t fid;in>>fid;ids.push_back(fid);}
                        faceMap[fname]=ids;
                    }
                    break;
                }
            } else {
                if(in.eof()) return;
                in.clear(); std::string junk; in>>junk;
            }
        }
    }
}

// ── Material mapping reader ──

void lte::read_mtr() {
    std::string fn = resolve_file("mtr");
    std::string path = lte_path(fn);
    std::ifstream f(path);
    if(!f.is_open()) { std::cout << "  Warning: " << path << " not found\n"; return; }
    std::string line;
    while(std::getline(f, line)) {
        std::istringstream iss(line); std::string solidName, matName;
        if(iss >> solidName >> matName) {
            for(auto& p : parts) { if(p.name == solidName) { p.material = matName; break; } }
        }
    }
    f.close();
}

// ── Material properties reader ──

void lte::read_mtrl_files() {
    std::set<std::string> needed;
    for(auto& p : parts) { if(!p.material.empty()) needed.insert(p.material); }
    lte_mtrl vac; vac.name = "vacuum"; vac.permittivity = 1.0; vac.permeability = 1.0; vac.conductivity = 0.0; vac.dielectric_loss_tangent = 0.0;
    mtrls["vacuum"] = vac;
    std::string matDir = lte_path("materials");
    for(const auto& mname : needed) {
        if(mtrls.find(mname) != mtrls.end()) continue;
        std::string mpath = matDir + "/" + mname + ".mtrl";
        std::ifstream f(mpath);
        if(!f.is_open()) {
            std::string lc = mname; for(auto& c : lc) c = (char)std::tolower((unsigned char)c);
            mpath = matDir + "/" + lc + ".mtrl"; f.open(mpath);
        }
        if(!f.is_open()) {
            std::cout << "  Warning: material file '" << mname << ".mtrl' not found\n";
            lte_mtrl m; m.name = mname; m.permittivity = 1.0; m.permeability = 1.0; m.conductivity = 0.0; m.dielectric_loss_tangent = 0.0;
            mtrls[mname] = m; continue;
        }
        std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        lte_mtrl m; m.name = mname; m.permittivity = 1.0; m.permeability = 1.0; m.conductivity = 0.0; m.dielectric_loss_tangent = 0.0;
        auto fv = [&](const std::string& key)->double { auto p=content.find(key); if(p==std::string::npos)return-1.0; std::string a=content.substr(p+key.size()); std::istringstream is(a); double v; if(is>>v)return v; return-1.0; };
        double v; v=fv("EPSILON_RELATIVE"); if(v>=0)m.permittivity=v; v=fv("MU_RELATIVE"); if(v>=0)m.permeability=v;
        v=fv("SIGMA"); if(v>=0)m.conductivity=v; v=fv("DIELECTRIC_LOSS_TANGENT"); if(v>=0)m.dielectric_loss_tangent=v;
        mtrls[mname] = m; f.close();
    }
}

// ── BC reader ──

void lte::read_bc() {
    std::string fn = resolve_file("bc");
    std::string path = lte_path(fn);
    std::ifstream f(path);
    if(!f.is_open()) { std::cout << "  Warning: " << path << " not found\n"; return; }
    std::cout << "  Reading " << path << "\n";

    std::string line; lte_bnd curBnd; bool inBnd = false, inPort = false;

    while(std::getline(f, line)) {
        while(!line.empty()&&(line.back()==' '||line.back()=='\t'||line.back()=='\r')) line.pop_back();
        if(line.empty()) { if(inBnd){ bnds.push_back(curBnd); inBnd=false; inPort=false; } continue; }
        std::istringstream iss(line); std::string first; iss >> first;

        if(first == "Face") {
            if(inBnd){ bnds.push_back(curBnd); inBnd=false; inPort=false; }
            curBnd = lte_bnd(); std::string fname; iss >> fname;
            if(!fname.empty()&&fname.back()==':') fname.pop_back();
            curBnd.name = fname; inBnd = true;
            auto it = faceMap.find(fname);
            if(it != faceMap.end()) curBnd.face_ids = it->second;
            continue;
        }
        if(first.find("Port_") == 0) {
            if(inBnd){ bnds.push_back(curBnd); inBnd=false; inPort=false; }
            curBnd = lte_bnd(); std::string pname = first;
            if(!pname.empty()&&pname.back()=='{') pname.pop_back();
            curBnd.name = pname; inBnd = true;
            auto it = faceMap.find(pname);
            if(it != faceMap.end()) curBnd.face_ids = it->second;
            continue;
        }
        if(!inBnd) continue;
        if(first=="Radiation"||first=="radiation"){ curBnd.type="Radiation"; }
        else if(first=="PerfectE"||first=="Perfect E"){ curBnd.type="PerfectE"; }
        else if(first=="PerfectH"||first=="Perfect H"){ curBnd.type="PerfectH"; }
        else if(first=="LumpedPort"||first=="Lumped Port"){ curBnd.type="LumpedPort"; inPort=true; }
        else if(first=="WavePort"||first=="Wave Port"){ curBnd.type="WavePort"; inPort=true; }
        else if((first=="ModeNum"||first=="NumMode")&&inPort){ iss>>curBnd.num_modes; }
        else if(first=="RenormImp"&&inPort){
            std::string rest; std::getline(iss,rest);
            auto ipos=rest.find("Impedance"); if(ipos!=std::string::npos){
                auto paren=rest.find('(',ipos); if(paren!=std::string::npos){
                    std::string np=rest.substr(paren+1);
                    std::istringstream ns(np); double r,i; char c;
                    if(ns>>r>>c>>i) curBnd.impedance=r;
                }
            }
        }
    }
    if(inBnd) bnds.push_back(curBnd);
    f.close(); std::cout << "  Read " << bnds.size() << " boundaries\n";
}

// ── Finalizemesh ──

void lte::Finalizemesh() {
    int mIdx = 0; msh->tetmtrl.clear();
    for(auto& p : parts) {
        auto mit=mtrls.find(p.material);
        double eps=1,mur=1,sig=0,tand=0; std::string mn="vacuum";
        if(mit!=mtrls.end()){ eps=mit->second.permittivity; mur=mit->second.permeability; sig=mit->second.conductivity; tand=mit->second.dielectric_loss_tangent; mn=mit->second.name; }
        else std::cout<<"  Warning: material '"<<p.material<<"' not found, using vacuum\n";
        mtrl mtr(p.name,mn,eps,mur,sig,tand); mtr.label=mIdx;
        for(size_t i=0;i<p.tet_ids.size();i++) {
            size_t tid=p.tet_ids[i]; if(tid<msh->nTetras){ msh->tetLab(tid)=mIdx; mtr.Tetras.resize(mtr.Tetras.n_elem+1); mtr.Tetras(mtr.Tetras.n_elem-1)=(arma::uword)tid; }
        }
        if(mtr.Tetras.n_elem>0){ std::cout<<"  "<<p.name<<" "<<mn<<"\n"; msh->tetmtrl.push_back(mtr); mIdx++; }
    }
    std::cout<<"  "<<msh->tetmtrl.size()<<" material regions\n";
    for(auto& b : bnds) {
        bc bcObj; bcObj.set_type(b.type); bcObj.name=b.name; bcObj.label=msh->facbc.size();
        std::cout<<"    "<<bcObj.name<<" "<<bcObj.type;
        if(bcObj.type==bc::wave_port){ bcObj.num_modes=b.num_modes; std::cout<<" "<<bcObj.num_modes; }
        if(bcObj.type==bc::lumped_port){ bcObj.impedance=b.impedance; std::cout<<" "<<bcObj.impedance<<" Ohm"; }
        std::cout<<"\n";
        for(size_t i=0;i<b.face_ids.size();i++){ size_t fid=b.face_ids[i]; if(fid<msh->nFaces){ msh->facLab(fid)=bcObj.label; bcObj.Faces.resize(bcObj.Faces.n_elem+1); bcObj.Faces(bcObj.Faces.n_elem-1)=(arma::uword)fid; }}
        msh->facbc.push_back(bcObj);
    }
    // Build face map for O(1) lookup
    std::map<std::pair<size_t,std::pair<size_t,size_t>>, size_t> fmap;
    for(size_t fid=0;fid<msh->nFaces;fid++) {
        size_t n[3]={(size_t)msh->facNodes(fid,0),(size_t)msh->facNodes(fid,1),(size_t)msh->facNodes(fid,2)};
        std::sort(n,n+3); fmap[std::make_pair(n[0],std::make_pair(n[1],n[2]))] = fid;
    }
    msh->tetFaces.resize(msh->nTetras,4);
    for(size_t tid=0;tid<msh->nTetras;tid++) {
        arma::uword n0=msh->tetNodes(tid,0),n1=msh->tetNodes(tid,1),n2=msh->tetNodes(tid,2),n3=msh->tetNodes(tid,3);
        size_t fc[4][3]={{n1,n2,n3},{n0,n2,n3},{n0,n1,n3},{n0,n1,n2}};
        for(int fi=0;fi<4;fi++) {
            std::sort(fc[fi],fc[fi]+3);
            auto it=fmap.find(std::make_pair(fc[fi][0],std::make_pair(fc[fi][1],fc[fi][2])));
            if(it!=fmap.end()){ size_t fid=it->second; msh->tetFaces(tid,fi)=(arma::uword)fid; arma::uvec a(1); a(0)=(arma::uword)tid; msh->facAdjTet(fid)=arma::join_cols(msh->facAdjTet(fid),a); }
        }
    }
    // facEdges
    std::map<std::pair<size_t,size_t>,size_t> em;
    for(size_t i=0;i<msh->nEdges;i++){ size_t n0=msh->edgNodes(i,0),n1=msh->edgNodes(i,1); if(n0>n1)std::swap(n0,n1); em[std::make_pair(n0,n1)]=i; }
    msh->facEdges.resize(msh->nFaces,3);
    for(size_t i=0;i<msh->nFaces;i++){ size_t n0=msh->facNodes(i,0),n1=msh->facNodes(i,1),n2=msh->facNodes(i,2); if(n0>n1)std::swap(n0,n1); if(n0>n2)std::swap(n0,n2); if(n1>n2)std::swap(n1,n2); msh->facEdges(i,0)=(arma::uword)em[std::make_pair(n0,n1)]; msh->facEdges(i,1)=(arma::uword)em[std::make_pair(n0,n2)]; msh->facEdges(i,2)=(arma::uword)em[std::make_pair(n1,n2)]; }
    msh->tetEdges.resize(msh->nTetras,6);
    for(size_t tid=0;tid<msh->nTetras;tid++){ size_t tn[4]={(size_t)msh->tetNodes(tid,0),(size_t)msh->tetNodes(tid,1),(size_t)msh->tetNodes(tid,2),(size_t)msh->tetNodes(tid,3)}; std::sort(tn,tn+4); msh->tetEdges(tid,0)=(arma::uword)em[std::make_pair(tn[0],tn[1])]; msh->tetEdges(tid,1)=(arma::uword)em[std::make_pair(tn[0],tn[2])]; msh->tetEdges(tid,2)=(arma::uword)em[std::make_pair(tn[0],tn[3])]; msh->tetEdges(tid,3)=(arma::uword)em[std::make_pair(tn[1],tn[2])]; msh->tetEdges(tid,4)=(arma::uword)em[std::make_pair(tn[1],tn[3])]; msh->tetEdges(tid,5)=(arma::uword)em[std::make_pair(tn[2],tn[3])]; }
}

// ── Store PLC ──

void lte::store_plc() {
    msh->plc_valid=true; msh->mesh_dim=3;
    int np=(int)msh->nNodes; msh->plc_points.resize(np*3);
    for(int i=0;i<np && i<(int)msh->nodPos.n_rows;i++){ msh->plc_points[i*3]=msh->nodPos(i,0); msh->plc_points[i*3+1]=msh->nodPos(i,1); msh->plc_points[i*3+2]=msh->nodPos(i,2); }
    int nf=(int)msh->nFaces; msh->plc_facet_markers.resize(nf); msh->plc_poly_vertex_counts.resize(nf); msh->plc_poly_vertex_list.resize(nf*3);
    for(int i=0;i<nf;i++){ msh->plc_facet_markers[i]=((size_t)i<msh->facLab.n_elem&&msh->facLab(i)!=msh->maxLab)?(int)msh->facLab(i):0; msh->plc_poly_vertex_counts[i]=3; msh->plc_poly_vertex_list[i*3]=(int)msh->facNodes(i,0)+1; msh->plc_poly_vertex_list[i*3+1]=(int)msh->facNodes(i,1)+1; msh->plc_poly_vertex_list[i*3+2]=(int)msh->facNodes(i,2)+1; }
    std::map<size_t,double> cx,cy,cz; std::map<size_t,int> cn;
    if(msh->nTetras>0&&msh->nNodes>0&&(size_t)msh->nNodes<=msh->nodPos.n_rows){
        for(size_t i=0;i<msh->nTetras;i++){ size_t lab=msh->tetLab(i); if(lab==msh->maxLab)continue; double x=0,y=0,z=0; bool ok=true; for(int j=0;j<4;j++){size_t ni=msh->tetNodes(i,j); if(ni<(size_t)msh->nNodes){ x+=msh->nodPos(ni,0); y+=msh->nodPos(ni,1); z+=msh->nodPos(ni,2); } else ok=false; } if(!ok)continue; cx[lab]+=x; cy[lab]+=y; cz[lab]+=z; cn[lab]++; }
        int nr=(int)cx.size(); if(nr>0){ msh->plc_regions.resize(nr*5); int idx=0; for(auto& kv:cx){ msh->plc_regions[idx*5]=kv.second/(4*cn[kv.first]); msh->plc_regions[idx*5+1]=cy[kv.first]/(4*cn[kv.first]); msh->plc_regions[idx*5+2]=cz[kv.first]/(4*cn[kv.first]); msh->plc_regions[idx*5+3]=(double)kv.first; msh->plc_regions[idx*5+4]=0.0; idx++; } }
    }
}
