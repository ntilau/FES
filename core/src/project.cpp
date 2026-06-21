#include <string>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <algorithm>
#include <map>
#include <utility>
#include <set>

#include "configuration.h"
#include "memory.h"
#include "project.h"
#include "option.h"
#include "pre_processor.h"
#include "mesh.h"

namespace {

// ── XML helpers (no external dependency, handles our limited schema) ──

// Write XML-escaped text to an ostream
static void xmlEscape(std::ostream& out, const std::string& s) {
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
}

struct xml_writer {
    std::ostream& out;
    int indent = 0;

    explicit xml_writer(std::ostream& o) : out(o) {}

    void pi(const std::string& target) {
        out << "<?" << target << "?>\n";
    }

    void open(const std::string& name) {
        out << std::string(indent, ' ') << '<' << name << ">\n";
        indent += 2;
    }

    void close(const std::string& name) {
        indent -= 2;
        out << std::string(indent, ' ') << "</" << name << ">\n";
    }

    // Element with text content: <name>content</name>
    void text(const std::string& name, const std::string& content) {
        out << std::string(indent, ' ') << '<' << name << '>';
        xmlEscape(out, content);
        out << "</" << name << ">\n";
    }

    void text(const std::string& name, double val, int prec = 17) {
        std::ostringstream ss;
        ss << std::setprecision(prec) << val;
        out << std::string(indent, ' ') << '<' << name << '>'
            << ss.str() << "</" << name << ">\n";
    }

    void text(const std::string& name, size_t val) {
        out << std::string(indent, ' ') << '<' << name << '>'
            << val << "</" << name << ">\n";
    }

    void text(const std::string& name, bool val) {
        out << std::string(indent, ' ') << '<' << name << '>'
            << (val ? "1" : "0") << "</" << name << ">\n";
    }

    // Self-closing element with attributes
    void leaf(const std::string& name) {
        out << std::string(indent, ' ') << '<' << name << "/>\n";
    }

    // Helper to build attr="val" pairs
    struct Attr {
        std::string k, v;
    };

    void leaf(const std::string& name, std::initializer_list<Attr> attrs) {
        out << std::string(indent, ' ') << '<' << name;
        for(const auto& a : attrs) {
            out << ' ' << a.k << "=\"";
            xmlEscape(out, a.v);
            out << '"';
        }
        out << "/>\n";
    }
};

// ── Minimal XML pull-parser ──
// Reads from an istream line by line. Handles:
//  <tag>text</tag>   (opening/closing, with text content)
//  <tag att="val"/>  (self-closing with attributes)
//  <!-- comment -->  (skipped)
//  <?pi ... ?>       (skipped)
// State machine: collects text between tags word by word.
class xml_parser {
    std::istream& in;
    // Current tag context stack
    std::vector<std::string> tagStack;
    // Current accumulated text content between tags
    std::string currentText;
    // Buffer of unprocessed lines
    std::string lineBuf;
    size_t linePos = 0;
    bool hasMore = true;

    // Fill lineBuf from in
    bool fillLine() {
        while(hasMore) {
            if(!std::getline(in, lineBuf)) {
                hasMore = false;
                return false;
            }
            // Strip trailing \r if any
            if(!lineBuf.empty() && lineBuf.back() == '\r')
                lineBuf.pop_back();
            linePos = 0;
            return true;
        }
        return false;
    }

    // Skip whitespace in lineBuf, filling a new line if needed
    void skipWs() {
        while(true) {
            while(linePos < lineBuf.size() && (lineBuf[linePos] == ' ' || lineBuf[linePos] == '\t'))
                linePos++;
            if(linePos < lineBuf.size()) break;
            if(!fillLine()) break;
        }
    }

    // Peek at current char
    int peek() {
        if(linePos >= lineBuf.size()) {
            if(!fillLine()) return EOF;
        }
        return (unsigned char)lineBuf[linePos];
    }

    // Advance and return current char
    int next() {
        int c = peek();
        if(c != EOF) linePos++;
        return c;
    }

    // Expect a specific char, advance past it
    bool expect(char c) {
        if(peek() != c) return false;
        linePos++;
        return true;
    }

public:
    // Tokens returned by nextToken()
    enum token_type {
        tok_open,       // <tag
        tok_close,      // </tag>
        tok_selfclose,  // <tag ... />
        tok_text,       // text content
        tok_comment,    // <!-- comment -->
        tok_pi,         // <?...?>
        tok_eof
    };

    struct Token {
        token_type type;
        std::string name;       // tag name for open/close/selfclose
        std::map<std::string, std::string> attrs;  // attributes
        std::string text;       // content for tok_text
    };

    explicit xml_parser(std::istream& is) : in(is) {}

    Token nextToken() {
        skipWs();
        int c = peek();
        if(c == EOF) return {tok_eof};

        if(c == '<') {
            next(); // consume '<'

            // Comment: <!-- ... -->
            if(peek() == '!' && linePos + 1 < lineBuf.size() && lineBuf[linePos+1] == '-' && lineBuf[linePos+2] == '-') {
                linePos += 3; // skip '!--'
                std::string comment;
                while(true) {
                    if(linePos + 2 < lineBuf.size() && lineBuf[linePos] == '-' && lineBuf[linePos+1] == '-' && lineBuf[linePos+2] == '>') {
                        linePos += 3;
                        break;
                    }
                    if(linePos >= lineBuf.size()) {
                        if(!fillLine()) break;
                        comment += '\n';
                        continue;
                    }
                    comment += (char)lineBuf[linePos++];
                }
                skipWs();
                return {tok_comment, "", {}, comment};
            }

            // Processing instruction: <? ... ?>
            if(peek() == '?') {
                next(); // consume '?'
                std::string pi;
                while(true) {
                    if(linePos + 1 < lineBuf.size() && lineBuf[linePos] == '?' && lineBuf[linePos+1] == '>') {
                        linePos += 2;
                        break;
                    }
                    if(linePos >= lineBuf.size()) {
                        if(!fillLine()) break;
                        pi += '\n';
                        continue;
                    }
                    pi += (char)lineBuf[linePos++];
                }
                skipWs();
                return {tok_pi, "", {}, pi};
            }

            // Closing tag: </name>
            if(peek() == '/') {
                next(); // consume '/'
                std::string name;
                while(peek() != EOF && peek() != '>' && peek() != ' ' && peek() != '\t') {
                    name += (char)next();
                }
                skipWs();
                if(peek() == '>') next(); // consume '>'
                skipWs();
                return {tok_close, name};
            }

            // Opening tag: <name ... [/>]
            std::string name;
            while(peek() != EOF && peek() != '>' && peek() != '/' && peek() != ' ' && peek() != '\t') {
                name += (char)next();
            }

            std::map<std::string, std::string> attrs;
            skipWs();

            while(peek() != EOF && peek() != '>' && peek() != '/') {
                // Read attribute name
                std::string attrName;
                while(peek() != EOF && peek() != '=' && peek() != ' ' && peek() != '\t' && peek() != '>' && peek() != '/') {
                    attrName += (char)next();
                }
                skipWs();
                if(peek() == '=') {
                    next(); // consume '='
                    skipWs();
                    if(peek() == '"') {
                        next(); // consume opening quote
                        std::string attrVal;
                        while(peek() != EOF && peek() != '"') {
                            // Handle XML entities
                            if(peek() == '&') {
                                next(); // consume '&'
                                std::string ent;
                                while(peek() != EOF && peek() != ';') {
                                    ent += (char)next();
                                }
                                if(peek() == ';') next(); // consume ';'
                                if(ent == "amp") attrVal += '&';
                                else if(ent == "lt") attrVal += '<';
                                else if(ent == "gt") attrVal += '>';
                                else if(ent == "quot") attrVal += '"';
                                else if(ent == "apos") attrVal += '\'';
                                else { attrVal += '&'; attrVal += ent; attrVal += ';'; }
                            } else {
                                attrVal += (char)next();
                            }
                        }
                        if(peek() == '"') next(); // consume closing quote
                        attrs[attrName] = attrVal;
                    }
                }
                skipWs();
            }

            bool selfClosing = false;
            if(peek() == '/') {
                selfClosing = true;
                next(); // consume '/'
            }
            if(peek() == '>') next(); // consume '>'
            skipWs();

            if(selfClosing) {
                return {tok_selfclose, name, attrs};
            } else {
                return {tok_open, name, attrs};
            }
        }

        // Text content
        std::string text;
        while(peek() != EOF && peek() != '<') {
            text += (char)next();
        }
        // Trim whitespace from text
        auto start = text.find_first_not_of(" \t\r\n");
        auto end = text.find_last_not_of(" \t\r\n");
        if(start == std::string::npos) text.clear();
        else text = text.substr(start, end - start + 1);
        skipWs();
        return {tok_text, "", {}, text};
    }
};


// ── XML mesh writer helpers ──
// Write a 2-column or 3-column Armadillo Mat/UMat as <row> elements with space-separated values.
// If cols==0, detect from n_cols.
template<typename T>
static void writeMatRowsXml(xml_writer& xml, const std::string& elementName,
                            const arma::Mat<T>& m, const std::string& countAttr = "")
{
    arma::uword cols = m.n_cols;
    std::string attr;
    if(!countAttr.empty()) attr = countAttr;
    else {
        attr = "count=\"" + std::to_string(m.n_rows) + "\"";
    }
    // Write opening tag with attributes
    xml.out << std::string(xml.indent, ' ') << '<' << elementName << ' ' << attr << ">\n";
    xml.indent += 2;
    for(arma::uword r = 0; r < m.n_rows; r++) {
        xml.out << std::string(xml.indent, ' ') << "<row>";
        for(arma::uword c = 0; c < cols; c++) {
            if(c > 0) xml.out << ' ';
            xml.out << m.at(r, c);
        }
        xml.out << "</row>\n";
    }
    xml.indent -= 2;
    xml.out << std::string(xml.indent, ' ') << "</" << elementName << ">\n";
}

// Write a uvec as <row> elements
static void writeUvecXml(xml_writer& xml, const std::string& elementName,
                         const arma::uvec& v, const std::string& countAttr = "")
{
    std::string attr;
    if(!countAttr.empty()) attr = countAttr;
    else {
        attr = "count=\"" + std::to_string(v.n_elem) + "\"";
    }
    xml.out << std::string(xml.indent, ' ') << '<' << elementName << ' ' << attr << ">\n";
    xml.indent += 2;
    for(arma::uword r = 0; r < v.n_elem; r++) {
        xml.out << std::string(xml.indent, ' ') << "<row>" << v(r) << "</row>\n";
    }
    xml.indent -= 2;
    xml.out << std::string(xml.indent, ' ') << "</" << elementName << ">\n";
}

// Write a vector<double> as <row> elements, each row printing cols values
static void writeDoubleVecXml(xml_writer& xml, const std::string& elementName,
                              const std::vector<double>& v, int cols,
                              const std::string& countAttr = "")
{
    std::string attr;
    if(!countAttr.empty()) attr = countAttr;
    else {
        size_t n = v.size() / (size_t)cols;
        attr = "count=\"" + std::to_string(n) + "\"";
    }
    xml.out << std::string(xml.indent, ' ') << '<' << elementName << ' ' << attr << ">\n";
    xml.indent += 2;
    size_t nrows = v.size() / (size_t)cols;
    for(size_t r = 0; r < nrows; r++) {
        xml.out << std::string(xml.indent, ' ') << "<row>";
        for(int c = 0; c < cols; c++) {
            if(c > 0) xml.out << ' ';
            xml.out << std::setprecision(17) << v[r * (size_t)cols + (size_t)c];
        }
        xml.out << "</row>\n";
    }
    xml.indent -= 2;
    xml.out << std::string(xml.indent, ' ') << "</" << elementName << ">\n";
}

// Write a vector<int> as <row> elements, each row printing cols values
static void writeIntVecXml(xml_writer& xml, const std::string& elementName,
                           const std::vector<int>& v, int cols,
                           const std::string& countAttr = "")
{
    std::string attr;
    if(!countAttr.empty()) attr = countAttr;
    else {
        size_t n = v.size() / (size_t)cols;
        attr = "count=\"" + std::to_string(n) + "\"";
    }
    xml.out << std::string(xml.indent, ' ') << '<' << elementName << ' ' << attr << ">\n";
    xml.indent += 2;
    size_t nrows = v.size() / (size_t)cols;
    for(size_t r = 0; r < nrows; r++) {
        xml.out << std::string(xml.indent, ' ') << "<row>";
        for(int c = 0; c < cols; c++) {
            if(c > 0) xml.out << ' ';
            xml.out << v[r * (size_t)cols + (size_t)c];
        }
        xml.out << "</row>\n";
    }
    xml.indent -= 2;
    xml.out << std::string(xml.indent, ' ') << "</" << elementName << ">\n";
}

// ── XML mesh writer: writes mesh data as XML elements ──

static void writeMeshXml(xml_writer& xml, mesh* msh)
{
    xml.open("mesh");

    // Dimension hint
    {
        std::ostringstream dimStr;
        dimStr << (msh->mesh_dim > 0 ? msh->mesh_dim : 3);
        xml.leaf("dimension", {{"value", dimStr.str()}});
    }

    // Nodes
    if(msh->nNodes > 0 && msh->nodPos.n_elem > 0) {
        writeMatRowsXml(xml, "nodes", msh->nodPos);
    }

    // Edges
    if(msh->nEdges > 0 && msh->edgNodes.n_elem > 0) {
        writeMatRowsXml(xml, "edges", msh->edgNodes);
    }

    // Edge labels (2D boundary markers)
    if(msh->edgLab.n_elem > 0) {
        writeUvecXml(xml, "edge_labels", msh->edgLab);
    }

    // Faces
    if(msh->nFaces > 0 && msh->facNodes.n_elem > 0) {
        writeMatRowsXml(xml, "faces", msh->facNodes);
    }

    // Face labels (boundary markers)
    if(msh->facLab.n_elem > 0) {
        writeUvecXml(xml, "face_labels", msh->facLab);
    }

    // Tetras
    if(msh->nTetras > 0 && msh->tetNodes.n_elem > 0) {
        writeMatRowsXml(xml, "tetras", msh->tetNodes,
                        "count=\"" + std::to_string(msh->nTetras) + "\"");
    }

    // Tetra labels (region markers)
    if(msh->tetLab.n_elem > 0) {
        writeUvecXml(xml, "tetra_labels", msh->tetLab);
    }

    // ── PLC geometry sections ──
    if(msh->plc_valid) {
        xml.open("plc");

        // poly: PLC point list marker tag name hack — use "plc_points_comment"

        if(!msh->plc_points.empty()) {
            writeDoubleVecXml(xml, "plc_points", msh->plc_points, 3);
        }
        if(!msh->plc_facet_markers.empty() || !msh->plc_poly_vertex_counts.empty()) {
            xml.open("plc_facets");
            if(!msh->plc_facet_markers.empty()) {
                xml.text("nfacets", msh->plc_facet_markers.size());
                writeIntVecXml(xml, "facet_markers", msh->plc_facet_markers, 1);
            }
            if(!msh->plc_poly_vertex_counts.empty()) {
                xml.text("npolygons", msh->plc_poly_vertex_counts.size());
                writeIntVecXml(xml, "polygon_vertex_counts", msh->plc_poly_vertex_counts, 1);
            }
            if(!msh->plc_poly_vertex_list.empty()) {
                writeIntVecXml(xml, "polygon_vertex_list", msh->plc_poly_vertex_list, 1);
            }
            if(!msh->plc_facet_holes.empty()) {
                writeDoubleVecXml(xml, "facet_holes", msh->plc_facet_holes, 3);
            }
            xml.close("plc_facets");
        }
        if(!msh->plc_volume_holes.empty()) {
            writeDoubleVecXml(xml, "volume_holes", msh->plc_volume_holes, 3);
        }
        if(!msh->plc_regions.empty()) {
            writeDoubleVecXml(xml, "plc_regions", msh->plc_regions, 5);
        }
        if(!msh->plc_2d_points.empty()) {
            writeDoubleVecXml(xml, "plc_2d_points", msh->plc_2d_points, 2);
        }
        if(!msh->plc_segments.empty()) {
            xml.open("plc_2d_segments");
            writeIntVecXml(xml, "segments", msh->plc_segments, 2);
            if(!msh->plc_seg_markers.empty()) {
                writeIntVecXml(xml, "seg_markers", msh->plc_seg_markers, 1);
            }
            xml.close("plc_2d_segments");
        }
        if(!msh->plc_2d_holes.empty()) {
            writeDoubleVecXml(xml, "plc_2d_holes", msh->plc_2d_holes, 2);
        }
        if(!msh->plc_2d_regions.empty()) {
            writeDoubleVecXml(xml, "plc_2d_regions", msh->plc_2d_regions, 4);
        }

        xml.close("plc");
    }

    xml.close("mesh");
}

// Split a space-separated string into value tokens
static std::vector<std::string> splitSpace(const std::string& s) {
    std::vector<std::string> out;
    std::istringstream iss(s);
    std::string tok;
    while(iss >> tok) out.push_back(tok);
    return out;
}

// ── bc type name helpers (shared with XML parser) ──

static bc::bcTYPE bcTypeFromName(const std::string& s) {
    if(s == "perfect_e")    return bc::perfect_e;
    if(s == "perfect_h")    return bc::perfect_h;
    if(s == "radiation")   return bc::radiation;
    if(s == "wave_port")    return bc::wave_port;
    if(s == "lumped_port")  return bc::lumped_port;
    return bc::perfect_e;
}

static const char* bcTypeName(bc::bcTYPE t) {
    switch(t) {
        case bc::perfect_e:    return "perfect_e";
        case bc::perfect_h:    return "perfect_h";
        case bc::radiation:   return "radiation";
        case bc::wave_port:    return "wave_port";
        case bc::lumped_port:  return "lumped_port";
        default:              return "perfect_e";
    }
}

// ── XML header + mesh write ──

void writeXmlHeader(std::ofstream& out, const option& opt,
                    const std::vector<mtrl>& mtrls,
                    const std::vector<bc>& bcs,
                    mesh* msh = nullptr)
{
    xml_writer xml(out);
    xml.pi("xml version=\"1.0\" encoding=\"UTF-8\"");
    xml.open("fes version=\"3\" tool=\"FES\"");

    xml.open("options");
    opt.serialize(out);
    xml.close("options");

    // Regions
    xml.open("regions");
    for(const auto& m : mtrls) {
        std::ostringstream epsr, mur, sigma, tand;
        epsr << std::setprecision(17) << m.epsr;
        mur  << std::setprecision(17) << m.mur;
        sigma<< std::setprecision(17) << m.sigma;
        tand << std::setprecision(17) << m.tand;
        xml.leaf("region", {
            {"name", m.sld_name},
            {"label", m.name},
            {"epsr", epsr.str()},
            {"mur", mur.str()},
            {"sigma", sigma.str()},
            {"tand", tand.str()}
        });
    }
    xml.close("regions");

    // Boundaries
    xml.open("boundaries");
    for(const auto& bc : bcs) {
        if(bc.type == bc::wave_port) {
            xml.leaf("boundary", {
                {"name", bc.name},
                {"label", std::to_string(bc.label)},
                {"type", bcTypeName(bc.type)},
                {"modes", std::to_string(bc.num_modes)}
            });
        } else if(bc.type == bc::lumped_port) {
            std::ostringstream imp;
            imp << std::setprecision(17) << bc.impedance;
            xml.leaf("boundary", {
                {"name", bc.name},
                {"label", std::to_string(bc.label)},
                {"type", bcTypeName(bc.type)},
                {"impedance", imp.str()}
            });
        } else {
            xml.leaf("boundary", {
                {"name", bc.name},
                {"label", std::to_string(bc.label)},
                {"type", bcTypeName(bc.type)}
            });
        }
    }
    xml.close("boundaries");

    // Mesh data (pure XML — no more binary sections)
    if(msh) {
        writeMeshXml(xml, msh);
    }

    xml.close("fes");
}

// ── Read entire .fes (pure XML) into string and parse ──
// Returns the full file text. Handles both old hybrid format
// (magic + XML + binary) and new pure-XML format seamlessly.
static std::string readFesXmlString(std::ifstream& in) {
    // Read whole file into a string
    std::string content;
    std::string line;
    while(std::getline(in, line)) {
        if(!line.empty() && line.back() == '\r')
            line.pop_back();
        content += line;
        content += '\n';
    }
    // Rewind for any subsequent reads (not strictly needed — we own the stream)
    // Check for old-format binary header (FES magic bytes) and strip it.
    // The old format: 3-byte magic "FES" followed by XML text.
    // If the XML header prefix isn't present, the file is corrupt.
    if(content.size() >= 3 && content[0] == 'F' && content[1] == 'E' && content[2] == 'S') {
        // Strip binary magic + any preceding binary clutter before XML
        auto xmlPos = content.find("<?xml");
        if(xmlPos != std::string::npos) {
            content = content.substr(xmlPos);
        }
    }
    // Also strip anything after and including <!--MESH--> if present (old-format binary tail)
    auto meshSentinel = content.find("<!--MESH-->");
    if(meshSentinel != std::string::npos) {
        // Check if there's XML mesh data before the sentinel — if so, keep only up to sentinel
        content = content.substr(0, meshSentinel);
    }
    return content;
}

// ── XML mesh reader ──
// Parses <mesh>...</mesh> elements and populates mesh fields.
// Caller must have set msh->nNodes, nEdges, nFaces, nTetras first
// (from the element attributes), or we'll set them from row counts.
static void readMeshXml(xml_parser& parser, mesh* msh) {
    bool inNodes = false, inEdges = false, inEdgeLabels = false;
    bool inFaces = false, inFaceLabels = false;
    bool inTetras = false, inTetraLabels = false;
    bool inPlc = false, inPlcFacets = false, inPlc2dSegs = false;

    // Accumulators for matrix data while reading <row> elements
    std::vector<double> nodeBuf;
    std::vector<double> edgeBuf;
    std::vector<double> faceBuf;
    std::vector<double> tetraBuf;
    arma::uword nodeCols = 0, edgeCols = 0, faceCols = 0, tetraCols = 0;
    arma::uword facLabIdx = 0, tetLabIdx = 0, edgLabIdx = 0;

    // Currently reading container tag (for elements nested in options/regions/boundaries
    // we recognize inside the fes root, but for mesh elements we use the pattern:
    // open → close for each section, reading <row> text in between)
    std::string currentContainer;

    while(true) {
        auto tok = parser.nextToken();
        if(tok.type == xml_parser::tok_eof) break;
        if(tok.type == xml_parser::tok_comment || tok.type == xml_parser::tok_pi) continue;

        // Called after <mesh> was already consumed by readFesXml
        if(tok.type == xml_parser::tok_open) {
            if(tok.name == "mesh") {
                // Safety: skip re-entrant <mesh> if it somehow appears nested
                continue;
            }
            // Everything inside <mesh> is mesh data
            {
                if(tok.name == "dimension") {
                    // self-closing or text — handled below
                } else if(tok.name == "nodes")           { inNodes = true; nodeBuf.clear(); }
                else if(tok.name == "edges")             { inEdges = true; edgeBuf.clear(); }
                else if(tok.name == "edge_labels")       { inEdgeLabels = true; }
                else if(tok.name == "faces")             { inFaces = true; faceBuf.clear(); }
                else if(tok.name == "face_labels")       { inFaceLabels = true; }
                else if(tok.name == "tetras")            { inTetras = true; tetraBuf.clear(); }
                else if(tok.name == "tetra_labels")      { inTetraLabels = true; }
                else if(tok.name == "plc")               { inPlc = true; msh->plc_valid = true; }
                else if(tok.name == "plc_facets" && inPlc) { inPlcFacets = true; }
                else if(tok.name == "plc_2d_segments" && inPlc) { inPlc2dSegs = true; }
                // All other elements inside mesh get a container tag tracked
                else { currentContainer = tok.name; }
            }
        }
        else if(tok.type == xml_parser::tok_close) {
            if(tok.name == "mesh") {
                break; // mesh done
            }
            if(tok.name == "nodes") {
                inNodes = false;
                arma::uword cols = nodeCols > 0 ? nodeCols : 3;
                arma::uword rows = (arma::uword)(nodeBuf.size() / (size_t)cols);
                if(rows > 0 && !nodeBuf.empty()) {
                    msh->nNodes = (size_t)rows;
                    msh->nodPos.set_size(rows, cols);
                    for(arma::uword r = 0; r < rows; r++)
                        for(arma::uword c = 0; c < cols; c++)
                            msh->nodPos(r, c) = nodeBuf[(size_t)(r * cols + c)];
                }
            }
            else if(tok.name == "edges") {
                inEdges = false;
                arma::uword cols = edgeCols > 0 ? edgeCols : 2;
                arma::uword rows = (arma::uword)(edgeBuf.size() / (size_t)cols);
                if(rows > 0 && !edgeBuf.empty()) {
                    msh->nEdges = (size_t)rows;
                    msh->edgNodes.set_size(rows, cols);
                    for(arma::uword r = 0; r < rows; r++)
                        for(arma::uword c = 0; c < cols; c++)
                            msh->edgNodes(r, c) = (arma::uword)edgeBuf[(size_t)(r * cols + c)];
                }
            }
            else if(tok.name == "edge_labels") {
                inEdgeLabels = false;
            }
            else if(tok.name == "faces") {
                inFaces = false;
                arma::uword cols = faceCols > 0 ? faceCols : 3;
                arma::uword rows = (arma::uword)(faceBuf.size() / (size_t)cols);
                if(rows > 0 && !faceBuf.empty()) {
                    msh->nFaces = (size_t)rows;
                    msh->facNodes.set_size(rows, cols);
                    for(arma::uword r = 0; r < rows; r++)
                        for(arma::uword c = 0; c < cols; c++)
                            msh->facNodes(r, c) = (arma::uword)faceBuf[(size_t)(r * cols + c)];
                }
                // Initialize derived arrays
                msh->facEdges.reset();
                msh->facAdjTet.set_size(msh->nFaces);
                msh->facLab.resize(msh->nFaces);
                msh->facLab.fill(msh->maxLab);
            }
            else if(tok.name == "face_labels") {
                inFaceLabels = false;
            }
            else if(tok.name == "tetras") {
                inTetras = false;
                arma::uword cols = tetraCols > 0 ? tetraCols : 4;
                arma::uword rows = (arma::uword)(tetraBuf.size() / (size_t)cols);
                if(rows > 0 && !tetraBuf.empty()) {
                    msh->nTetras = (size_t)rows;
                    msh->tetNodes.set_size(rows, cols);
                    for(arma::uword r = 0; r < rows; r++)
                        for(arma::uword c = 0; c < cols; c++)
                            msh->tetNodes(r, c) = (arma::uword)tetraBuf[(size_t)(r * cols + c)];
                    msh->tetLab.resize(msh->nTetras);
                    msh->tetLab.fill(msh->maxLab);
                }
            }
            else if(tok.name == "tetra_labels") {
                inTetraLabels = false;
            }
            else if(tok.name == "plc") { inPlc = false; }
            else if(tok.name == "plc_facets") { inPlcFacets = false; }
            else if(tok.name == "plc_2d_segments") { inPlc2dSegs = false; }
            else { currentContainer.clear(); }
        }
        else if(tok.type == xml_parser::tok_selfclose) {
            if(tok.name == "dimension") {
                auto it = tok.attrs.find("value");
                if(it != tok.attrs.end()) {
                    try { msh->mesh_dim = std::stoi(it->second); } catch(...) {}
                }
            }
        }
        else if(tok.type == xml_parser::tok_text) {
            // <row> ... </row> text within a data container
            if(inNodes) {
                auto parts = splitSpace(tok.text);
                if(!parts.empty()) {
                    if(nodeCols == 0) nodeCols = (arma::uword)parts.size();
                    for(const auto& p : parts) {
                        try { nodeBuf.push_back(std::stod(p)); } catch(...) {}
                    }
                }
            }
            else if(inEdges) {
                auto parts = splitSpace(tok.text);
                if(!parts.empty()) {
                    if(edgeCols == 0) edgeCols = (arma::uword)parts.size();
                    for(const auto& p : parts) {
                        try { edgeBuf.push_back((double)std::stoull(p)); } catch(...) {}
                    }
                }
            }
            else if(inEdgeLabels) {
                auto parts = splitSpace(tok.text);
                if(!parts.empty()) {
                    arma::uword needed = edgLabIdx + (arma::uword)parts.size();
                    if(needed > msh->edgLab.n_elem)
                        msh->edgLab.resize(needed);
                    for(const auto& p : parts) {
                        try { msh->edgLab(edgLabIdx++) = (arma::uword)std::stoull(p); } catch(...) {}
                    }
                }
            }
            else if(inFaces) {
                auto parts = splitSpace(tok.text);
                if(!parts.empty()) {
                    if(faceCols == 0) faceCols = (arma::uword)parts.size();
                    for(const auto& p : parts) {
                        try { faceBuf.push_back((double)std::stoull(p)); } catch(...) {}
                    }
                }
            }
            else if(inFaceLabels) {
                auto parts = splitSpace(tok.text);
                if(!parts.empty()) {
                    arma::uword needed = facLabIdx + (arma::uword)parts.size();
                    if(needed > msh->facLab.n_elem)
                        msh->facLab.resize(needed);
                    for(const auto& p : parts) {
                        try { msh->facLab(facLabIdx++) = (arma::uword)std::stoull(p); } catch(...) {}
                    }
                }
            }
            else if(inTetras) {
                auto parts = splitSpace(tok.text);
                if(!parts.empty()) {
                    if(tetraCols == 0) tetraCols = (arma::uword)parts.size();
                    for(const auto& p : parts) {
                        try { tetraBuf.push_back((double)std::stoull(p)); } catch(...) {}
                    }
                }
            }
            else if(inTetraLabels) {
                auto parts = splitSpace(tok.text);
                if(!parts.empty()) {
                    arma::uword needed = tetLabIdx + (arma::uword)parts.size();
                    if(needed > msh->tetLab.n_elem)
                        msh->tetLab.resize(needed);
                    for(const auto& p : parts) {
                        try { msh->tetLab(tetLabIdx++) = (arma::uword)std::stoull(p); } catch(...) {}
                    }
                }
            }
            // ── PLC data reading ──
            else if(inPlcFacets) {
                auto parts = splitSpace(tok.text);
                if(!parts.empty() && !currentContainer.empty()) {
                    if(currentContainer == "facet_markers") {
                        for(const auto& p : parts) {
                            try { msh->plc_facet_markers.push_back(std::stoi(p)); } catch(...) {}
                        }
                    } else if(currentContainer == "polygon_vertex_counts") {
                        for(const auto& p : parts) {
                            try { msh->plc_poly_vertex_counts.push_back(std::stoi(p)); } catch(...) {}
                        }
                    } else if(currentContainer == "polygon_vertex_list") {
                        for(const auto& p : parts) {
                            try { msh->plc_poly_vertex_list.push_back(std::stoi(p)); } catch(...) {}
                        }
                    } else if(currentContainer == "facet_holes") {
                        for(const auto& p : parts) {
                            try { msh->plc_facet_holes.push_back(std::stod(p)); } catch(...) {}
                        }
                    }
                }
            }
            else if(inPlc2dSegs) {
                auto parts = splitSpace(tok.text);
                if(!parts.empty() && !currentContainer.empty()) {
                    if(currentContainer == "segments") {
                        for(const auto& p : parts) {
                            try { msh->plc_segments.push_back(std::stoi(p)); } catch(...) {}
                        }
                    } else if(currentContainer == "seg_markers") {
                        for(const auto& p : parts) {
                            try { msh->plc_seg_markers.push_back(std::stoi(p)); } catch(...) {}
                        }
                    }
                }
            }
            else if(inPlc && !currentContainer.empty()) {
                auto parts = splitSpace(tok.text);
                if(!parts.empty()) {
                    if(currentContainer == "plc_points") {
                        for(const auto& p : parts) {
                            try { msh->plc_points.push_back(std::stod(p)); } catch(...) {}
                        }
                    } else if(currentContainer == "volume_holes") {
                        for(const auto& p : parts) {
                            try { msh->plc_volume_holes.push_back(std::stod(p)); } catch(...) {}
                        }
                    } else if(currentContainer == "plc_regions") {
                        for(const auto& p : parts) {
                            try { msh->plc_regions.push_back(std::stod(p)); } catch(...) {}
                        }
                    } else if(currentContainer == "plc_2d_points") {
                        for(const auto& p : parts) {
                            try { msh->plc_2d_points.push_back(std::stod(p)); } catch(...) {}
                        }
                    } else if(currentContainer == "plc_2d_holes") {
                        for(const auto& p : parts) {
                            try { msh->plc_2d_holes.push_back(std::stod(p)); } catch(...) {}
                        }
                    } else if(currentContainer == "plc_2d_regions") {
                        for(const auto& p : parts) {
                            try { msh->plc_2d_regions.push_back(std::stod(p)); } catch(...) {}
                        }
                    }
                }
            }
        }
    }
}

// ── Read all XML from .fes ──
void readFesXml(std::ifstream& in, option& opt,
                std::vector<mtrl>& mtrls,
                std::vector<bc>& bcs,
                mesh* msh)
{
    // Read the whole .fes file into a string (handles old hybrid format too)
    std::string content = readFesXmlString(in);

    // Known option keys for warning on unrecognized elements
    static const std::set<std::string> knownoptionKeys = {
        "solver", "assembly", "name", "dbg", "dbl", "niter", "toll",
        "h_ord", "p_ord", "freq", "l_freq", "h_freq", "n_freqs",
        "n_harm", "relax",
        "einc",
        "field", "rad", "n_theta", "n_phi",
        "poly", "poly_cmd",
        "href_cmd",
        "dd", "ddn", "dds", "n_jor_gs", "nl", "n_dd", "power",
        "Ex", "Ey", "Ez", "kx", "ky", "kz"
    };

    std::istringstream headerStream(content);
    xml_parser parser(headerStream);
    bool inoptions = false;
    bool inRegions = false;
    bool inBoundaries = false;
    bool inMesh = false;
    // Current option key being read (for text content between open/close tags)
    std::string curoptionKey;

    while(true) {
        auto tok = parser.nextToken();

        if(tok.type == xml_parser::tok_eof) {
            break;
        }

        if(tok.type == xml_parser::tok_comment || tok.type == xml_parser::tok_pi) {
            continue;
        }

        if(tok.type == xml_parser::tok_open) {
            if(tok.name == "fes") {
                // root element, nothing to do
            } else if(tok.name == "options") {
                inoptions = true;
            } else if(tok.name == "regions") {
                inRegions = true;
            } else if(tok.name == "boundaries") {
                inBoundaries = true;
            } else if(tok.name == "mesh") {
                inMesh = true;
                // Parse mesh elements through a sub-parser so we don't pollute the main state
                readMeshXml(parser, msh);
                inMesh = false;
            } else if(inoptions) {
                // This is an option key with text content (e.g. <solver>direct</solver>)
                curoptionKey = tok.name;
            } else {
                // Unrecognized container element — warn
                std::cerr << "Warning: unrecognized FES element <" << tok.name << ">\n";
            }
            continue;
        }

        if(tok.type == xml_parser::tok_close) {
            if(tok.name == "options") {
                inoptions = false;
            } else if(tok.name == "regions") {
                inRegions = false;
            } else if(tok.name == "boundaries") {
                inBoundaries = false;
            } else if(inoptions) {
                curoptionKey.clear();
            }
            continue;
        }

        if(tok.type == xml_parser::tok_selfclose) {
            if(tok.name == "region" && inRegions) {
                mtrl m;
                m.label = mtrls.size();
                auto it = tok.attrs.find("name");
                if(it != tok.attrs.end()) m.sld_name = it->second;
                it = tok.attrs.find("label");
                if(it != tok.attrs.end()) m.name = it->second;
                it = tok.attrs.find("epsr");
                if(it != tok.attrs.end()) m.epsr = std::stod(it->second);
                it = tok.attrs.find("mur");
                if(it != tok.attrs.end()) m.mur = std::stod(it->second);
                it = tok.attrs.find("sigma");
                if(it != tok.attrs.end()) m.sigma = std::stod(it->second);
                it = tok.attrs.find("tand");
                if(it != tok.attrs.end()) m.tand = std::stod(it->second);
                m.updmtrl();
                mtrls.push_back(m);
            } else if(tok.name == "boundary" && inBoundaries) {
                bc bc;
                auto it = tok.attrs.find("name");
                if(it != tok.attrs.end()) bc.name = it->second;
                it = tok.attrs.find("label");
                if(it != tok.attrs.end()) bc.label = std::stoul(it->second);
                it = tok.attrs.find("type");
                if(it != tok.attrs.end()) bc.type = bcTypeFromName(it->second);
                it = tok.attrs.find("modes");
                if(it != tok.attrs.end()) bc.num_modes = std::stoi(it->second);
                it = tok.attrs.find("impedance");
                if(it != tok.attrs.end()) bc.impedance = std::stod(it->second);
                bcs.push_back(bc);
            } else if(inoptions) {
                if(knownoptionKeys.find(tok.name) == knownoptionKeys.end()) {
                    std::cerr << "Warning: unrecognized FES option <" << tok.name << "/>\n";
                }
            } else if(tok.name == "fes") {
                // self-closing fes — empty file, fine
            } else {
                std::cerr << "Warning: unrecognized FES element <" << tok.name << "/>\n";
            }
            continue;
        }

        if(tok.type == xml_parser::tok_text) {
            if(inoptions && !curoptionKey.empty()) {
                const std::string& val = tok.text;
                const std::string& key = curoptionKey;

                if(knownoptionKeys.find(key) == knownoptionKeys.end()) {
                    std::cerr << "Warning: unrecognized FES option <" << key << ">\n";
                    continue;
                }

                // Map key to option field
                if(key == "solver") {
                    opt.solver = option::solver_type_from_name(val);
                } else if(key == "assembly") {
                    opt.assembly = option::assemb_type_from_name(val);
                } else if(key == "name") {
                    // name comes from CLI, never stored
                } else if(key == "dbg") {
                    opt.dbg = (val == "1");
                } else if(key == "dbl") {
                    opt.dbl = (val == "1");
                } else if(key == "niter") {
                    try { opt.niter = (size_t)std::stoull(val); } catch(...) {}
                } else if(key == "toll") {
                    try { opt.toll = std::stod(val); } catch(...) {}
                } else if(key == "h_ord") {
                    try { opt.h_ord = (size_t)std::stoull(val); } catch(...) {}
                } else if(key == "p_ord") {
                    try { opt.p_ord = (size_t)std::stoull(val); } catch(...) {}
                } else if(key == "freq") {
                    try { opt.freq = std::stod(val); } catch(...) {}
                } else if(key == "l_freq") {
                    try { opt.l_freq = std::stod(val); } catch(...) {}
                } else if(key == "h_freq") {
                    try { opt.h_freq = std::stod(val); } catch(...) {}
                } else if(key == "n_freqs") {
                    try { opt.n_freqs = (size_t)std::stoull(val); } catch(...) {}
                } else if(key == "n_harm") {
                    try { opt.n_harm = (size_t)std::stoull(val); } catch(...) {}
                } else if(key == "relax") {
                    try { opt.relax = std::stod(val); } catch(...) {}
                } else if(key == "einc") {
                    opt.einc = (val == "1");
                } else if(key == "field") {
                    opt.field = (val == "1");
                } else if(key == "rad") {
                    opt.rad = (val == "1");
                } else if(key == "n_theta") {
                    try { opt.n_theta = std::stod(val); } catch(...) {}
                } else if(key == "n_phi") {
                    try { opt.n_phi = std::stod(val); } catch(...) {}
                } else if(key == "poly") {
                    // CLI-only; skip on XML reload (would re-trigger meshing)
                } else if(key == "poly_cmd") {
                    // CLI-only; skip on XML reload
                } else if(key == "href_cmd") {
                    opt.href_cmd = val;
                } else if(key == "dd") {
                    opt.dd = (val == "1");
                } else if(key == "ddn") {
                    opt.ddn = (val == "1");
                } else if(key == "dds") {
                    opt.dds = (val == "1");
                } else if(key == "n_jor_gs") {
                    opt.n_jor_gs = (val == "1");
                } else if(key == "nl") {
                    opt.nl = (val == "1");
                } else if(key == "n_dd") {
                    try { opt.n_dd = (size_t)std::stoull(val); } catch(...) {}
                } else if(key == "power") {
                    try { opt.power = std::stod(val); } catch(...) {}
                } else if(key == "Ex") {
                    try { opt.E[0] = std::stod(val); } catch(...) {}
                } else if(key == "Ey") {
                    try { opt.E[1] = std::stod(val); } catch(...) {}
                } else if(key == "Ez") {
                    try { opt.E[2] = std::stod(val); } catch(...) {}
                } else if(key == "kx") {
                    try { opt.k[0] = std::stod(val); } catch(...) {}
                } else if(key == "ky") {
                    try { opt.k[1] = std::stod(val); } catch(...) {}
                } else if(key == "kz") {
                    try { opt.k[2] = std::stod(val); } catch(...) {}
                } else if(key.substr(0, 5) == "Vbnd:") {
                    try { opt.Vbnd[key.substr(5)] = std::stod(val); } catch(...) {}
                }
            }
            continue;
        }
    }
}

} // anonymous namespace

project::project(std::ofstream& logFile, option& pOpt) : opt(&pOpt), msh(new mesh())
{
    arma::wall_clock prjtt;
    prjtt.tic();
    logFile << "% Loading files:\n";
    logFile << "project: " << opt->name << "\n";
    logFile << "Homogeneous refinement: p = " << opt->p_ord << ", h = " << opt->h_ord << "\n";
    std::cout << "project:   " << opt->name << "\n";
    std::cout << "Main frequency: " << opt->freq << "\n";
    std::cout << "p = " << opt->p_ord << ", h = " << opt->h_ord << "\n";
    if(opt->poly)
    {
        logFile << "Parsing .poly project file\n";
        preprocessing(this);
        save_fes();
    }
    else
    {
        logFile << "Parsing FE project files\n";
        load_fes();

        // ── Apply generic cli_override map after loading .fes XML ──
        // Re-uses the same key→field mapping as readFesXml() does for XML
        // text elements.  cli_override keys match the serialized <key> tag names.
        if(!opt->cli_override.empty()) {
            opt->apply_cli();
            save_fes();
            std::cout << "options updated in " << opt->name << ".fes\n";
        }
    }
    if(opt->h_ord > 0)
    {
        std::cout << "Performing h refinement\n";
        for(size_t i=0; i<opt->h_ord; i++)
        {
            msh->refine_homogeneous();
        }
        save_fes();
    }
    // Normalize node order in edges/faces/tetras so saved mesh is canonical
    msh->normalize_order();
    // Re-save so .fes contains the normalized mesh
    save_fes();
    // mesh statistics
    logFile << "Nodes  = " << msh->nNodes << "\n"
            << "Edges  = " << msh->nEdges << "\n"
            << "Faces  = " << msh->nFaces << "\n"
            << "Tetras = " << msh->nTetras << "\n";
    msh->check_regular(logFile);
    logFile << "++" << prjtt.toc() << " s\n";
    std::cout << "Nodes  = " << msh->nNodes << "\n"
              << "Edges  = " << msh->nEdges << "\n"
              << "Faces  = " << msh->nFaces << "\n"
              << "Tetras = " << msh->nTetras << "\n";
    msh->Savefield(opt->name);
    // mesh partitioning (domain decomposition)
    if(opt->dd)
    {
        msh->Partitionmesh(opt->n_dd);
        logFile << "Domains = " << opt->n_dd << "\n";
        std::cout << "Domains = " << opt->n_dd << "\n";
    }
    mem_stat::print(std::cout);
    mem_stat::print(logFile);
}

project::~project()
{
}

// ── Binary save ──

void project::save_fes()
{
    std::ofstream out(std::string(opt->name + ".fes").c_str());

    // ── Pure XML: options, regions, boundaries, and mesh data ──
    writeXmlHeader(out, *opt, msh->tetmtrl, msh->facbc, msh);

    out.close();
}

// ── Load from pure XML .fes ──

void project::load_fes()
{
    std::ifstream in(std::string(opt->name + ".fes").c_str());
    if(!in.is_open())
        throw std::runtime_error(opt->name + ".fes" + " not available");

    // ── Read full XML: options + regions + boundaries + mesh data ──
    readFesXml(in, *opt, msh->tetmtrl, msh->facbc, msh);

    in.close();

    // ── Rebuild derived connectivity from basic topology ──
    if(msh->nEdges > 0 && msh->edgNodes.n_elem > 0) {
        // Build edge map: sorted node pair → edge index
        std::map<std::pair<size_t,size_t>, size_t> edgesMap;
        for(size_t i = 0; i < msh->nEdges; i++) {
            size_t n0 = msh->edgNodes(i,0);
            size_t n1 = msh->edgNodes(i,1);
            if(n0 > n1) std::swap(n0, n1);
            edgesMap[std::make_pair(n0, n1)] = i;
        }

        // Rebuild facEdges from facNodes + edge map
        if(msh->nFaces > 0) {
            msh->facEdges.set_size(msh->nFaces, 3);
            for(size_t i = 0; i < msh->nFaces; i++) {
                size_t n0 = msh->facNodes(i,0);
                size_t n1 = msh->facNodes(i,1);
                size_t n2 = msh->facNodes(i,2);
                if(n0 > n1) std::swap(n0, n1);
                if(n0 > n2) std::swap(n0, n2);
                if(n1 > n2) std::swap(n1, n2);
                msh->facEdges(i,0) = edgesMap[std::make_pair(n0, n1)];
                msh->facEdges(i,1) = edgesMap[std::make_pair(n0, n2)];
                msh->facEdges(i,2) = edgesMap[std::make_pair(n1, n2)];
            }
        }

        // Rebuild tetEdges, tetFaces, facAdjTet from tetNodes
        if(msh->nTetras > 0) {
            // Build face map: sorted node triple → face index
            std::map<std::pair<size_t,std::pair<size_t,size_t>>, size_t> facesMap;
            for(size_t i = 0; i < msh->nFaces; i++) {
                size_t n0 = msh->facNodes(i,0);
                size_t n1 = msh->facNodes(i,1);
                size_t n2 = msh->facNodes(i,2);
                if(n0 > n1) std::swap(n0, n1);
                if(n0 > n2) std::swap(n0, n2);
                if(n1 > n2) std::swap(n1, n2);
                facesMap[std::make_pair(n0, std::make_pair(n1, n2))] = i;
            }

            msh->tetEdges.set_size(msh->nTetras, 6);
            msh->tetFaces.set_size(msh->nTetras, 4);
            msh->facAdjTet.set_size(msh->nFaces);
            for(size_t i = 0; i < msh->nTetras; i++) {
                size_t n0 = msh->tetNodes(i,0);
                size_t n1 = msh->tetNodes(i,1);
                size_t n2 = msh->tetNodes(i,2);
                size_t n3 = msh->tetNodes(i,3);

                // Sort tetra nodes for consistent edge/face lookup
                size_t tn[4] = {n0, n1, n2, n3};
                std::sort(tn, tn + 4);

                // tetEdges (6 edges of a tetrahedron, sorted)
                msh->tetEdges(i,0) = edgesMap[std::make_pair(tn[0], tn[1])];
                msh->tetEdges(i,1) = edgesMap[std::make_pair(tn[0], tn[2])];
                msh->tetEdges(i,2) = edgesMap[std::make_pair(tn[0], tn[3])];
                msh->tetEdges(i,3) = edgesMap[std::make_pair(tn[1], tn[2])];
                msh->tetEdges(i,4) = edgesMap[std::make_pair(tn[1], tn[3])];
                msh->tetEdges(i,5) = edgesMap[std::make_pair(tn[2], tn[3])];

                // tetFaces (4 faces of a tetrahedron, face opposite each vertex)
                msh->tetFaces(i,0) = facesMap[std::make_pair(tn[1], std::make_pair(tn[2], tn[3]))];
                msh->tetFaces(i,1) = facesMap[std::make_pair(tn[0], std::make_pair(tn[2], tn[3]))];
                msh->tetFaces(i,2) = facesMap[std::make_pair(tn[0], std::make_pair(tn[1], tn[3]))];
                msh->tetFaces(i,3) = facesMap[std::make_pair(tn[0], std::make_pair(tn[1], tn[2]))];

                // facAdjTet: each face gets this tetra added
                for(size_t j = 0; j < 4; j++) {
                    size_t fid = msh->tetFaces(i, j);
                    arma::uvec adj(1);
                    adj(0) = (arma::uword)i;
                    msh->facAdjTet(fid) = arma::join_cols(msh->facAdjTet(fid), adj);
                }
            }
        }
    }

    // ── Rebuild facbc[i].Faces from facLab markers ──
    // Match by label (same as preprocessing.cpp), not by direct index,
    // because facLab stores TetGen markers, not facbc indices.
    for(size_t i = 0; i < msh->facbc.size(); i++) {
        uint64_t cLab = msh->facbc[i].label;
        for(size_t fid = 0; fid < msh->nFaces; fid++) {
            if(cLab == msh->facLab(fid)) {
                arma::uvec& faces = msh->facbc[i].Faces;
                faces.resize(faces.n_elem + 1);
                faces(faces.n_elem - 1) = (arma::uword)fid;
            }
        }
    }

    // ── Rebuild tetmtrl[i].Tetras from tetLab markers ──
    for(size_t i = 0; i < msh->nTetras; i++) {
        uint64_t lab = msh->tetLab(i);
        if(lab < msh->tetmtrl.size()) {
            arma::uvec& tets = msh->tetmtrl[lab].Tetras;
            tets.resize(tets.n_elem + 1);
            tets(tets.n_elem - 1) = (arma::uword)i;
        }
    }
}
