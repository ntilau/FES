// C entry points around TetGen's C++ interface, for ctypes.
//
// TetGen's library API is built from C++ classes: ``tetgenio`` has a virtual
// destructor and non-trivial constructors, so ctypes cannot own an instance
// safely.  These wrappers keep the objects opaque, hand the output arrays out
// as plain pointers, and turn the ``TETLIBRARY`` abort path (which throws an
// int instead of calling exit) into a return code.
//
// Compiled into the shared library by ``tools/build_tetgen.py``.  This is
// optional tooling, not part of the Python package's build.
#include "tetgen.h"

extern "C" {

void* fes_tetgenio_new(void)
{
    return (void*) new tetgenio();
}

void fes_tetgenio_delete(void* p)
{
    delete (tetgenio*) p;
}

int fes_tetgenio_load_poly(void* p, const char* name)
{
    return ((tetgenio*) p)->load_poly((char*) name) ? 1 : 0;
}

int fes_tetgen_tetrahedralize(const char* switches, void* in, void* out)
{
    try {
        tetrahedralize((char*) switches, (tetgenio*) in, (tetgenio*) out,
                       (tetgenio*) 0, (tetgenio*) 0);
    } catch(int code) {
        return code;
    } catch(...) {
        return -1;
    }
    return 0;
}

int fes_firstnumber(void* p) { return ((tetgenio*) p)->firstnumber; }
int fes_numberofpoints(void* p) { return ((tetgenio*) p)->numberofpoints; }
double* fes_pointlist(void* p) { return ((tetgenio*) p)->pointlist; }
int fes_numberofpointattributes(void* p)
{
    return ((tetgenio*) p)->numberofpointattributes;
}
double* fes_pointattributelist(void* p)
{
    return ((tetgenio*) p)->pointattributelist;
}

int fes_numberofedges(void* p) { return ((tetgenio*) p)->numberofedges; }
int* fes_edgelist(void* p) { return ((tetgenio*) p)->edgelist; }
int* fes_edgemarkerlist(void* p) { return ((tetgenio*) p)->edgemarkerlist; }

int fes_numberoftrifaces(void* p) { return ((tetgenio*) p)->numberoftrifaces; }
int* fes_trifacelist(void* p) { return ((tetgenio*) p)->trifacelist; }
int* fes_trifacemarkerlist(void* p) { return ((tetgenio*) p)->trifacemarkerlist; }

int fes_numberoftetrahedra(void* p) { return ((tetgenio*) p)->numberoftetrahedra; }
int* fes_tetrahedronlist(void* p) { return ((tetgenio*) p)->tetrahedronlist; }
double* fes_tetrahedronattributelist(void* p)
{
    return ((tetgenio*) p)->tetrahedronattributelist;
}
int fes_numberoftetrahedronattributes(void* p)
{
    return ((tetgenio*) p)->numberoftetrahedronattributes;
}

}
