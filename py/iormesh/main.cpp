/*
    Finite Element 2D Mesh interface tool for matlab
	(c) Laurent Ntibarikure
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define INPUTLINESIZE 1024

#ifdef __cplusplus
extern "C" {
#endif
#include "matfiles.h"
#ifdef __cplusplus
}
#endif

struct mesh
{
    int dim;
    int NNODE;
    int NELE;
    int NEDGE;
    double* node;
    double* ele;
    double* edge;
    double* ele2edge;
    double* nodeLab;
    double* eleLab;
    double* edgeLab;
    int nextras;
    int nodemarkers;
    int elemarkers;
    int edgemarkers;
};

char *findfield(char *string)
{
    char *result;
    result = string;
    while((*result != '\0') && (*result != '#')
            && (*result != ' ') && (*result != '\t'))
    {
        result++;
    }
    while((*result != '\0') && (*result != '#')
            && (*result != '.') && (*result != '+') && (*result != '-')
            && ((*result < '0') || (*result > '9')))
    {
        result++;
    }
    if(*result == '#')
    {
        *result = '\0';
    }
    return result;
}

char *readline(char *string, FILE *infile, char *infilename)
{
    char *result;
    do
    {
        result = fgets(string, INPUTLINESIZE, infile);
        if(result == (char *) NULL)
        {
            printf("  Error:  Unexpected end of file idx %s.\n", infilename);
            exit(1);
        }
        while((*result != '\0') && (*result != '#')
                && (*result != '.') && (*result != '+') && (*result != '-')
                && ((*result < '0') || (*result > '9')))
        {
            result++;
        }
    }
    while((*result == '#') || (*result == '\0'));
    return result;
}

void readNodeFile(char* filename, mesh* m, double* scale)
{
//    printf("Opening %s.\n", filename);
    FILE* infile = fopen(filename, "r");
    if(infile == (FILE *) NULL)
    {
        printf("  Error:  Cannot access file %s.\n", filename);
        exit(1);
    }
    char inputline[INPUTLINESIZE];
    char *stringptr = readline(inputline, infile, filename);
    m->NNODE = (int) strtol(stringptr, &stringptr, 0);
    stringptr = findfield(stringptr);
    if(*stringptr == '\0')
    {
        m->dim = 2;
    }
    else
    {
        m->dim = (int) strtol(stringptr, &stringptr, 0);
    }
    stringptr = findfield(stringptr);
    if(*stringptr == '\0')
    {
        m->nextras = 0;
    }
    else
    {
        m->nextras = (int) strtol(stringptr, &stringptr, 0);
    }
    stringptr = findfield(stringptr);
    if(*stringptr == '\0')
    {
        m->nodemarkers = 0;
    }
    else
    {
        m->nodemarkers = (int) strtol(stringptr, &stringptr, 0);
    }
    m->node = (double*) malloc(2*(m->NNODE)*sizeof(double));
    m->nodeLab = (double*) malloc((m->NNODE)*sizeof(double));
    for(int idx = 0; idx < m->NNODE; idx++)
    {
        stringptr = readline(inputline, infile, filename);
        stringptr = findfield(stringptr);
        m->node[2*idx] = *scale * (double) strtod(stringptr, &stringptr);
        stringptr = findfield(stringptr);
        m->node[2*idx+1] = *scale * (double) strtod(stringptr, &stringptr);
        /* Read the vertex attributes. */
        for(int j = 2; j < 2 + m->nextras; j++)
        {
            stringptr = findfield(stringptr);
//            if (*stringptr == '\0')
//            {
//                vertexloop[j] = 0.0;
//            }
//            else
//            {
//                vertexloop[j] = (REAL) strtod(stringptr, &stringptr);
//            }
        }
        if(m->nodemarkers)
        {
            /* Read a vertex marker. */
            stringptr = findfield(stringptr);
            if(*stringptr == '\0')
            {
                m->nodeLab[idx] = 0.0;
            }
            else
            {
                m->nodeLab[idx] = (double) strtol(stringptr, &stringptr, 0);
            }
        }
        else
        {
            /* If no markers are specified idx the file, they default to zero. */
            m->nodeLab[idx] = 0.0;
        }
//        printf("%f %f %d\n", m->node[2*idx], m->node[2*idx+1], m->nodeLab[idx]);
    }
    fclose(infile);
}

void readEleFile(char* elefilename, mesh* m)
{
//    printf("Opening %s.\n", elefilename);
    FILE* infile = fopen(elefilename, "r");
    if(infile == (FILE *) NULL)
    {
        printf("  Error:  Cannot access file %s.\n", elefilename);
        exit(1);
    }
    char inputline[INPUTLINESIZE];
    char *stringptr = readline(inputline, infile, elefilename);
    m->NELE = (int) strtol(stringptr, &stringptr, 0);
    stringptr = findfield(stringptr);
    stringptr = findfield(stringptr);
    if(*stringptr == '\0')
    {
        m->elemarkers = 0;
    }
    else
    {
        m->elemarkers = (int) strtol(stringptr, &stringptr, 0);
    }
    m->ele = (double*) malloc(3*(m->NELE)*sizeof(double));
    m->ele2edge = (double*) malloc(3*(m->NELE)*sizeof(double));
    m->eleLab = (double*) malloc((m->NELE)*sizeof(double));
    for(int idx = 0; idx < m->NELE; idx++)
    {
        stringptr = readline(inputline, infile, elefilename);
        stringptr = findfield(stringptr);
        m->ele[3*idx] = (double) strtod(stringptr, &stringptr);
        stringptr = findfield(stringptr);
        m->ele[3*idx+1] = (double) strtod(stringptr, &stringptr);
        stringptr = findfield(stringptr);
        m->ele[3*idx+2] = (double) strtod(stringptr, &stringptr);
        /* Read the vertex attributes. */
//        for (int j = 2; j < 2 + m->nextras; j++)
//        {
//            stringptr = findfield(stringptr);
////            if (*stringptr == '\0')
////            {
////                vertexloop[j] = 0.0;
////            }
////            else
////            {
////                vertexloop[j] = (REAL) strtod(stringptr, &stringptr);
////            }
//        }
        if(m->elemarkers)
        {
            /* Read a vertex marker. */
            stringptr = findfield(stringptr);
            if(*stringptr == '\0')
            {
                m->eleLab[idx] = 0;
            }
            else
            {
                m->eleLab[idx] = (double) strtol(stringptr, &stringptr, 0);
            }
        }
        else
        {
            /* If no markers are specified idx the file, they default to zero. */
            m->eleLab[idx] = 0;
        }
//        printf("%f %f %f %f\n", m->ele[3*idx], m->ele[3*idx+1], m->ele[3*idx+2], m->eleLab[idx]);
    }
    fclose(infile);
}

void readEdgeFile(char* filename, mesh* m)
{
//    printf("Opening %s.\n", filename);
    FILE* infile = fopen(filename, "r");
    if(infile == (FILE *) NULL)
    {
        printf("  Error:  Cannot access file %s.\n", filename);
        exit(1);
    }
    char inputline[INPUTLINESIZE];
    char *stringptr = readline(inputline, infile, filename);
    m->NEDGE = (int) strtol(stringptr, &stringptr, 0);
    stringptr = findfield(stringptr);
    if(*stringptr == '\0')
    {
        m->edgemarkers = 0;
    }
    else
    {
        m->edgemarkers = (int) strtol(stringptr, &stringptr, 0);
    }
    m->edge = (double*) malloc(2*(m->NEDGE)*sizeof(double));
    m->edgeLab = (double*) malloc((m->NEDGE)*sizeof(double));
    for(int idx = 0; idx < m->NEDGE; idx++)
    {
        stringptr = readline(inputline, infile, filename);
        stringptr = findfield(stringptr);
        m->edge[2*idx] = (double) strtod(stringptr, &stringptr);
        stringptr = findfield(stringptr);
        m->edge[2*idx+1] = (double) strtod(stringptr, &stringptr);
        /* Read the vertex attributes. */
//        for (int j = 2; j < 2 + m->nextras; j++)
//        {
//            stringptr = findfield(stringptr);
////            if (*stringptr == '\0')
////            {
////                vertexloop[j] = 0.0;
////            }
////            else
////            {
////                vertexloop[j] = (REAL) strtod(stringptr, &stringptr);
////            }
//        }
        if(m->edgemarkers)
        {
            /* Read a vertex marker. */
            stringptr = findfield(stringptr);
            if(*stringptr == '\0')
            {
                m->edgeLab[idx] = 0;
            }
            else
            {
                m->edgeLab[idx] = (double) strtol(stringptr, &stringptr, 0);
            }
        }
        else
        {
            /* If no markers are specified idx the file, they default to zero. */
            m->edgeLab[idx] = 0;
        }
//        printf("%f %f %f\n", m->edge[2*idx], m->edge[2*idx+1], m->edgeLab[idx]);
    }
    fclose(infile);
}

void sortMeshIndices(mesh* m)
{
    double tmp = 0.0;
    int idx;
//#pragma omp parallel for shared(m) private(idx)
    for(idx=0; idx<m->NEDGE; idx++)
    {
        if(m->edge[2*idx] > m->edge[2*idx+1])
        {
            tmp = m->edge[2*idx];
            m->edge[2*idx] = m->edge[2*idx+1];
            m->edge[2*idx+1] = tmp;
        }
    }
//#pragma omp parallel for shared(m) private(idx)
    for(idx=0; idx<m->NELE; idx++)
    {
        if(m->ele[3*idx] > m->ele[3*idx+1])
        {
            tmp = m->ele[3*idx];
            m->ele[3*idx] = m->ele[3*idx+1];
            m->ele[3*idx+1] = tmp;
        }
        if(m->ele[3*idx+1] > m->ele[3*idx+2])
        {
            tmp = m->ele[3*idx+1];
            m->ele[3*idx+1] = m->ele[3*idx+2];
            m->ele[3*idx+2] = tmp;
        }
        if(m->ele[3*idx] > m->ele[3*idx+1])
        {
            tmp = m->ele[3*idx];
            m->ele[3*idx] = m->ele[3*idx+1];
            m->ele[3*idx+1] = tmp;
        }
    }
}

void buildEle2Edge(mesh* m)
{
    double e0,e1,e2,s0,s1;
    int idx, sidx;
    for(idx = 0; idx < m->NELE; idx++)
    {
        e0 = m->ele[3*idx];
        e1 = m->ele[3*idx+1];
        e2 = m->ele[3*idx+2];
//#pragma omp parallel for shared(idx,e0,e1,e2,m) private(sidx,s0,s1)
        for(sidx=0; sidx < m->NEDGE; sidx++)
        {
            s0 = m->edge[2*sidx];
            s1 = m->edge[2*sidx+1];
            if(s0 == e0)
            {
                if(s1 == e1)
                {
                    m->ele2edge[3*idx+2] = sidx+1;
                }
                else if(s1 == e2)
                {
                    m->ele2edge[3*idx+1] = -sidx-1;
                }
            }
            else if(s0 == e1)
            {
                if(s1 == e0)
                {
                    m->ele2edge[3*idx+2] = -sidx-1;
                }
                else if(s1 == e2)
                {
                    m->ele2edge[3*idx] = sidx+1;
                }
            }
            else if(s0 == e2)
            {
                if(s1 == e0)
                {
                    m->ele2edge[3*idx+1] = sidx+1;
                }
                else if(s1 == e1)
                {
                    m->ele2edge[3*idx] = -sidx-1;
                }
            }
        } // running all edges
    }
}

int main(int argc, char* argv[])
{
    if(argc <= 1)
    {
        printf("Usage: IOrMesh filename q_a_._A hRefinement scale\n");
        return (0);
    }
    char* filename = argv[1];
    char* cmdstr = argv[2];
    double scale = 1.0;
    int hRefidxement = 1;
    if(argc > 3)
    {
        hRefidxement = atof(argv[3]);
        if(hRefidxement < 1)
        {
            hRefidxement = 1;
        }
    }
    if(argc > 4)
    {
        scale = atof(argv[4]);
    }
    mesh cMesh[hRefidxement];
    int ih = 0;
    char triangleCommand[FILENAME_MAX];
    strcpy(triangleCommand,"triangle -p");
    strcat(triangleCommand,cmdstr);
    strcat(triangleCommand,"DeQ ");
    strcat(triangleCommand,filename);
    strcat(triangleCommand,".poly");
    system(triangleCommand);
    char nodefilename[FILENAME_MAX];
    strcpy(nodefilename,filename);
    strcat(nodefilename,".1.node");
    readNodeFile(nodefilename, &cMesh[ih], &scale);
    char elefilename[FILENAME_MAX];
    strcpy(elefilename,filename);
    strcat(elefilename,".1.ele");
    readEleFile(elefilename, &cMesh[ih]);
    char edgefilename[FILENAME_MAX];
    strcpy(edgefilename,filename);
    strcat(edgefilename,".1.edge");
    readEdgeFile(edgefilename, &cMesh[ih]);
    char deleteCommand[FILENAME_MAX];
    strcpy(deleteCommand,"rm -f ");
    strcat(deleteCommand,filename);
    strcat(deleteCommand,".1.*");
    system(deleteCommand);
    sortMeshIndices(&cMesh[ih]);
    buildEle2Edge(&cMesh[ih]);
    for(int ih=0; ih<hRefidxement; ih++)
    {
        char level[2];
        sprintf(level, "%d", ih+1);
        char meshfilename[FILENAME_MAX];
        strcpy(meshfilename,filename);
        strcat(meshfilename,".h");
        strcat(meshfilename,level);
        strcat(meshfilename,".mat");
        int* err = 0;
        MATFILE* data = openmatfile(meshfilename, err);
        matfile_addmatrix(data, (char*)"node", cMesh[ih].node, cMesh[ih].NNODE, 2, 0);
        matfile_addmatrix(data, (char*)"nlab", cMesh[ih].nodeLab, cMesh[ih].NNODE, 1, 0);
        matfile_addmatrix(data, (char*)"ele", cMesh[ih].ele, cMesh[ih].NELE, 3, 0);
        matfile_addmatrix(data, (char*)"elab", cMesh[ih].eleLab, cMesh[ih].NELE, 1, 0);
        matfile_addmatrix(data, (char*)"spig", cMesh[ih].ele2edge, cMesh[ih].NELE, 3, 0);
        matfile_addmatrix(data, (char*)"spig2", cMesh[ih].edge, cMesh[ih].NEDGE, 2, 0);
        matfile_addmatrix(data, (char*)"slab", cMesh[ih].edgeLab, cMesh[ih].NEDGE, 1, 0);
        matfile_close(data);
    }
    return 0;
}
