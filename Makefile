.SUFFIXES: .cpp .o

CC=g++
RES=windres

BINDIR= ./bin
SRCDIR= ./
OBJDIR= ./obj

CFLAGS = -std=gnu++98 -m64 -fopenmp -static -s -I./ -I./ext/tetgen -I./ext/mumps/include -I./ext/arma/include -I./ext/metis -I./ext/gmm -DTETLIBRARY
LFLAGS = -std=gnu++98 -m64 -fopenmp -static -s -L./ext -lsmumps -ldmumps -lcmumps -lzmumps -lmumps_common -lmpiseq -lpord -lmetis -ltet -larpack -llapack -lopenblas -lgfortran -lquadmath -lpsapi -liphlpapi

OBJS = $(addprefix $(OBJDIR)/, AssElStat.o AssLin.o AssLinDD.o AssLinSchur.o AssNL.o BC.o Coupl.o DoF.o Eigen.o EleMat.o EqSys.o Field.o HFSS.o Main.o Mesh.o Mtrl.o MUMPS.o Option.o Project.o Quad.o Rad.o Shape.o TetGen.o FE.rc.o)

ifdef OS
   RM = del /F /S /Q
   FixPath = $(subst /,\,$1)
else
   ifeq ($(shell uname), Linux)
      RM = rm -f
      FixPath = $1
   endif
endif

all: $(OBJS)
	$(CC) -o $(BINDIR)/FE $(OBJS) $(LFLAGS)

$(OBJDIR)/%.o : $(SRCDIR)/%.cpp
	$(CC) $(CFLAGS) -c  $< -o $@

$(OBJDIR)/%.rc.o : $(SRCDIR)/%.rc
	$(RES) $< -o $@

.PHONY: clean
clean:
	$(RM) $(call FixPath,$(OBJDIR)/*.o)


