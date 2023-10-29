.SUFFIXES: .cpp .o

CC=g++ -w
RES=windres

BINDIR= ./bin
SRCDIR= ./
OBJDIR= ./obj

CFLAGS = -std=gnu++98 -m64 -fopenmp -static -s -I./ -I./ext/tetgen -I./ext/mumps/include -I./ext/arma/include -I./ext/metis -I./ext/gmm -DTETLIBRARY
LFLAGS = -std=gnu++98 -m64 -fopenmp -static -s -L./ext -lsmumps -ldmumps -lcmumps -lzmumps -lmumps_common -lmpiseq -lpord -lmetis -ltet -larpack -llapack -lopenblas -lgfortran -lquadmath -lpsapi -liphlpapi
SRCS=$(wildcard  $(SRCDIR)/*.cpp)
OBJS=$(patsubst $(SRCDIR)/%.cpp, $(OBJDIR)/%.o, $(SRCS)) $(OBJDIR)/FE.rc.o

ifdef OS
   RM = del /F /S /Q
   FixPath = $(subst /,\,$1)
else
   ifeq ($(shell uname), Linux)
      RM = rm -f
      FixPath = $1
   endif
endif

all: $(BINDIR) $(OBJDIR) $(OBJS)
	$(CC) -o $(BINDIR)/FE $(OBJS) $(LFLAGS)

$(OBJDIR)/%.o : $(SRCDIR)/%.cpp
	$(CC) $(CFLAGS) -c  $< -o $@

$(BINDIR):
	if [ ! -d "$(BINDIR)" ]; then mkdir $(BINDIR); fi

$(OBJDIR):
	if [ ! -d "$(OBJDIR)" ]; then mkdir $(OBJDIR); fi

$(OBJDIR)/%.rc.o : $(SRCDIR)/%.rc
	$(RES) $< -o $@

.PHONY: clean
clean:
	$(RM) $(call FixPath,$(OBJDIR)/*.o)

.PHONY: test
test:
	cd $(BINDIR) && FE.exe WR10 1e10 +p 2


