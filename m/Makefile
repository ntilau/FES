# mFES - MATLAB Finite Element Solver
# =====================================
# IOrMesh : 2D mesh interface tool for MATLAB (Triangle wrapper)
#   (c) Laurent Ntibarikure
#
# Targets:
#   make IOrMesh     Build mesh generator (C++/C)
#   make triangle    Build Triangle library binary (called by IOrMesh)
#   make all         Build both
#   make test        Run MATLAB/Octave tests
#   make clean       Remove build artifacts

CXX      ?= g++
CC       ?= gcc
CFLAGS   ?= -Wall -O3 -ffast-math -march=native -Wno-deprecated
LDFLAGS  ?=

IORMESH_DIR = FEpre/iormesh-src
IORMESH_BIN = FEpre/iormesh-src/IOrMesh

# ─── IOrMesh (mesh generator) ─────────────────────────────────────────
# Note: macOS APFS is case-insensitive; avoid name collision between
# the source dir (iormesh-src/) and the installed binary (IOrMesh).

IOrMesh: $(IORMESH_BIN)

$(IORMESH_BIN): $(IORMESH_DIR)/main.o $(IORMESH_DIR)/matfiles.o
	$(CXX) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(IORMESH_DIR)/main.o: $(IORMESH_DIR)/main.c $(IORMESH_DIR)/matfiles.h
	$(CXX) $(CFLAGS) -c -o $@ $<

$(IORMESH_DIR)/matfiles.o: $(IORMESH_DIR)/matfiles.c $(IORMESH_DIR)/matfiles.h
	$(CC) $(CFLAGS) -c -o $@ $<

# ─── Triangle (called by IOrMesh as `triangle ...`) ──────────────────
triangle: $(IORMESH_DIR)/triangle

$(IORMESH_DIR)/triangle: $(IORMESH_DIR)/triangle.c $(IORMESH_DIR)/triangle.h
	$(CC) $(CFLAGS) -o $@ $(IORMESH_DIR)/triangle.c -lm

# ─── All ────────────────────────────────────────────────────────────────
all: IOrMesh triangle

# ─── Install ────────────────────────────────────────────────────────────
install: IOrMesh triangle
	cp $(IORMESH_BIN) FEpre/IOrMesh
	cp $(IORMESH_DIR)/triangle FEpre/
	@echo 'Installed: FEpre/IOrMesh  FEpre/triangle'

# ─── MATLAB / Octave helpers ──────────────────────────────────────────
MATLAB   ?= matlab
OCTAVE   ?= octave
ENGINE    = $(or $(shell which $(OCTAVE) 2>/dev/null),$(shell which $(MATLAB) 2>/dev/null))

ifeq ($(ENGINE),)
define run_m
	@echo 'No MATLAB or Octave found. Set MATLAB= or OCTAVE= to a valid binary.'
endef
else
define run_m
	$(ENGINE) -q --eval "addpath(genpath('.')); $1; exit"
endef
endif

PROJECTS  = ProjectElectrostatics ProjectModalAnalysis \
            ProjectModalAnalysisOpenStrip ProjectThermalDistro \
            ProjectThermalDistroDG ProjectCapacitiveClearance \
            ProjectCapacitiveClearanceCorrelated ProjectWaveGuide \
            ProjectWaveScatteringDD ProjectWaveScatteringDDglob \
            ProjectWaveScatteringEincEs ProjectWaveScatteringFullField \
            ProjectBilateralFilter ProjectBilateralFilterHB \
            ProjectBilateralFilterNL ProjectTwoPostFilter \
            ProjectTwoPostFilterHB ProjectTwoPostFilterNL \
            ProjectCirculator ProjectCirculatorDDschur \
            ProjectCirculatorIMP ProjectCirculatorIMP_TestHarmonicConv \
            ProjectCirculatorIMPDDschur ProjectCirculatorIMPDDschur_Acceleration \
            ProjectCableParassitics ProjectCoaxWideband ProjectNewDDscattering \
            ProjectScatteringFloquet

TESTS     = fem1ode TestCoupling TestCouplingGeneric ThermalFE EntCirc

DD_TESTS  = Accuracy Convergence ConvergenceDD ConvergenceDDref \
            DDfreq FETI Horn Miller WR90 CPWlin CheckSys CheckSysDD \
            CheckSysFE CheckSysGMRES SolversComparison TestBoeing \
            TestEfficient TestHgrad TestInv TestSchur AccuracyDDWEEP \
            AccuracyTFE Bilat PowerHist PlotPatterns diagram

test: $(TESTS:%=test-%)

test-%: Tests/%.m
	$(call run_m, $*)

test-all: test test-dd

test-dd: $(DD_TESTS:%=test-dd-%)

test-dd-%: Tests/DD/%.m
	$(call run_m, cd Tests/DD; $*)

test-nl: test-nl-FerriteTensor

test-nl-%: Tests/NL/%.m
	$(call run_m, cd Tests/NL; $*)

projects: $(PROJECTS:%=project-%)

project-%: %.m
	$(call run_m, $*)

# ─── Run arbitrary MATLAB file ─────────────────────────────────────────
run:
ifndef FILE
	$(error Usage: make run FILE=<name-without-.m>)
endif
	$(call run_m, $(FILE))

# ─── Clean ──────────────────────────────────────────────────────────────
clean:
	rm -f $(IORMESH_BIN) $(IORMESH_DIR)/triangle
	rm -f FEpre/IOrMesh FEpre/triangle
	rm -f $(IORMESH_DIR)/*.o $(IORMESH_DIR)/obj/*.o
	rm -f *.asv *~ *.log *.mat
	rm -f FEass/*.asv FEpre/*.asv FEpost/*.asv
	rm -f Tests/*.asv Tests/DD/*.asv Tests/NL/*.asv

distclean: clean
	rm -f octave-workspace

.PHONY: all install clean distclean IOrMesh triangle \
        test test-all test-dd test-nl projects run
