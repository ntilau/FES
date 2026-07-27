# mFES - MATLAB Finite Element Solver
# =====================================
# IOrMesh : 2D mesh interface tool for MATLAB (Triangle wrapper)
#   (c) Laurent Ntibarikure
#
# Directory layout (mirrors py/):
#   fes/       — package root
#     core/      — assembly routines
#     mesh/      — mesh I/O, geometry writers, IGES/, iormesh-src/
#     post/      — VTK field export
#     projects/  — project drivers (FEM test cases)
#   tests/     — standalone / debug / DD / NL test scripts
#
# Targets:
#   make IOrMesh     Build mesh generator (C++/C)
#   make triangle    Build Triangle library binary (called by IOrMesh)
#   make all         Build both
#   make test        Run all project-based FEM test cases
#   make tests       Run standalone test scripts
#   make clean       Remove build artifacts

CXX      ?= g++
CC       ?= gcc
CFLAGS   ?= -Wall -O3 -ffast-math -march=native -Wno-deprecated
LDFLAGS  ?=

IORMESH_DIR = fes/mesh/iormesh-src
IORMESH_BIN = fes/mesh/iormesh-src/IOrMesh

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
	cp $(IORMESH_BIN) fes/mesh/IOrMesh
	cp $(IORMESH_DIR)/triangle fes/mesh/
	@echo 'Installed: fes/mesh/IOrMesh  fes/mesh/triangle'

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

TESTS     = ProjectElectrostatics ProjectModalAnalysis \
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

EXTRAS    = fem1ode TestCoupling TestCouplingGeneric ThermalFE EntCirc

DD_EXTRAS = Accuracy Convergence ConvergenceDD ConvergenceDDref \
            DDfreq FETI Horn Miller WR90 CPWlin CheckSys CheckSysDD \
            CheckSysFE CheckSysGMRES SolversComparison TestBoeing \
            TestEfficient TestHgrad TestInv TestSchur AccuracyDDWEEP \
            AccuracyTFE Bilat PowerHist PlotPatterns diagram

test: $(TESTS:%=test-%)

test-%: fes/projects/%.m
	$(call run_m, $*)

tests: $(EXTRAS:%=tests-%)

tests-%: tests/%.m
	$(call run_m, $*)

tests-dd: $(DD_EXTRAS:%=tests-dd-%)

tests-dd-%: tests/DD/%.m
	$(call run_m, cd tests/DD; $*)

tests-nl:

tests-nl-%: tests/NL/%.m
	$(call run_m, cd tests/NL; $*)

# ─── Run arbitrary MATLAB file ─────────────────────────────────────────
run:
ifndef FILE
	$(error Usage: make run FILE=<name-without-.m>)
endif
	$(call run_m, $(FILE))

# ─── Clean ──────────────────────────────────────────────────────────────
clean:
	rm -f $(IORMESH_BIN) $(IORMESH_DIR)/triangle
	rm -f fes/mesh/IOrMesh fes/mesh/triangle
	rm -f $(IORMESH_DIR)/*.o $(IORMESH_DIR)/obj/*.o
	rm -f *.asv *~ *.log *.mat
	rm -f fes/core/*.asv fes/mesh/*.asv fes/post/*.asv
	rm -f fes/projects/*.asv tests/*.asv tests/DD/*.asv tests/NL/*.asv

distclean: clean
	rm -f octave-workspace

.PHONY: all install clean distclean IOrMesh triangle \
        test tests tests-dd tests-nl run
