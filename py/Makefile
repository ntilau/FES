# pyFES — IOrMesh tool & mesh data processing
# Builds the C mesher and regenerates .h1.mat files from .poly files.

IORMESH   = iormesh
IORMESH_DIR := $(abspath $(IORMESH))
DATA      = data

# Default Triangle arguments (appended after "-p"):
#   q   quality mesh (minimum angle, e.g. q34)
#   a   area constraint (e.g. a0.01)
#   A   region attributes from .poly region records
#   u   user-defined constraint
# Override per-file:  make data/WaveGuide.h1.mat ARGS="q34a0.01A"
ARGS ?= q34A

CC      = cc
CXX     = c++
CFLAGS  = -O2 -Wall -Wno-deprecated-declarations
CXXFLAGS = -O2 -Wall -Wno-deprecated-declarations -Wno-vla-cxx-extension
LDLIBS  = -lm

.PHONY: all build clean info list

# ---- Build the IOrMesh binary ------------------------------------
build: $(IORMESH)/IOrMesh

$(IORMESH)/IOrMesh: $(IORMESH)/main.cpp $(IORMESH)/matfiles.c $(IORMESH)/matfiles.h $(IORMESH)/triangle
	$(CC) $(CFLAGS) -c -o $(IORMESH)/matfiles.o $(IORMESH)/matfiles.c
	$(CXX) $(CXXFLAGS) -c -o $(IORMESH)/main.o $(IORMESH)/main.cpp
	$(CXX) $(CXXFLAGS) -o $@ $(IORMESH)/main.o $(IORMESH)/matfiles.o $(LDLIBS)

# ---- Process a single .poly file ----------------------------------
# Usage: make data/Foo.h1.mat                 (uses default ARGS)
#        make data/Foo.h1.mat ARGS="q33a1A"   (override switches)
$(DATA)/%.h1.mat: $(DATA)/%.poly $(IORMESH)/IOrMesh
	PATH="$(IORMESH_DIR):$$PATH" "$(IORMESH_DIR)/IOrMesh" "$(@:.h1.mat=)" "$(ARGS)"

# ---- Process all .poly files missing their .h1.mat -----------------
all: build
	@count=0; \
	for poly in $(DATA)/*.poly; do \
		name=$$(basename "$$poly" .poly); \
		mat="$(DATA)/$$name.h1.mat"; \
		if [ ! -f "$$mat" ]; then \
			echo ">>> $$name  (ARGS=\"$(ARGS)\")"; \
			PATH="$(IORMESH_DIR):$$PATH" "$(IORMESH_DIR)/IOrMesh" "$(DATA)/$$name" "$(ARGS)"; \
			count=$$((count + 1)); \
		fi; \
	done; \
	if [ $$count -eq 0 ]; then \
		echo "All .poly files already have matching .h1.mat files."; \
	else \
		echo "Processed $$count file(s)."; \
	fi

# ---- Rebuild all .h1.mat files (even existing ones) ----------------
rebuild: build
	@count=0; \
	for poly in $(DATA)/*.poly; do \
		name=$$(basename "$$poly" .poly); \
		echo ">>> $$name  (ARGS=\"$(ARGS)\")"; \
		PATH="$(IORMESH_DIR):$$PATH" "$(IORMESH_DIR)/IOrMesh" "$(DATA)/$$name" "$(ARGS)"; \
		count=$$((count + 1)); \
	done; \
	echo "Rebuilt $$count file(s)."

# ---- Utilities -----------------------------------------------------
list:
	@echo ".poly files in $(DATA)/:"; \
	for poly in $(DATA)/*.poly; do \
		name=$$(basename "$$poly" .poly); \
		mat="$(DATA)/$$name.h1.mat"; \
		if [ -f "$$mat" ]; then echo "  [OK]  $$name.h1.mat"; \
		else echo "  [MISS] $$name (run: make data/$$name.h1.mat)"; fi; \
	done

info:
	@echo "IOrMesh: triangle-based mesher -> MATLAB .h1.mat files"
	@echo "  Source:  $(IORMESH)/"
	@echo "  Data:    $(DATA)/ (*.poly  ->  *.h1.mat)"
	@echo "  Args:    ARGS=$(ARGS)"
	@echo ""
	@echo "Commands:"
	@echo "  make build          — compile IOrMesh"
	@echo "  make data/X.h1.mat  — mesh a single .poly file"
	@echo "  make all            — mesh all .poly files missing .h1.mat"
	@echo "  make rebuild        — re-mesh all .poly files"
	@echo "  make list           — show which .poly files are done"
	@echo "  make clean          — remove build artifacts"

clean:
	rm -f $(IORMESH)/IOrMesh $(IORMESH)/*.o
