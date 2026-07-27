CORE_DIR := cpp
BUILD_DIR := $(CORE_DIR)/build
CMAKE := cmake
CMAKE_BUILD_TYPE ?= Release
FES := $(BUILD_DIR)/fes
MODEL_DIR := data

# Export build dir so 'fes' works directly from shell
export PATH := $(BUILD_DIR):$(PATH)

# All .fes projects (basenames in data/)
FES_PROJECTS := $(patsubst $(MODEL_DIR)/%.fes,%,$(wildcard $(MODEL_DIR)/*.fes))

# ─── Top-level targets ──────────────────────────────────────────────────

.PHONY: all help config build clean help-cpp help-py help-m \
        py-setup py-test m-build m-test m-projects \
        $(FES_PROJECTS)

all: build

help: help-cpp help-py help-m

# ─── C++ backend (cpp/) ─────────────────────────────────────────────

help-cpp:
	@echo "=== C++ backend (cpp/) ==="
	@echo "  make build       Build fes (cmake configure + compile)"
	@echo "  make test        Run all .fes models (load & mesh check)"
	@echo "  make <model>     Run a single model (e.g. WR90)"
	@echo "  make config      Reconfigure cmake"
	@echo "  make clean       Remove build directory"
	@echo "  The fes binary is at $(FES)"
	@echo ""

config:
	$(CMAKE) -S $(CORE_DIR) -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE)

build: config
	$(CMAKE) --build $(BUILD_DIR) --config $(CMAKE_BUILD_TYPE)

clean:
	rm -rf $(BUILD_DIR)

# Each .fes model: load and exit
define fes_target
$(1): build
	@echo "--- $(1) ---"
	cd $(MODEL_DIR) && $(abspath $(FES)) $(1)
endef
$(foreach p,$(FES_PROJECTS),$(eval $(call fes_target,$(p))))

test: build
	@echo "Running all $(words $(FES_PROJECTS)) models..."
	@failed=""; FES_BIN="$(abspath $(FES))"; \
	for p in $(FES_PROJECTS); do \
		echo "--- $$p ---"; \
		cd $(MODEL_DIR) && $$FES_BIN $$p || failed="$$failed $$p"; \
		cd ..; \
	done; \
	if [ -n "$$failed" ]; then \
		echo "FAILED:$$failed"; \
	else \
		echo "All models passed"; \
	fi

# ─── Python backend (py/) ───────────────────────────────────────────────

PY_VENV := py/.venv

help-py:
	@echo "=== Python backend (py/) ==="
	@echo "  make py-setup    Create .venv and install fes package"
	@echo "  make py-test     Run pytest on fes"
	@echo ""

py-setup:
	cd py && ./configure

py-test: $(PY_VENV)
	cd py && $(abspath $(PY_VENV))/bin/python -m pytest tests/ -v

$(PY_VENV):
	cd py && ./configure

# ─── MATLAB backend (m/) ────────────────────────────────────────────────

help-m:
	@echo "=== MATLAB backend (m/) ==="
	@echo "  make m-build     Build IOrMesh and Triangle (mesh tools)"
	@echo "  make m-test      Run all FEM test cases (Project*.m in tests/)"
	@echo "  make m-tests     Run standalone/DD test scripts (tests/)"
	@echo ""

m-build:
	$(MAKE) -C m all

m-test:
	$(MAKE) -C m test

m-tests:
	$(MAKE) -C m tests
