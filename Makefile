CORE_DIR := core
BUILD_DIR := $(CORE_DIR)/build
CMAKE := cmake
CMAKE_BUILD_TYPE ?= Release
FES := $(BUILD_DIR)/fes
MODEL_DIR := mdl

# Export build dir so 'fes' works directly from shell
export PATH := $(BUILD_DIR):$(PATH)

# All .fes projects (basenames in mdl/)
FES_PROJECTS := $(patsubst $(MODEL_DIR)/%.fes,%,$(wildcard $(MODEL_DIR)/*.fes))

.PHONY: all help config build clean $(FES_PROJECTS)

all: build

help:
	@echo "Usage: make [target]"
	@echo "Targets:"
	@echo "  build       Build fes"
	@echo "  clean       Remove build directory"
	@echo "  test        Run all .fes models (verify load, mesh only)"
	@echo ""
	@echo "The fes binary is at $(FES)"
	@echo "  export PATH=\$$(pwd)/$(BUILD_DIR):\$$PATH"
	@echo "  fes <model> <args>"

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
