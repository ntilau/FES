# IOrMesh — 2D Mesh Interface Tool for MATLAB

Finite element 2D mesh generator.  Reads a `.poly` file (Triangle
format), triangulates it, and writes a `.mat` file with mesh data
that MATLAB's `load()` can read directly.

Called by `FEpre/IOrPoly.m` in the mFES project.

## Components

| File | Author | Role |
|---|---|---|
| `main.c` | Laurent Ntibarikure | Mesh driver: invokes Triangle, parses output, builds element-to-edge maps, writes `.mat` |
| `matfiles.c` / `matfiles.h` | Malcolm McLean | Portable ANSI C MATLAB .mat file exporter |
| `triangle.c` / `triangle.h` | Jonathan Richard Shewchuk | Triangle library — quality mesh generator & Delaunay triangulator (v1.6) |

## Build

```sh
make IOrMesh          # build mesh generator only
make triangle         # build Triangle binary only
make all              # build both
make install          # copy binaries to FEpre/
make clean            # remove build artifacts
```

## Usage

```sh
cd FEpre/iormesh-src
./IOrMesh <basename> "<args>"
```

The binary passes the given arguments to Triangle with `-pDeQ`
appended, so common Triangle flags like `-q` (quality), `-a`
(max area), `-A` (region attributes) can be passed through.

## License

- **Triangle** (Jonathan Richard Shewchuk) — permissive license.
  Free for private, research, and institutional use.  Commercial
  distribution requires arrangement with the author.  See
  `LICENSE-Triangle.txt`.

- **matfiles** (Malcolm McLean) — obtained from the MATLAB File
  Exchange.  See `LICENSE-matfiles.txt`.
