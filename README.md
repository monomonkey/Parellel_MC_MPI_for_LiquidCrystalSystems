# MC_NVT_MPI: Monte Carlo Gay-Berne Simulation with MPI

This repository contains a C implementation of a Monte Carlo simulation for Gay-Berne particles in an NVT (or NPT) ensemble, parallelized using MPI with a domain decomposition strategy. It supports the inclusion of a spherical colloidal particle interacting with the mesogens.

## Model Description

The simulation models a system of ellipsoidal particles interacting via the **Gay-Berne (GB) potential**, a standard anisotropic potential for liquid crystals. 

### Key Features
*   **Ensemble**: NVT (Canonical) or NPT (Isobaric-Isothermal), controlled by `mc_ensemble_type`.
*   **Potential**: Gay-Berne potential handling position-dependent and orientation-dependent interactions.
*   **Colloid**: Option to include a large spherical colloidal particle (`colloid_wall_option`), simulating its interaction with the GB solvent (`GBCollRod`).
*   **Boundary Conditions**: Periodic boundary conditions (PBC) are applied to the simulation box.

## Domain Decomposition Strategy

To handle large systems efficiently, the simulation box is spatially decomposed using MPI:

1.  **Block Decomposition**: The simulation domain is divided into a 3D grid of **Blocks**. Each MPI rank is responsible for one Block.
2.  **Cell Subdivision**: Each Block is further subdivided into **8 local subcells** (2x2x2 grid).
3.  **Ghost Cells & Extended Grid**:
    *   To calculate interactions across rank boundaries, each Block maintains an **Extended Grid** of 3x3x3 cells (27 total).
    *   The center (1,1,1) is the local active subcell.
    *   The surrounding cells are either local neighbors or **Ghost Cells** received from neighboring MPI ranks.
4.  **Active Cell Loop**: The Monte Carlo sweep iterates through the 8 local subcells. For each active subcell, a 3x3x3 neighborhood is constructed by exchanging data with relevant neighbors using non-blocking MPI communication (`MPI_Isend`, `MPI_Irecv`).

## Code Structure

*   **`MCnptColloid.c`**: The single source file containing the entire simulation logic, MPI handling, and physics definitions.
*   **`parametersFile.txt`**: Configuration file defining physical constants, system size, and simulation parameters.

### Key Functions
*   **`main`**: Orchestrates the simulation, MPI initialization, and the main Monte Carlo loop over subcells.
*   **`build_neighbor_cells`**: Constructs the 3x3x3 grid for the active subcell, managing the complex symmetric exchange of ghost cells with neighbors.
*   **`copy_cell_content`**: Helper to deep-copy cell data (particles and metadata) from local storage to the working grid.
*   **`clean_neighbor_cells`**: Frees the temporary memory used by the extended grid after processing a subcell.
*   **`pack_cell` / `unpack_cell`**: Serializes/deserializes `Cell` structures for MPI transmission.
*   **`attempt_monte_carlo_move`**: (Placeholder/In-progress) Handles the Metropolis-Hastings logic for particle moves using the constructed neighbor grid.

## Compilation

The code requires an MPI implementation (e.g., MPICH, OpenMPI) and a C compiler.

```bash
mpicc MCnptColloid.c -o MCnptColloid -lm
```

## Usage

1.  Ensure `parametersFile.txt` is present in the execution directory and properly configured.
2.  Run the simulation using `mpirun` or `mpiexec`. The number of processes should match the desired decomposition.

```bash
mpirun -np <Number_of_Processes> ./MCnptColloid
```

## Parameters

The `parametersFile.txt` should contain keys for system setup, such as:
*   `num_particles`: Total number of particles.
*   `system_temperature`, `system_pressure`, `system_density`.
*   `gb_*`: Gay-Berne potential parameters (`kappa`, `epsilon0`, `mu_exponent`, etc.).
*   `total_mc_cycles`: Simulation duration.
