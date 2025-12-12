#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include <mpi.h>

#define tercio  1.0/3.0
#define pi      3.1415926535898
#define racu3   sqrt(3.0)
#define dosqui	2.0/15.0
#define POINTS_ON_SPHERE 1000000           // Total number of random points on sphere
#define I_MAX_RAND       (1.0f/RAND_MAX)   // Inverse of the maximum random value a

/**********************************************************************************************/
/**********************************************************************************************/

typedef struct {
    // Particle metadata
    int global_index;   // Particle index in the whole system
    int local_index;    // Particle index inside its block

    int cell_owner;     // Cell owner index of this particle
    int block_owner;    // Block owner index of this particle
    
    double position[3]; // Particle position (x, y, z)
    double orient[3];   // Particle orientation (x, y, z)

    int num_neighbs;    // number of neighbours
    int max_neighbs;    // maximum number of neighbours per particle

    int* neighbs_index; // neighbours indexes

} Particle;


typedef struct {

    // Subcell metadata
    int index;                 // Local index within MPI block (0-7 for A-H)
    int block_owner;           // Rank which this cell belongs

    int capacity;              // Maximum number of particles per cell
    int num_particles;         // Current number of particles

    double size[3];            // Size of the cell
    int    coords[3];          // Local coordinates in the block
    double domain_min[3];      // Minimum coordinates of this block
    double domain_max[3];      // Maximum coordinates of this block

    int is_reset;              // Is this the currently active cell?

    // Particle data for THIS cell
    int* particles_index;      // Particle indexes of the particles in the cell
    Particle* particles;       // Dynamic array for particles in the cell

} Cell;

typedef struct {

    int rank;

    // MPI block information
    int capacity;               // Maximum number of particles per block
    int num_particles;          // Current number of particles

    int    coords[3];           // Coordinates in the MPI grid
    double size[3];             // Size of the block
    double domain_min[3];       // Minimum coordinates of this block
    double domain_max[3];       // Maximum coordinates of this block
    
    // Cells information
    int    cells_index[8];      // Index of each cell, it indicates the order to perform each sweep
    double cell_size[3];        // Size of each cell in x,y,z

    int are_cells_reset;        // Flag to indicate if cells are correctly allocated and initialized

    // int* registery[3];          // Index to hold all particle global index with its block, cell and local index
    
    Particle* batch;            // Batch of particles copied from/to master_rank

    // The 2×2×2 local cells (0, 1, ..., 7)
    // Indexed by [x][y][z] where each is 0 or 1
    Cell local_cells[2][2][2];  
    
    // The 3×3×3 structure including ghost neighbors
    // Indexed by [x][y][z] where each is -1, 0 or 1
    Cell extended_grid[3][3][3];

} Block;


/**********************************************************************************************/
/**********************************************************************************************/
int     num_particles;                        // Number of total particles
int     max_neighbs;                          // Maximum number of neighbors per particle
int     cell_capacity;                        // Maximum number of particles a cell can hold
int     block_capacity;                       // Maximum number of particles a block can hold
int     total_mc_cycles;                      // Number of MonteCarlo cycles
int     configuration_save_interval;          // How often the system parameters are saved in a file
int     initial_configuration_type;           // Initial configuration type (random, arranged, from a file)
int     statistics_print_interval;            // How often the system mean values are displayed
int     colloid_wall_option;                  // Boolean value to take into account a spherical colloid (0=False, 1=True)
int     mc_ensemble_type;                     // Type of Thermodynamic ensemble (1=NVT, 2=NPT)
long    total_accepted_moves = 0;             // Counter for the total accepted moves
long    total_attempted_moves = 0;            // Counter for the total attempted moves
float   max_displacement;                     // Maximum displacement allowed in (x, y, z)
float   max_rotation;                         // Maximum rotation allowed
float   rot_vec_x[POINTS_ON_SPHERE];          // Random points on the unit sphere (x coord)
float   rot_vec_y[POINTS_ON_SPHERE];          // Random points on the unit sphere (y coord)
float   rot_vec_z[POINTS_ON_SPHERE];          // Random points on the unit sphere (z coord)
double  box_volume;                           // Volume of the simulation box
double  system_pressure;                      // System pressure
double  system_temperature;                   // System temperature
double  system_density;                       // System density
double  gb_epsilon0;                          // Gay-Berne epsilon value
double  gb_4eps0;                             // 4 times the Gay-Berne epsilon
double  gb_kappa;                             // Aspect ratio κ = (σ_ee/σ_ss)
double  gb_kappa_prime;                       // Gay-Berne kappa prime value
double  gb_mu_exponent;                       // Gay-Berne mu exponent value
double  gb_nu_exponent;                       // Gay-Berne nu exponent value
double  gb_sigma_endtoend;                    // Mesogen length (σ_ee, end-to-end)
double  gb_sigma_sidetoside;                  // Mesogen width (σ_ss, side-to-side)
double  cutoff_radius;                        // Cutoff radius
double  verlet_radius;                        // Verlet cutoff radius
double  gb_inercia;                           // Moment of inertia value
double  gb_beta;                              // Thermodynamic Beta  β = (kB * T)⁻¹
double  cutoff_radius_sq;                     // Cutoff radius squared
double  r_verlet_sq;                          // Verlet cutoff radius squared
double  gb_kappa_sq;                          // Squared aspect ratio κ² = (σ_ee/σ_ss)²
double  gb_chi_sq;                            // Squared shape anisotropy χ²
double  verlet_volume;                        // Volume of a sphere using verlet cutoff radius
double  msad;                                 // Mean squared angular displacement parameter
double  gb_chi;                               // Shape anisotropy χ
double  gb_k_imu;                             // κ' raised to 1/μ power
double  gb_chi_prime;                         // Chi prime (χ')
double  colloid_radius;                       // Radius of the central spherical colloid
double  colloid_radius_sq;                    // Squared radius of the central spherical colloid
double  colloid_volume;                       // Volume of the colloid particle
double  colloid_interaction_amplitude;        // Colloid interaction amplitude
double  colloid_interaction_epsilon;          // Colloid interaction epsilon value
double  box_size[3];          		          // Simulation box dimensions
double  colloid_coords[3];                    // Colloid center coordinates (x, y, z)
double  colloid_rod_cutoff;                   // Cutoff distance for colloid-rod interactions

MPI_Datatype Particle_MPI;
MPI_Datatype Cell_MPI;

/**********************************************************************************************/
/**********************************************************************************************/

void    read_simulation_parameters(void);
int     read_next_non_comment_line(FILE *file, char *buffer, size_t buffer_size);

void    print_resum(void);

void    print_intermedio(int current_cycle , float n_ave, float ave_ene, float system_energy, float ave_ordpar, float global_order_parameter, float ave_dens, int accepted_moves, int attempted_moves, float dir_x, float dir_y, float dir_z);

void    print_resumfinal(float ave_ene, float ave_dens, float ave_PressTns[4], float ave_ordpar,
                         float system_energy, float dir_x, float dir_y, float dir_z, clock_t start, clock_t end);
  
void    print_Qmga(double *position_x, double *position_y, double *position_z, double *orient_x, double *orient_y, double *orient_z,
		   int option);

void    EnergyCheck(double system_energy, double *position_x, double *position_y, double *position_z, double *orient_x, double *orient_y, 
		        double *orient_z, double *rxi, double *ryi, double *rzi, int *neighbor_count_array, int *neighbor_list_array, 
		        int max_neig);

void    check_no_colloid_overlaps(double *position_x, double *position_y, double *position_z);

static inline double  compute_distance_sq(double x1, double y1, double z1, double x2, double y2, double z2, double *dx, double *dy, double *dz);

void    get_system_const(void);

void    estima_vecinos(int *max_neighbs, int *otra_estimacion, double *diff_SqrRadi, 
		int *NumPart_inByt, int *neig_inByt, float *factor_lista);
	
void    initialize_random_positions(double *position_x, double *position_y, double *position_z, double *orient_x, double *orient_y, double *orient_z);

void    initialize_cubic_lattice_positions(double *position_x, double *position_y, double *position_z, double *orient_x, double *orient_y, double *orient_z);

void    read_initial_configuration(double *position_x, double *position_y, double *position_z, double *orient_x, double *orient_y, double *orient_z);

void    print_pos_xyz(double *pos_x, double *pos_y, double *pos_z, int option);

void    print_Trayectoria(double *position_x, double *position_y, double *position_z, double *orient_x, double *orient_y, double *orient_z, int option);

void    free_3_arrays(double **v1, double **v2, double **v3);

void    list_vecRed(double *position_x, double *position_y, double *position_z, double *rxi, double *ryi, double *rzi, int *neighbor_count_array, int *neighbor_list_array);

void    build_neighbor_lists(int nvest, double *position_x, double *position_y, double *position_z, double *rxo, double *ryo, double *rzo, int *neighbor_count_array, int *neighbor_list_array);

double  max_dsp(double *position_x, double *position_y, double *position_z, double *rxo, double *ryo, double *rzo);

static inline double g_omega(double eiej, double amasb2, double amenb2, double omega);

double  pij_GB(double ei[3], double ej[3], double rij, double irij, double iei, double iej, double eiej, double dr[3]);

double  GBCollRod(int i, double ri[3], double ei[3], double Coll_pos[3]);

double  GBij(int particle_index, double *position_x, double *position_y, double *position_z, double *orient_x, double *orient_y, double *orient_z, double fij[3]);

double  GBij_PosMod(int particle_index, double ri[3], double ei[3], double *position_x, double *position_y, double *position_z, double *orient_x, double *orient_y, double *orient_z); 

double  GBij_lstnb(int i, double *position_x, double *position_y, double *position_z, double *orient_x, double *orient_y, double *orient_z, int *neighbor_count_array, int *neighbor_list_array);

void    apply_pbc(double *x, double *y, double *z);

void    attempt_monte_carlo_move(double *position_x, double *position_y, double *position_z, double *orient_x, double *orient_y, double *orient_z, float max_displacement, double *system_energy, 
		     int *neighbor_count_array, int *neighbor_list_array, long *acc, double *max_displacement_squared, size_t *histograma);

double  GBij_PosMod_lstnb(int i, double ri[3], double ei[3], double *position_x, double *position_y, double *position_z, double *orient_x, double *orient_y, double *orient_z, int *neighbor_count_array, int *neighbor_list_array);

double  tot_ene_lstnb(double *position_x, double *position_y, double *position_z, double *orient_x, double *orient_y, double *orient_z, int *neighbor_count_array, int *neighbor_list_array);

static inline int find_largest_eigenvalue_index(float a[3]);

void    diagonalize_symmetric_matrix(float input_matrix[3][3], 
                                     float eigenvectors[3][3], 
                                     float eigenvalues[3]);

void    compute_liquid_crystal_order_parameters(int num_particles, 
                                                double *orientation_x, 
                                                double *orientation_y, 
                                                double *orientation_z, 
                                                float *director_x, 
                                                float *director_y, 
                                                float *director_z, 
                                                float *order_parameter);

void    update_running_averages(int opcion, int nump, float *n_ave, float *ave_ene, float *ave_dens, float ave_pressTns[4], 
	                            float *ave_ordpar, float energia, float dens, float press_tns[4], 
		                        float ord_par);

void    jacobi(int np, float a[np][np], float d[np], float v[np][np]);
	
double  tot_ene(double *position_x, double *position_y, double *position_z, double *orient_x, double *orient_y, double *orient_z);

void    esc_fort1(double *position_x, double *position_y, double *position_z, double *orient_x, double *orient_y, double *orient_z);

void    backup_restart_file(double *position_x, double *position_y, double *position_z, double *orient_x, double *orient_y, double *orient_z);

static inline void   pto_ransphr(float *x_m, float *y_m, float *z_m, float frac);

void    Marsaglia4MC(float *arr_x, float *arr_y, float *arr_z, float radio);

static inline double compute_dot_product(double x1, double y1, double z1, double x2, double y2, double z2);

/***********************************************************************************************************/
/***********************************************************************************************************/
/***********************************************************************************************************/
/***********************************************************************************************************/

void get_3D_rank_grid_dims(int* n_ranks, int* grid_dims);

void get_block_size(double* block_size, double* box_size, int* grid_dims);

void get_block_coords(int block_coords[][3], int* grid_dims);

void block_init(Block* block, int rank, int capacity, int* block_coords, double* block_size);

void cell_init(Cell *cell, int block_owner, int capacity, int* cell_coords, double* cell_size, double block_domain_min[3]);

void particle_init(Particle* particle, int index, double position[3], double orient[3], int max_neighbs);

void particle_plain_init(Particle* particle);

/* MATH ****************************************************************************************/
int** allocate_int_matrix(int rows, int cols);

double** allocate_dbl_matrix(int rows, int cols);

Particle** allocate_particle_matrix(int rows, int cols);

void free_int_matrix(int** matrix, int rows);

void free_dbl_matrix(double** matrix, int rows);

void free_particle_matrix(Particle** matrix, int rows, int cols);

void free_particle_array(Particle* array, int rows);

int  cell_coords_to_index(int x, int y, int z);

void index_to_cell_coords(int index, int* x, int* y, int* z);

void shuffle_int_array(int* array, int n);

void swap(int* a, int* b);

int  get_block_index(int block_coords[3], int grid_dims[3]);

static inline int mod(int a, int b);



void error_checker(int result, int expected, const char* paramName);
int read_next_parameter(FILE *file, char *paramName, char *value);
void load_initial_conditions(const char* filename);

void broadcast_global_vars(int master_rank);
void broadcast_block_params(int master_rank, double block_size[3], int block_coords[][3], int n_ranks);
void broadcast_cell_params(int master_rank, int cell_capacity);

void get_initial_config(int initial_configuration_type,
                        double *position_x, double *position_y, double *position_z, 
                        double *orient_x, double *orient_y, double *orient_z);

void callocate_3_dbl_arrays(double **arr1, double **arr2, double **arr3, int length);
void callocate_3_float_arrays(float **arr1, float **arr2, float **arr3, int length);
int get_block_capacity(double block_size[3]);
int get_max_num_neighb(double verlet_volume);
int get_cell_capacity(int block_capacity);

void initialize_global_vars(const char* parameters_file,
                           double **position_x,  double **position_y, double **position_z,
                            double **orient_x,    double **orient_y,  double **orient_z,
                            double **rxi, double **ryi, double **rzi,
                            float *rot_vec_x, float *rot_vec_y, float *rot_vec_z,
                            float unitary_radius);


void get_particle_owners(double position[3], int* block_owner, int* cell_index, double block_size[3], int grid_dims[3]);
int get_owner_block_index(double position[3], double block_size[3], int grid_dims[3], int block_coords[3]);
int get_owner_cell_index(double position[3], double block_size[3], int block_coords[3], int cell_coords[3]);

MPI_Datatype create_particle_mpi_type(void);
MPI_Datatype create_cell_mpi_type(MPI_Datatype Particle_MPI);
void mpi_datatypes_init(MPI_Datatype* Particle_MPI, MPI_Datatype* Cell_MPI);

void send_particle(const Particle *particle, int dest, int tag, MPI_Comm comm);
void recv_particle(Particle *particle, int source, int tag, MPI_Comm comm);

char* pack_particle(const Particle* p, int* out_bytes);
int   unpack_particle(Particle* p, const char* buf);
void  copy_particle(Particle* dst, Particle* src);

void distr_particles_to_batches(Particle* box_particles, 
                                   Particle** batch_matrix, 
                                   int* particles_per_batch, 
                                   double* block_size, int* grid_dims, int n_ranks);

char* pack_particle_array(Particle* particle_array, int n_particles, int* total_bytes_out);
int   unpack_particle_array(Particle* particle_array, int n_particles, char* buffer);

char* pack_cell(const Cell* c, int* out_bytes);
int   unpack_cell(Cell* c, const char* buf);

void block_free(Block* block);
void cell_free(Cell *cell);
void particle_free(Particle* particle);

void dist_batch_to_cells(Block* block);
void reset_block_cells(Block* block);
void reset_cell(Cell* cell);

void copy_cell(Cell* dest, const Cell* src);
void build_neighbor_cells(Block* block, int active_cell_index, int grid_dims[3]);
void clean_neighbor_cells(Block* block);

int get_cell_owner_full(int cell_rel_coords[3], 
                           int current_block_coords[3], 
                           int owner_block_coords[3],
                           int block_offset[3] 
                           int grid_dims[3], 
                           int local_grid_dim);

int get_cell_owner(int cell_coords[3], 
                   int current_block_coords[3], 
                   int grid_dims[3], 
                   int local_grid_dim);

/**********************************************************************************************/
/*****************          Programa principal Monte Carlo NVT (Gay-Berne)        *************/
/**********************************************************************************************/

int main(int argc, char** argv) {

    // MPI variables
    int rank, n_ranks;

    // Set up the MPI environment
    MPI_Init(&argc, &argv);

    // MPI variables
    int master_rank = 0;
    int grid_dims[3] = {0, 0, 0};

    // Get the number of ranks and the 3D rank-grid dimensions
    get_3D_rank_grid_dims(&n_ranks, grid_dims);
    // Get my rank
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    char parameters_file[100] = "parametersFile.txt";

    // Block variables
    int    block_coords[n_ranks][3];      // Cartesian coordinates of each block (e.g. (1, 1, 0))
    double block_size[3];                 // Length of each cubic-like box (Lx × Ly × Lz)

    // ##################################################################################
    // ##################################################################################

    // Simulation control variables
	int current_cycle ;                   // Current MC cycle counter
    int i_Nbyt;                           // Memory size for neighbor neighbor_list_array indexing array (in bytes)
    int i_nbbyt;                          // Memory size for neighbor neighbor_list_array data array (in bytes)  
    int oesti;                            // Another neighbor estimation parameter
	int *neighbor_count_array;            // Neighbor neighbor_list_array: neighbor_count_array[i] = end index for particle i's neighbors in 'neighbor_list_array'
    int *neighbor_list_array;             // Neighbor neighbor_list_array: contains actual neighbor indices for all particles
    int neighbor_list_update_needed;      // Flag indicating if neighbor neighbor_list_array needs updating (1=update, 0=no update)
    int check_energy = 0;                 // Debug flag to verify system_energy calculations (0=off, 1=on)
	
    // Statistics
	long  accepted_moves = 0;             // Counter for accepted MC moves in current block
    long  attempted_moves = 0;            // Counter for total attempted MC moves in current block
	float acceptance_ratio ;              // Acceptance percentage = (accept/attempted_moves)*100
    float neighbors_update_factor = 1.0;  // Factor for neighbor neighbor_list_array update criterion

    // Particle Data Arrays
	double *position_x, *position_y, *position_z;   // Current particle positions (x, y, z coordinates)
    double *rxi, *ryi, *rzi;                        // Reference positions for neighbor neighbor_list_array updates
	double *orient_x, *orient_y, *orient_z;         // Particle orientation vectors (unit vectors)

    double** all_positions;                         // All particle positions (x, y, z) coordinates
    double** all_orients;                           // All particle orientations in (x, y, z) direction vectors
    
    // Energy and Move Tracking
    double system_energy = 0.0;              // Total system system_energy (Gay-Berne + colloid interactions)
    double max_displacement_squared = 0.0;   // Maximum displacement squared since last neighbor neighbor_list_array update
    double squared_skin_depth;               // Threshold squared distance for neighbor neighbor_list_array updates
    
    // ##################################################################################
    // ##################################################################################

    Particle* box_particles = NULL;

    // Start random seed
	srand(time(0));
    
    //Master rank read/loads the initial parameters and fill position and orientation arrays
    if (rank == master_rank) {

        // Read/load initial parameters and fill the primary arrays like position, orient, rx and rot_vec
        initialize_global_vars(parameters_file,
                               &position_x,  &position_y, &position_z,
                               &orient_x, &orient_y, &orient_z,
                               &rxi, &ryi, &rzi,
                               rot_vec_x, rot_vec_y, rot_vec_z,
                               1.0);

        // For now let's duplicate the position and orientation information into a single matrix
        all_positions = allocate_dbl_matrix(num_particles, 3);
        all_orients   = allocate_dbl_matrix(num_particles, 3);

        for (int i = 0; i < num_particles; i++) {
            
            all_positions[i][0] = position_x[i];
            all_positions[i][1] = position_y[i];
            all_positions[i][2] = position_z[i];

            all_orients[i][0]   = orient_x[i];
            all_orients[i][1]   = orient_y[i];
            all_orients[i][2]   = orient_z[i];
        }
        
        // Calculate constants using the initial params
	    get_system_const();

        max_neighbs = get_max_num_neighb(verlet_volume);

        // Create an array of num_particles particles struct
        box_particles = (Particle *) malloc(num_particles * sizeof(Particle));
        
        for (int i = 0; i < num_particles; i++) {
            particle_init(&box_particles[i], i, all_positions[i], all_orients[i], max_neighbs);
        }

        printf("================================================\n"
               "Initializing domain (block) decomposition\n\n"
               "Total cores used         : %d\n"
               "MPI choose decomposition : %d × %d × %d\n", n_ranks,
                                                            grid_dims[0], 
                                                            grid_dims[1], 
                                                            grid_dims[2]);

        get_block_size(block_size, box_size, grid_dims);
        block_capacity = get_block_capacity(block_size);
        cell_capacity  = get_cell_capacity(block_capacity);

        printf("Block_size               : %.2lf × %.2lf × %.2lf\n", block_size[0], 
                                                                     block_size[1], 
                                                                     block_size[2]);

        get_block_coords(block_coords, grid_dims);
        
        printf("================================================\n");

        fflush(stdout);

        free_dbl_matrix(all_positions, num_particles); all_positions = NULL;
        free_dbl_matrix(all_orients, num_particles);   all_orients   = NULL;
    }

    MPI_Barrier(MPI_COMM_WORLD); // ######################################################

    // Rank 0 broadcast the initial parameters read
    broadcast_global_vars(master_rank);
    
    // At this point, all ranks have initialized its global variables
    // so we're ready to initialize our blocks
    broadcast_block_params(master_rank, block_size, block_coords, n_ranks);
    broadcast_cell_params(master_rank, cell_capacity);

    // Each rank builds their own particle's batch to exchange particles between master rank and other ranks
    // Also recieves the cell indexes to which each particle belongs and the total number of particles exchanged.

    Block block;
    block_init(&block, rank, block_capacity, block_coords[rank], block_size);

    if (rank == master_rank) {

        // Build particle blocks for each rank
        Particle** particle_batches = allocate_particle_matrix(n_ranks, block_capacity);
        int particles_per_batch[n_ranks];

        distr_particles_to_batches(box_particles, particle_batches, particles_per_batch, 
                                   block_size, grid_dims, n_ranks);

        int   batch_bytes;
        char* batch_buffer;

        block.num_particles = particles_per_batch[0];

        batch_buffer = pack_particle_array(particle_batches[0], particles_per_batch[0], &batch_bytes);
        unpack_particle_array(block.batch, particles_per_batch[0], batch_buffer);

        free(batch_buffer);

        // Distribute the rest of particles to their respective ranks
        // Also distribute the particle_cell_indexes and the particles_per_batch
        // Start in 1 since 0 is the master_rank
        for (int i=1; i < n_ranks; i++) {

            int   batch_bytes;
            char* batch_buffer;

            MPI_Send(&particles_per_batch[i], 1, MPI_INT, i, i, MPI_COMM_WORLD);

            batch_buffer = pack_particle_array(particle_batches[i], particles_per_batch[i], &batch_bytes);
            MPI_Send(batch_buffer, batch_bytes, MPI_BYTE, i, i+n_ranks, MPI_COMM_WORLD);
            
            free(batch_buffer);
        }


        free_particle_matrix(particle_batches, n_ranks, block_capacity);

        // printf("\n"
        //         "I am rank %d \n"
        //         "My 1st particle glob_index  %d\n"
        //         "My 1st particle max_neighs  %d\n"
        //         "My 1st particle coords are  (%.2lf,%.2lf,%.2lf)\n"
        //         "My 1st particle orient are  (%.2lf,%.2lf,%.2lf)\n",
        //         rank,
        //         block.batch[0].global_index, 
        //         block.batch[0].max_neighbs,
        //         block.batch[0].position[0],
        //         block.batch[0].position[1],
        //         block.batch[0].position[2],
        //         block.batch[0].orient[0],
        //         block.batch[0].orient[1],
        //         block.batch[0].orient[2]);


    } 
    else {

        // RECEIVE DATA FROM MASTER RANK
        MPI_Status status;
        int batch_bytes;

        // Recv number of particles per block
        MPI_Recv(&block.num_particles, 1, MPI_INT, master_rank, rank, MPI_COMM_WORLD, &status);

        // Recv particle data  ------------------------------------------------------------------┐
        // Probe packed buffer
        MPI_Probe(master_rank, rank+n_ranks, MPI_COMM_WORLD, &status);

        MPI_Get_count(&status, MPI_BYTE, &batch_bytes);
        
        // Receive buffer
        char* batch_buffer = malloc(batch_bytes);
        MPI_Recv(batch_buffer, 
                 batch_bytes, 
                 MPI_BYTE, 
                 master_rank, 
                 rank+n_ranks, 
                 MPI_COMM_WORLD, 
                 MPI_STATUS_IGNORE);
        
        // Unpack particle
        unpack_particle_array(block.batch, block.num_particles, batch_buffer);
        
        free(batch_buffer);  //------------------------------------------------------------------┘
    }

    dist_batch_to_cells(&block);

    printf("\n"
            "I am rank %d \n"
            "My block num_particle is     %d\n"
            "Cell [0][0][0] has           %d\n"
            "Cell [1][0][0] has           %d\n"
            "Cell [0][1][0] has           %d\n"
            "Cell [1][1][0] has           %d\n"
            "Cell [0][0][1] has           %d\n"
            "Cell [0][1][1] has           %d\n"
            "Cell [0][1][1] has           %d\n"
            "Cell [1][1][1] has           %d\n"
            "TOTAL =                      %d\n",
            rank,
            block.num_particles,
            block.local_cells[0][0][0].num_particles,
            block.local_cells[1][0][0].num_particles,
            block.local_cells[0][1][0].num_particles,
            block.local_cells[1][1][0].num_particles,
            block.local_cells[0][0][1].num_particles,
            block.local_cells[1][0][1].num_particles,
            block.local_cells[0][1][1].num_particles,
            block.local_cells[1][1][1].num_particles,
            block.local_cells[0][0][0].num_particles + 
            block.local_cells[1][0][0].num_particles + 
            block.local_cells[0][1][0].num_particles + 
            block.local_cells[1][1][0].num_particles + 
            block.local_cells[0][0][1].num_particles + 
            block.local_cells[1][0][1].num_particles + 
            block.local_cells[0][1][1].num_particles + 
            block.local_cells[1][1][1].num_particles);


    // Starts MC cycle inside each block

    // Shuffle cell order inside master_rank
    if (rank == master_rank) {
        // Shuffle cell activating order
        shuffle_int_array(block.cells_index, 8);

        // Send the cell order to other ranks
        for (int i=1; i < n_ranks; i++) {
            MPI_Send(block.cells_index, 8, MPI_INT, i, i, MPI_COMM_WORLD);
        }
    } else {
        //Receive the cell ordering
        MPI_Recv(block.cells_index, 8, MPI_INT, master_rank, rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    // Loop through each cell in the block to activate them
    for (int i = 0; i < 8; i++) {
        
        // Convert index to cell coords
        int active_cell_index = block.cells_index[i];
        int active_cell_coords[3];
        
        index_to_cell_coords(active_cell_index, 
                             &active_cell_coords[0], 
                             &active_cell_coords[1], 
                             &active_cell_coords[2]);

        // Activate cell [i]
        // Implicitly handled by build_neighbor_cells which considers 'i' as active

        // Build neighbouring cells
        // ### NEED TO CREATE THE COPY, PACK AND UNPACK CELL FUNCTIONS
        build_neighbor_cells(&block, i, grid_dims);
        
        // Once created the neighbouring 3x3x3 cells
        // Loop through each active cell particle and calculate neighbours
        // Active cell is always at [1][1][1] in extended_grid
        Cell* active_cell = &block.extended_grid[1][1][1];
        
        if (active_cell->num_particles > 0) {
            for (int p=0; p < active_cell->num_particles; p++) {
                
                // Particle* particle = &active_cell->particles[p];
                
                // TODO: Implement MC move for Particle struct
                // This requires adapting 'attempt_monte_carlo_move' to use 'Particle' struct
                // and the 3x3x3 'extended_grid' for neighbor search.
                // For now, we leave the structure ready.
                
                // Steps:
                // 1. Calculate Energy (using extended_grid neighbors)
                // 2. Propose Move
                // 3. Check Overlaps (using extended_grid)
                // 4. Calculate New Energy
                // 5. Metropolis Criterion
            }
        }
        
        // Reset (free and malloc-clean) the neighbouring cells 
        clean_neighbor_cells(&block);

        // There is no need to send/recv cells unless to build neighb 3x3x3 cells
    }

   
    MPI_Barrier(MPI_COMM_WORLD); // ######################################################


    if (rank == master_rank) {
        free_3_arrays(&position_x, &position_y, &position_z);
        free_3_arrays(&orient_x, &orient_y, &orient_z);
        free_3_arrays(&rxi, &ryi, &rzi);

        for (int i = 0; i < num_particles; i++) {
            particle_free(&box_particles[i]);
        }
        free(box_particles);  box_particles = NULL;
    }
    
    block_free(&block);

    return MPI_Finalize();

    return 0;




























    // // Simulation control variables
	// int current_cycle ;                 // Current MC cycle counter
    // int i_Nbyt;               // Memory size for neighbor neighbor_list_array indexing array (in bytes)
    // int i_nbbyt;              // Memory size for neighbor neighbor_list_array data array (in bytes)  
    // int max_neighbs ;              // Maximum number of neighbors per particle
    // int oesti;                // Another neighbor estimation parameter
	// int *neighbor_count_array;               // Neighbor neighbor_list_array: neighbor_count_array[i] = end index for particle i's neighbors in 'neighbor_list_array'
    // int *neighbor_list_array;                // Neighbor neighbor_list_array: contains actual neighbor indices for all particles
    // int neighbor_list_update_needed;               // Flag indicating if neighbor neighbor_list_array needs updating (1=update, 0=no update)
    // int check_energy = 0;     // Debug flag to verify system_energy calculations (0=off, 1=on)
	
    // // Statistics
	// long accepted_moves = 0;          // Counter for accepted MC moves in current block
    // long attempted_moves = 0;         // Counter for total attempted MC moves in current block
	// float acceptance_ratio ;             // Acceptance percentage = (accept/attempted_moves)*100
    // float neighbors_update_factor  = 1.0;     // Factor for neighbor neighbor_list_array update criterion

    // // Particle Data Arrays
	// double *position_x, *position_y, *position_z;      // Current particle positions (x, y, z coordinates)
    // double *rxi, *ryi, *rzi;   // Reference positions for neighbor neighbor_list_array updates
	// double *orient_x, *orient_y, *orient_z;      // Particle orientation vectors (unit vectors)
    
    // // Energy and Move Tracking
    // double system_energy = 0.0;       // Total system system_energy (Gay-Berne + colloid interactions)
    // double max_displacement_squared = 0.0;        // Maximum displacement squared since last neighbor neighbor_list_array update
    // double squared_skin_depth;            // Threshold squared distance for neighbor neighbor_list_array updates

    // Time
    clock_t start_time, end_time;

	start_time = clock();   // Inicia reloj para medir tiempo de ejecución

	/************* pasar a función ***************/
    /*
     * if (sodium_init() < 0){
	 *     printf("No sodium no party\n");
	 * }
    */
	/********************************************/

    // Reads the file 'run_mcGBwcolloid.txt' which contains the initial parameters
    read_simulation_parameters();

    // Start random seed
	srand(time(0));

    /* Allocate memory for 3 1D arrays
     * position_x,  position_y,  position_z  are the particle positions (x, y, z coordinates)
     * rx1, ryi, rzi are the reference positions for neighbor neighbor_list_array updates
     * orient_x,  orient_y,  orient_z  are the particle orientation vectors (unit vectors)
     */
    callocate_3_dbl_arrays(&position_x,  &position_y,  &position_z,  num_particles);
    callocate_3_dbl_arrays(&orient_x,  &orient_y,  &orient_z,  num_particles);
    callocate_3_dbl_arrays(&rxi, &ryi, &rzi, num_particles);

    // Select how to start the initial positions and orientations of each particle
    get_initial_config(initial_configuration_type,
                       position_x, position_y, position_z, 
                       orient_x, orient_y, orient_z);

    // Generate uniformly POINTS_ON_SPHERE over the unitary sphere surface
    // Note that in Marsaglia4MC we use POINTS_ON_SPHERE / 2
    // because it generates a random 3D point and its negative
    Marsaglia4MC(rot_vec_x, rot_vec_y, rot_vec_z, 1.0);

    // Calculate some constants
	get_system_const();

	estima_vecinos(&max_neighbs , &oesti, &squared_skin_depth, &i_Nbyt, &i_nbbyt, &neighbors_update_factor );

	neighbor_list_array    = (int *) malloc(i_nbbyt);   // List with a size for num_particles
	neighbor_count_array   = (int *) malloc(i_Nbyt);    // List with a size for num_particles * max_neighbs  (max_neighbour_per_particle)

    // Print simulation parameters so far
	print_resum();

    // Minimum mesogen thickness
	float min_thick;
	min_thick = (gb_sigma_endtoend < gb_sigma_sidetoside ? gb_sigma_endtoend : gb_sigma_sidetoside);

    // Average parameters                     
    float n_ave = 0.0;
    float ave_ene, ave_dens, ave_ordpar, press_tns[4] = {0.0};
    float dir_x, dir_y, dir_z, global_order_parameter = 0.0, ave_PressTns[4];

    size_t *histogram;

    /* Promedios de parámetros */                     
	histogram = (size_t *) malloc(num_particles*sizeof(size_t));
	for (size_t cont=0; cont < num_particles; cont++){
	     histogram[cont] = 0;
	}

    // Initialize all average parameters to zero with option = 0
	update_running_averages(0, num_particles, &n_ave, &ave_ene, &ave_dens, ave_PressTns,
                            &ave_ordpar, system_energy, system_density, press_tns, global_order_parameter);

    // Calculate the number of neighbours and save it in neighbor_list_array
    // and neighbor_count_array tells me, in an elaborate way, how many neighbours each particle has
    build_neighbor_lists(max_neighbs * num_particles, 
                         position_x, position_y, position_z, 
                         rxi, ryi, rzi, 
                         neighbor_count_array, 
                         neighbor_list_array);

	system_energy = tot_ene(position_x, position_y, position_z, orient_x, orient_y, orient_z);

	EnergyCheck(system_energy, position_x, position_y, position_z, 
                               orient_x, orient_y, orient_z, 
                               rxi, ryi, rzi, 
                               neighbor_count_array, 
                               neighbor_list_array, 
                               max_neighbs);

	check_no_colloid_overlaps(position_x, position_y, position_z);

	compute_liquid_crystal_order_parameters(num_particles, 
                                            orient_x, orient_y, orient_z, 
                                            &dir_x, &dir_y, &dir_z, 
                                            &global_order_parameter);   // Calcula parámetro de orden global

    print_Trayectoria(position_x, position_y, position_z, orient_x, orient_y, orient_z, 0);  // Trayectoria con orientaciones

    /**
     * MAIN MONTE CARLO SIMULATION LOOP
     * 
     * This loop executes the core Monte Carlo sampling procedure for the specified
     * number of cycles. Each cycle represents one attempted move per particle on average.
     * The loop manages particle moves, neighbor list updates, data collection, and
     * adaptive step size adjustment.
     */
	for (current_cycle = 1; current_cycle < (total_mc_cycles + 1); current_cycle++)  // Inicio ciclo de MC
	{

        attempted_moves += 1;

        // STEP 1: Attempt Monte Carlo Move
        // Each cycle attempts one particle move (on average one move per particle per N cycles)
        attempt_monte_carlo_move(position_x, position_y, position_z, 
                                 orient_x, orient_y, orient_z, 
                                 max_displacement, 
                                 &system_energy, 
                                 neighbor_count_array, 
                                 neighbor_list_array, 
                                 &accepted_moves, 
                                 &max_displacement_squared,
                                 histogram);

        // STEP 2: Check if neighbor lists need updating
        // Neighbor lists become invalid when particles move too far from their reference positions
        // We track the maximum squared displacement and rebuild lists when it exceeds a threshold
	    neighbor_list_update_needed = (max_displacement_squared > neighbors_update_factor * squared_skin_depth);         // Reviso si es momento de actualizar lista (max_displacement_squared está al cuadrado)

        if (neighbor_list_update_needed) {

            // Reset displacement tracker
	        max_displacement_squared = 0.0;

            // Rebuild neighbor lists from current particle positions
            build_neighbor_lists(max_neighbs * num_particles, 
                                 position_x, position_y, position_z, 
                                 rxi, ryi, rzi, 
                                 neighbor_count_array, 
                                 neighbor_list_array);

            // Recalculate system energy with new neighbor lists
           // This ensures energy consistency after list rebuild
	        system_energy = tot_ene_lstnb(position_x, position_y, position_z, 
                                          orient_x, orient_y, orient_z, 
                                          neighbor_count_array, 
                                          neighbor_list_array);
        }

        // STEP 3: Periodic Configuration Saving and Intermediate Output
        // Save system state and print statistics at regular intervals
	    if (current_cycle  % configuration_save_interval == 0){

            // Print current simulation statistics to console
            print_intermedio(current_cycle, 
                             n_ave, 
                             ave_ene, 
                             system_energy, 
                             ave_ordpar, 
                             global_order_parameter, 
                             ave_dens, 
                             accepted_moves, 
                             attempted_moves, 
                             dir_x, dir_y, dir_z);

            // Optional energy verification for debugging
	        if (check_energy == 1)
            {
                EnergyCheck(system_energy, 
                            position_x, position_y, position_z, 
                            orient_x, orient_y, orient_z, 
                            rxi, ryi, rzi, 
                            neighbor_count_array, 
                            neighbor_list_array, 
                            max_neighbs );
            }

            // Save current configuration to trajectory file
            print_Trayectoria(position_x, position_y, position_z, orient_x, orient_y, orient_z, 1);
            // Create backup of restart file
            backup_restart_file(position_x, position_y, position_z, orient_x, orient_y, orient_z);              // backup of fortSphere.1
	    }

        // STEP 4: Periodic Order Parameter Calculation and Statistics Update
        // Calculate liquid crystal order parameters and update running averages
	    if (current_cycle  % statistics_print_interval == 0){

	        compute_liquid_crystal_order_parameters(num_particles, 
                                                    orient_x, orient_y, orient_z, 
                                                    &dir_x, &dir_y, &dir_z, 
                                                    &global_order_parameter);

		    update_running_averages(1, num_particles, 
                                    &n_ave, 
                                    &ave_ene, 
                                    &ave_dens, 
                                    ave_PressTns, 
                                    &ave_ordpar, 
                                    system_energy, 
		                            system_density, 
                                    press_tns, 
                                    global_order_parameter);                      // Calcula promedios
	    }

        // STEP 5: Adaptive Step Size Adjustment
        // Adjust move sizes based on acceptance ratio to maintain optimal sampling efficiency
        // This check occurs every 10,000 attempted moves (not cycles)
	    if (attempted_moves == 10000){  // Reviso desplazamientos cada 10000 intentos
	        acceptance_ratio  = (float)accepted_moves / ((float)attempted_moves);
		    
            // Accumulate global statistics
            total_accepted_moves += accepted_moves; 
            total_attempted_moves += attempted_moves;   
		    
            // Reset block counters for next 10,000 moves
            accepted_moves = 0;
            attempted_moves = 0;

            // Adjust move sizes based on acceptance ratio
	        if (acceptance_ratio  < 0.3)
            {
                // Acceptance too low - decrease move sizes for better acceptance
                max_displacement  -= max_displacement*0.05;
                
                // Prevent moves from becoming too small
                if (max_displacement < 0.0001*min_thick)
                {
		            max_displacement  = 0.01*min_thick;

                    // Also decrease rotation size if translation is at minimum
		            if (max_rotation > 0.000001) 
                    {
                        max_rotation -= 0.5*max_rotation;
                    }
		        }
	        }
	        else if (acceptance_ratio  > 0.55)
            {
                // Acceptance too high - increase move sizes for better sampling
                max_displacement  += max_displacement*0.05;
                
                // Also increase rotation size if translation is at maximum
                if (max_displacement > 1.5*min_thick) 
                {
		            max_displacement  = 0.75*min_thick;
		            if (max_rotation < 0.7)
                    {
                        max_rotation += 0.5*max_rotation;
                    }
		        }
	        }
	    }
	}   // END MONTE CARLO LOOP

	if (attempted_moves < 10000){
	    total_accepted_moves += accepted_moves; total_attempted_moves += attempted_moves;   // acumulo intentos para estadística global
	}

    update_running_averages(2, num_particles, &n_ave, &ave_ene, &ave_dens, ave_PressTns, &ave_ordpar, system_energy,
              system_density, press_tns, global_order_parameter);

	end_time = clock();

	print_resumfinal(ave_ene, ave_dens, ave_PressTns, ave_ordpar, 
			system_energy, dir_x, dir_y, dir_z, start_time, end_time);

	EnergyCheck(system_energy, position_x, position_y, position_z, orient_x, orient_y, orient_z, rxi, ryi, rzi, neighbor_count_array, neighbor_list_array, max_neighbs );

    print_Qmga(position_x, position_y, position_z, orient_x, orient_y, orient_z, 0); // Sólo escribo una configuración al final
	/**********/
	FILE* f_hist;
	f_hist = fopen("Histograma_Movimientos.dat","w");
	for (size_t cont=0; cont < num_particles; cont++){
		fprintf(f_hist,"%zu       %zu\n",cont,histogram[cont]);
	}
	fclose(f_hist);
	/**********/

	esc_fort1(position_x, position_y, position_z, orient_x, orient_y, orient_z);

	print_pos_xyz(position_x, position_y, position_z, 0);  // header of file
	print_pos_xyz(position_x, position_y, position_z, 1);  // last configuration without orientations

	free(histogram);
    free_3_arrays(&position_x, &position_y, &position_z);
    free_3_arrays(&orient_x, &orient_y, &orient_z);
    free_3_arrays(&rxi, &ryi, &rzi);
	free(neighbor_list_array);
	free(neighbor_count_array);

	return 0;

}

/**********************************************************************************************/
/********************                DEFINICIÓN DE FUNCIONES              *********************/
/**********************************************************************************************/

//****************          Lee variables iniciales desde run.txt       ************************

/**
 * Reads simulation parameters from self-documented configuration file
 * 
 * This function opens the configuration file "run_mcGBwcolloid.txt" and reads
 * all simulation parameters including system properties, Gay-Berne potential
 * parameters, Monte Carlo settings, and colloid properties.
 * 
 * The function ignores comment lines (starting with '#') and validates that
 * all expected parameters are successfully read.
 */
void read_simulation_parameters(void) 
{
    int parameters_read_successfully = 0;
    FILE *configuration_file;
    char line_buffer[256];

    // STEP 1: Open configuration file
    configuration_file = fopen("run_mcGBwcolloid.txt", "r");

    // STEP 2: Verify file opened successfully
    if (configuration_file == NULL) {
        perror("Error opening configuration file 'run_mcGBwcolloid.txt'");
        printf("\nCRITICAL ERROR: Configuration file not found in project directory.\n");
        printf("Please ensure 'run_mcGBwcolloid.txt' exists in the root folder.\n");
        exit(1);
    }

    // STEP 3: Read simulation ensemble type (NVT/NPT)
    if (read_next_non_comment_line(configuration_file, line_buffer, sizeof(line_buffer)) &&
        sscanf(line_buffer, "%d", &mc_ensemble_type) == 1)
        parameters_read_successfully += 1;
    
    // STEP 4: Read system properties
    if (read_next_non_comment_line(configuration_file, line_buffer, sizeof(line_buffer)) &&
        sscanf(line_buffer, "%d %lf %lf %lf %lf %lf", 
               &num_particles, &system_density, &system_temperature, 
               &system_pressure, &gb_sigma_sidetoside, &gb_epsilon0) == 6)
        parameters_read_successfully += 6;
    
    // STEP 5: Read Gay-Berne anisotropy parameters
    if (read_next_non_comment_line(configuration_file, line_buffer, sizeof(line_buffer)) &&
        sscanf(line_buffer, "%lf %lf %lf %lf", 
               &gb_kappa, &gb_kappa_prime, &gb_mu_exponent, &gb_nu_exponent) == 4)
        parameters_read_successfully += 4;
    
    // STEP 6: Read Monte Carlo move parameters
    if (read_next_non_comment_line(configuration_file, line_buffer, sizeof(line_buffer)) &&
        sscanf(line_buffer, "%f %f",  
               &max_displacement, &max_rotation) == 2)
        parameters_read_successfully += 2;
    
    // STEP 7: Read initial configuration type
    if (read_next_non_comment_line(configuration_file, line_buffer, sizeof(line_buffer)) &&
        sscanf(line_buffer, "%d", &initial_configuration_type) == 1)
        parameters_read_successfully += 1;
    
    // STEP 8: Read interaction cutoff parameters
    if (read_next_non_comment_line(configuration_file, line_buffer, sizeof(line_buffer)) &&
        sscanf(line_buffer, "%lf %lf", 
               &cutoff_radius, &verlet_radius) == 2)
        parameters_read_successfully += 2;
    
    // STEP 9: Read Monte Carlo cycle parameters
    if (read_next_non_comment_line(configuration_file, line_buffer, sizeof(line_buffer)) &&
        sscanf(line_buffer, "%d %d %d", 
               &total_mc_cycles, &configuration_save_interval, &statistics_print_interval) == 3)
        parameters_read_successfully += 3;
    
    // STEP 10: Read colloid interaction parameters
    if (read_next_non_comment_line(configuration_file, line_buffer, sizeof(line_buffer)) &&
        sscanf(line_buffer, "%d %lf %lf %lf", 
               &colloid_wall_option, &colloid_interaction_epsilon, 
               &colloid_interaction_amplitude, &colloid_radius) == 4)
        parameters_read_successfully += 4;

    // STEP 11: Validate all parameters were read successfully
    const int expected_parameter_count = 23;
    if (parameters_read_successfully != expected_parameter_count) {
        printf("\nERROR: Parameter reading incomplete. Expected %d values, read %d.\n", 
               expected_parameter_count, parameters_read_successfully);
        printf("Missing or invalid parameters in configuration file.\n");
        exit(1);
    }

    // STEP 12: Calculate derived Gay-Berne parameter
    gb_sigma_endtoend = gb_sigma_sidetoside * gb_kappa;

    // STEP 13: Close configuration file
    fclose(configuration_file);
    
    printf("Successfully read %d parameters from configuration file.\n", parameters_read_successfully);
}

/**
 * Helper function to read the next non-comment line from file
 * Returns 1 if a valid line was found, 0 on EOF or error
 */
int read_next_non_comment_line(FILE *file, char *buffer, size_t buffer_size) 
{
    while (fgets(buffer, buffer_size, file)) {
        // Skip empty lines and comment lines
        if (buffer[0] != '#' && buffer[0] != '\n' && buffer[0] != '\r') {
            // Remove trailing newline characters
            size_t len = strlen(buffer);
            if (len > 0 && buffer[len-1] == '\n') buffer[len-1] = '\0';
            if (len > 1 && buffer[len-2] == '\r') buffer[len-2] = '\0';
            return 1;
        }
    }
    return 0;
}

//*********************************************************************************************
//************           Imprime encabezado con resumen de la simulación       ****************
void print_resum(void) {
	char ensamble[] = {'N', 'V','T', '\0' }; // Default

	if (mc_ensemble_type == 2) ensamble[1] = 'P';
	else mc_ensemble_type = 1;      // Tratando de evitar errores

	printf("\n");
	printf("\t        Simulación de Monte Carlo %s - GB   \t\n", ensamble);
	printf("\t                     DISB                     \t\n");
	printf("\n");
	printf("Número de partículas                         :%12d    \n",      num_particles);
	printf("Ancho mesógeno (Gay-Berne)                   :     %2.5lf \n",  gb_sigma_sidetoside);
	printf("Largo mesógeno (Gay-Berne)                   :     %2.5lf \n",  gb_sigma_endtoend);
	printf("Densidad                                     :     %2.5lf \n",  system_density);
	printf("Temperatura                                  :     %2.5lf \n",  system_temperature);
	printf("Presión                                      :     %2.5lf \n",  system_pressure);
	printf("Kappa  (Gay-Berne)                           :     %2.5lf \n",  gb_kappa);
	printf("Kappa' (Gay-Berne)                           :     %2.5lf \n",  gb_kappa_prime);
	printf("Mu     (Gay-Berne)                           :     %2.5lf \n",  gb_mu_exponent);
	printf("Nu     (Gay-Berne)                           :     %2.5lf \n",  gb_nu_exponent);
	printf("X      (Gay-Berne)                           :     %2.5lf \n",  gb_chi);
	printf("X'     (Gay-Berne)                           :     %2.5lf \n",  gb_chi_prime);
	printf("Momento de inercia                           :     %2.5lf \n",  gb_inercia);
	printf("Radio de corte                               :     %2.5lf \n",  cutoff_radius);
	printf("Máxima longitud de desplazamiento partícula  :     %2.8lf \n",  max_displacement);
	printf("Norma vector aleatorio en esfera (Marsaglia) :     %2.8lf \n",  max_rotation);
	printf("Número de ciclos                             :%12d    \n",      total_mc_cycles);
	printf("Radio de verlet (lista vecinos)              :     %2.5lf \n",  verlet_radius);
	printf("Guarda configuración cada                    :%12d    \n",      configuration_save_interval);
	printf("Imprime energía en pantalla cada             :%12d    \n",      statistics_print_interval);
	if (initial_configuration_type == 0) printf("Configuración inicial de la simulación       : 0 (fortSphere.1) \n");
	if (initial_configuration_type == 1) printf("Configuración inicial de la simulación       : 1 (random) \n");
	if (initial_configuration_type == 2) printf("Configuración inicial de la simulación       : 2 (cubic)  \n");
	printf("Radio de colloide                            :%2.5lf\n", colloid_radius);
	printf("\n");
	printf("\n");
    fflush(stdout);

}
//*********************************************************************************************
//************                Imprime estado actual de la simulación           ****************
void print_intermedio(int current_cycle , 
                      float n_ave, 
                      float ave_ene, 
                      float system_energy, 
                      float ave_ordpar, 
                      float global_order_parameter, 
                      float ave_dens, 
                      int accepted_moves, 
                      int attempted_moves, 
                      float dir_x, float dir_y, float dir_z)
{

    printf("\n *********   Al paso %d, promediando %d configuraciones:   ********* \n", current_cycle , ((int)n_ave));
	printf("Energía promedio:      %5.7f \n", ave_ene/n_ave);
	printf("Energía instantánea:   %5.7f \n", system_energy);
	printf("Par. orden promedio:   %5.7f \n", ave_ordpar/n_ave);
	printf("Par. orden:      %5.7f \n", global_order_parameter);
	printf("Densidad promedio:     %5.7f \n", ave_dens/n_ave);
	printf("Densidad  :      %5.7f \n", system_density);
	printf("Dimensiones caja : %3.5f x %3.5f x %3.5f \n", box_size[0], box_size[1], box_size[2]);
	printf("Vomumen   :      %5.7f \n", box_volume);
	printf("Director x:      %5.7f \n", dir_x);
	printf("Director y:      %5.7f \n", dir_y);
	printf("Director z:      %5.7f \n", dir_z);
	printf("Particle displacement acceptance ratio: %f \n", 100.0*(float)accepted_moves/((float)attempted_moves));
	printf("Desplazamiento part = %f; Rotación part = %f\n", max_displacement, max_rotation);

}


//*********************************************************************************************
//************                   Imprime un resumen de la simulación           ****************
void print_resumfinal(float ave_ene, float ave_dens, float ave_PressTns[4], float ave_ordpar,
		float system_energy, float dir_x, float dir_y, float dir_z, clock_t start, clock_t end) {

	float pordr;
	pordr = 100.0*(float)total_accepted_moves/((float)total_attempted_moves);

	double total_time, segundos = 0.0;
	size_t dias = 0, horas = 0, minutos = 0;
	total_time = ((double)end - (double)start)/CLOCKS_PER_SEC;
	if (total_time > 86400){
		dias = (size_t)(total_time/86400.0);
		total_time -= (double)(dias*86400);
	}
	if (total_time > 3600){
		horas = (size_t)(total_time/3600.0);
		total_time -= (double)(horas*3600);
	}
	if (total_time > 60){
		minutos = (size_t)(total_time/60.0);
		total_time -= (double)(minutos*60);
	}
	segundos = total_time;

	char ensamble[] = {'N', 'V','T', '\0' }; // Default
	if (mc_ensemble_type == 2) ensamble[1] = 'P';

	printf("\n *********************************************************** \n");
	printf("\t        Final de la simulación MC %s          \t\n", ensamble);
	printf("\t  Tiempo total de ejecución: %zu días, %zu hrs, %zu mins %2.5lf segs  \t\n",
			dias, horas, minutos, segundos);
	printf(" *********************************************************** \n");
	printf("Número de partículas                         :%12d    \n",      num_particles);
	printf("Temperatura                                  :     %2.5lf \n",  system_temperature);
	printf("Presión                                      :     %2.5lf \n",  system_pressure);
	printf("Energía promedio                             :     %5.7f \n", ave_ene);
	printf("Densidad promedio                            :     %5.7f \n", ave_dens);
	printf("Presión promedio                             :     %5.7f \n", ave_PressTns[0]);
	printf("Par. orden promedio                          :     %5.7f \n", ave_ordpar);
	printf("Energía final                                :     %5.7f \n", system_energy);
	printf("Densidad final                               :     %5.7f \n", system_density);
	printf("Dimensiones caja                             : %3.5f x %3.5f x %3.5f \n", box_size[0], 
                                                                                      box_size[1], 
                                                                                      box_size[2]);
	printf("Radio colloide                               :     %2.5lf \n",  colloid_radius);
	printf("Director x                                   :     %5.7f \n", dir_x);
	printf("Director y                                   :     %5.7f \n", dir_y);
	printf("Director z                                   :     %5.7f \n", dir_z);
	printf("Máxima longitud de desplazamiento partícula  :     %2.8lf \n",  max_displacement);
	printf("Norma vector aleatorio en esfera (Marsaglia) :     %2.8lf \n",  max_rotation);
	printf("Porcentaje de aceptación movimiento partícula:     %2.5f \n", pordr);

	printf("\n");
	printf("\n");

}
//*********************************************************************************************
//**************  Revision of positions of nematogens to avoid overlaps with colloids  *************
void check_no_colloid_overlaps(double *position_x, double *position_y, double *position_z){
     
    double dx, dy, dz, r2;
    int    i;

    for (i=0; i<num_particles; i++){
        dx = position_x[i] - 0.5*box_size[0];
        dy = position_y[i] - 0.5*box_size[1];
        dz = position_z[i] - 0.5*box_size[2];
        r2 = dx*dx + dy*dy + dz*dz;
        
        if (r2 < (colloid_radius*colloid_radius)) {
            printf("Error in initial conditions: mesogen %d is inside the colloid\n", i);
            exit(1);
        }
    }
}
//*********************************************************************************************
//**************  Una revisión del valor de la energía del sistema con 3 métodos  *************
void EnergyCheck(double system_energy, double *position_x, double *position_y, double *position_z, double *orient_x, double *orient_y, 
		double *orient_z, double *rxi, double *ryi, double *rzi, int *neighbor_count_array, int *neighbor_list_array, 
		int max_neig){
	double inv_nump = 1.0/num_particles;
	printf("Revisando cálculo de energía total del sistema: \n\n");

	double ener_plain = tot_ene(position_x, position_y, position_z, orient_x, orient_y, orient_z);
	double ener_list = tot_ene_lstnb(position_x, position_y, position_z, orient_x, orient_y, orient_z, neighbor_count_array, neighbor_list_array);
	build_neighbor_lists(max_neig*num_particles, position_x, position_y, position_z, rxi, ryi, rzi, neighbor_count_array, neighbor_list_array);
	double ener_nwlist = tot_ene_lstnb(position_x, position_y, position_z, orient_x, orient_y, orient_z, neighbor_count_array, neighbor_list_array);

	printf("Valor actual en la simulación     : %8.4lf ; E/N: %3.7lf\n", system_energy, 
			system_energy*inv_nump);
	printf("Recalculando el valor             : %8.4lf ; E/N: %3.7lf\n", ener_plain, 
			ener_plain*inv_nump);
	printf("Valor obtenido con última lista   : %8.4lf ; E/N: %3.7lf\n", ener_list, 
			ener_list*inv_nump);
	printf("Valor obtenido con una nueva lista: %8.4lf ; E/N: %3.7lf\n", ener_nwlist, 
			ener_nwlist*inv_nump);
}
//*********************************************************************************************
//*************************  Lista de vecinos para cada hebra (j>i) *******************************
void list_vecRed(double *position_x, double *position_y, double *position_z, double *rxo, double *ryo, double *rzo, int *neighbor_count_array, int *neighbor_list_array){
   int    i, j, veci_loc;
   double dx, dy, dz, r2;

   veci_loc = 0;

#pragma omp parallel for private(i)
   for (i=0; i<num_particles; i++)
   {
   	rxo[i] = position_x[i];
   	ryo[i] = position_y[i];
   	rzo[i] = position_z[i];
   }

   for (i=0; i<num_particles-1 ; i++)
   {
        for (j=i+1; j<num_particles; j++)
        {
             r2 = compute_distance_sq(position_x[i], position_y[i], position_z[i], position_x[j], position_y[j], position_z[j], &dx, &dy, &dz);

             if (r2 < r_verlet_sq)
	     {
                 neighbor_list_array[veci_loc] = j;
                 veci_loc      += 1;
             }
        }
        neighbor_count_array[i] = veci_loc;
   }
}
//*************************      Lista total de vecinos para cada partícula    ********************
void build_neighbor_lists(int nvest, double *position_x, double *position_y, double *position_z, double *rxo, double *ryo, double *rzo, 
		         int *neighbor_count_array, int *neighbor_list_array){
    
    int    i, j;
    int    veci_loc;        
    int    loc_cont;        // Counter for the number of neighbours for each particle
    double dx, dy, dz, r2;

    veci_loc = 0;

    #pragma omp parallel for private(i)
    for (i=0; i<num_particles; i++)
    {
        rxo[i] = position_x[i];
        ryo[i] = position_y[i];
        rzo[i] = position_z[i];
    }

    for (i=0; i<num_particles ; i++)
    {
        loc_cont = 0;
        
        #pragma omp parallel for private(j, dx, dy, dz, r2) reduction(+: loc_cont)
        for (j=0; j<num_particles; j++)
        {
            if (i != j)
            {
                r2 = compute_distance_sq(position_x[i], position_y[i], position_z[i], position_x[j], position_y[j], position_z[j], &dx, &dy, &dz);
                
                // If particle j is inside the cut radius around particle i, then it's its neighbour
                if (r2 < r_verlet_sq) loc_cont += 1;
            }
        }
        
        // In this loop, veci_loc acts as a cummulative sum
        veci_loc += loc_cont;

        // Check if the stimated number of neighbours * number of particles is greater than the actual
        // calculated total number of neighbours veci_loc
        if (veci_loc >= nvest) printf("veci_loc = %d y nvest = %d; i = %d \n", veci_loc, nvest, i);  
        
        // neighbor_count_array saves the cummulative sum of neighbours and will act as an index saver
        neighbor_count_array[i]  = veci_loc;
    }

    /* Now, the for i loops over each particle again, but it saves
     * in neighbor_list_array the index of the neighbour and neighbor_count_array acts as a delimiter
     * to say that for particle 0, you have neighbor_count_array[0] neighbours which indexes are 
     * saved in the neighbor_list_array array
     * NOTE THAT THIS IS NOT THE BEST IF WE WANT TO BOOST THE PROGRAM
     */
    #pragma omp parallel for private(i, j, r2, dx, dy, dz, loc_cont, veci_loc)
    for (i=0; i<num_particles ; i++)
    {
        veci_loc = 0;
        loc_cont = 0;

        if (i != 0) veci_loc = neighbor_count_array[i-1];

        for (j=0; j<num_particles; j++)
        {
            if (i != j)
            {
                r2 = compute_distance_sq(position_x[i], position_y[i], position_z[i], position_x[j], position_y[j], position_z[j], &dx, &dy, &dz);

                if (r2 < r_verlet_sq)
                {
                    neighbor_list_array[veci_loc] = j;
                    veci_loc      += 1;
                }
            }
        }
    }
}
//*********************************************************************************************
//*************************    Estima desplazamiento máximo de partículas  ************************
double max_dsp(double *position_x, double *position_y, double *position_z, double *rxo, double *ryo, double *rzo){
	int    i;
	double dx, dy, dz, dspmx, dspind;

	dspmx    = 0.0;

	for (i=0; i<num_particles ; i++)
	{

		dx = position_x[i] - rxo[i];
		dy = position_y[i] - ryo[i];
		dz = position_z[i] - rzo[i];

		//      minima_ima(&dx, &dy, &dz);

		dx = fabs(dx);
		dy = fabs(dy);
		dz = fabs(dz);

		dspind = dx >= dy ? (dx >= dz ? dx : dz) : (dy >= dz ? dy : dz); // mayor de los 3

		if (dspind >= dspmx) dspmx = dspind;                             // mayor absoluto

	}

	return dspmx;

}
//*********************************************************************************************
//***************    Calculate interaction between particle i vs everyone else    *************
double GBij(int i, 
            double *position_x, double *position_y, double *position_z, 
            double *orient_x, double *orient_y, double *orient_z, 
            double fij[3]) {
   
    int    j;
    double dx, dy, dz, r2, ei2, ej2, nei, nej, PhiGB, total_interaction_energy, rij, irij;
    double comprx, compry, comprz, pres = 0.0, cp[3], iei, iej, eiej;

    memset(fij, 0, 3*sizeof(double));
    total_interaction_energy = 0.0;

    #pragma omp parallel for private(r2, dx, dy, dz, j, rij, irij, ei2, ej2, nei, nej, iei, iej, eiej, PhiGB) reduction(+: total_interaction_energy)
    for (j=0; j<num_particles ; j++)
    {
        if (i != j)
        {
            // Distance between particles i and j
            r2 = compute_distance_sq(position_x[i], position_y[i], position_z[i], position_x[j], position_y[j], position_z[j], &dx, &dy, &dz);

            // If the distance between i and j is less than the cut distance, proceed
            if (r2 < cutoff_radius_sq)
            {
                rij     = sqrt(r2);
                irij    = 1.0/rij;
                ei2     = compute_dot_product(orient_x[i], orient_y[i], orient_z[i], 
                                              orient_x[i], orient_y[i], orient_z[i]);  // Squared magnitude of i
                ej2     = compute_dot_product(orient_x[j], orient_y[j], orient_z[j], 
                                              orient_x[j], orient_y[j], orient_z[j]);  // Squared magnitude of j
                nei     = sqrt(ei2);                                       // Magnitude of i
                nej     = sqrt(ej2);                                       // Magnitude of j
                iei     = 1.0/nei; 
                iej     = 1.0/nej;
                eiej    = iei*iej * 
                          compute_dot_product(orient_x[i], orient_y[i], orient_z[i], 
                                              orient_x[j], orient_y[j], orient_z[j]);  // Cosine of the angle between vectors i and j
                
                PhiGB   = pij_GB((double [3]) {orient_x[i], orient_y[i], orient_z[i]}, 
                                 (double [3]) {orient_x[j], orient_y[j], orient_z[j]}, 
                                 rij, irij, iei, iej, eiej, 
                                 (double [3]) {dx, dy, dz});
                total_interaction_energy  += PhiGB;

                fij[0] += -4.0*gb_epsilon0*0.0;
                fij[1] += -4.0*gb_epsilon0*0.0;
                fij[2] += -4.0*gb_epsilon0*0.0;
                comprx  = fij[0]*dx;
                compry  = fij[1]*dy;
                comprz  = fij[2]*dz;
                cp[0]  += comprx;
                cp[1]  += compry;
                cp[2]  += comprz;
                pres   += comprx + compry + comprz;
            }
        }
    }

    return total_interaction_energy;
}
//*********************************************************************************************
//*************************    Interacción GB de una part. con todas    ***********************
double GBij_PosMod(int i, double ri[3], double ei[3], double *position_x, double *position_y, double *position_z, double *orient_x, double *orient_y, double *orient_z){
   
    int    j;
    double dx, dy, dz, r2, ei2, ej2, nei, nej, PhiGB, total_interaction_energy, rij, irij;
    double iei, iej, eiej;

    total_interaction_energy = 0.0;

    #pragma omp parallel for private(r2, dx, dy, dz, j, rij, irij, ei2, ej2, nei, nej, iei, iej, eiej, PhiGB) reduction(+: total_interaction_energy)
    for (j=0; j<num_particles ; j++)
    {
        if (i != j)
        {
            r2 = compute_distance_sq(ri[0], ri[1], ri[2], position_x[j], position_y[j], position_z[j], &dx, &dy, &dz);

            if (r2 < cutoff_radius_sq)
            {
                rij     = sqrt(r2);
                irij    = 1.0/rij;
                ei2     = compute_dot_product(ei[0], ei[1], ei[2], ei[0], ei[1], ei[2]);
                ej2     = compute_dot_product(orient_x[j], orient_y[j], orient_z[j], orient_x[j], orient_y[j], orient_z[j]);
                nei     = sqrt(ei2);
                nej     = sqrt(ej2);
                iei     = 1.0/nei; 
                iej     = 1.0/nej;
                eiej    = iei*iej*compute_dot_product(ei[0], ei[1], ei[2], orient_x[j], orient_y[j], orient_z[j]);
                PhiGB   = pij_GB(ei, (double [3]) {orient_x[j], orient_y[j], orient_z[j]}, rij, irij, iei, iej, eiej, (double [3]) {dx, dy, dz});
                total_interaction_energy  += PhiGB;
            }
        }
    }

    return total_interaction_energy;
}
//*********************************************************************************************
//*************************     Energía total del sistema con pot GB    ***********************
double tot_ene(double *position_x, double *position_y, double *position_z, double *orient_x, double *orient_y, double *orient_z){
    
    int    i;
    double dumv, wall, fzai[3];

    dumv = 0.0;
    wall = 0.0;

    for (i=0; i<num_particles ; i++)
    {
        dumv += GBij(i, position_x, position_y, position_z, orient_x, orient_y, orient_z, fzai);   // El ciclo está paralelizado dentro de la función
    }

    if (colloid_wall_option == 1){
        #pragma omp parallel for reduction(+: wall)
        for (i=0; i<num_particles ; i++)
        {
            wall += GBCollRod(i, (double [3]) {position_x[i], position_y[i], position_z[i]}, 
                                 (double [3]) {orient_x[i], orient_y[i], orient_z[i]}, 
                                 colloid_coords);
        }
    }

    return 0.5*dumv + wall;   // cuento doble: "i con j" y "j con i"
}
/**
 * Computes the Gay-Berne interaction energy for a particle with all its neighbors
 * using precomputed neighbor lists for efficient O(1) neighbor access.
 * 
 * This function calculates the total Gay-Berne potential energy between a specified
 * particle and all other particles within its interaction cutoff radius. The function
 * uses compressed neighbor lists to only check particles that are potentially within
 * the interaction range, significantly improving performance over O(N²) pairwise checks.
 * 
 * The function is parallelized using OpenMP to distribute the neighbor energy
 * calculations across multiple threads, with reduction for thread-safe energy summation.
 * 
 * @param particle_index          Index of the particle for which to compute energy
 * @param position_x              Array of x-coordinates for all particles
 * @param position_y              Array of y-coordinates for all particles  
 * @param position_z              Array of z-coordinates for all particles
 * @param orientation_x           Array of x-orientation vectors for all particles
 * @param orientation_y           Array of y-orientation vectors for all particles
 * @param orientation_z           Array of z-orientation vectors for all particles
 * @param neighbor_count_array    Array where neighbor_count_array[i] = end index of particle i's neighbors
 * @param neighbor_list_array     Flattened array containing all neighbor indices
 * 
 * @return Total Gay-Berne potential energy between particle i and all its neighbors
 */
double GBij_lstnb(int particle_index, 
                  double *position_x, double *position_y, double *position_z, 
                  double *orient_x, double *orient_y, double *orient_z, 
                  int *neighbor_count_array, int *neighbor_list_array) 
{

    int    j, k, neighbor_start_index, neighbor_end_index;
    double dx, dy, dz, r2, ei2, ej2, nei, nej, PhiGB, total_interaction_energy, rij, irij;
    double iei, iej, eiej;

    // STEP 1: Initialize energy accumulator
    total_interaction_energy = 0.0;

    // STEP 2: Determine range of neighbors in the flattened neighbor list
    // The neighbor list is stored in compressed format:
    // - neighbor_count_array[particle_index] contains the END index for particle_index's neighbors
    // - Neighbors for particle particle_index are stored in neighbor_list_array from 
    //   neighbor_count_array[particle_index-1] to neighbor_count_array[particle_index]
    neighbor_start_index = 0;
    if (particle_index > 0) 
    {
        neighbor_start_index = neighbor_count_array[particle_index - 1];
    }
    neighbor_end_index = neighbor_count_array[particle_index];

    // STEP 3: Parallel loop over all neighbors using OpenMP
    // Each thread processes a subset of neighbors and accumulates energy independently
    // The 'reduction(+:total_interaction_energy)' ensures thread-safe summation
    #pragma omp parallel for \
    private(k, j, r2, dx, dy, dz, rij, irij, ei2, ej2, nei, nej, iei, iej, eiej, PhiGB) reduction(+: total_interaction_energy)
    for (k=neighbor_start_index; k<neighbor_end_index ; k++)  // Interacción con otras partículas dentro de la lista de vecinos
        {
            j  = neighbor_list_array[k];
            r2 = compute_distance_sq(position_x[particle_index], position_y[particle_index], position_z[particle_index], position_x[j], position_y[j], position_z[j], &dx, &dy, &dz);

            if (r2 < cutoff_radius_sq)
            {
                rij     = sqrt(r2);
                irij    = 1.0/rij;
                ei2     = compute_dot_product(orient_x[particle_index], orient_y[particle_index], orient_z[particle_index], orient_x[particle_index], orient_y[particle_index], orient_z[particle_index]);
                ej2     = compute_dot_product(orient_x[j], orient_y[j], orient_z[j], orient_x[j], orient_y[j], orient_z[j]);
                nei     = sqrt(ei2);
                nej     = sqrt(ej2);
                iei     = 1.0/nei; 
                iej     = 1.0/nej;
                eiej    = iei*iej*compute_dot_product(orient_x[particle_index], orient_y[particle_index], orient_z[particle_index], orient_x[j], orient_y[j], orient_z[j]);
                PhiGB   = pij_GB((double [3]) {orient_x[particle_index], orient_y[particle_index], orient_z[particle_index]}, (double [3]) {orient_x[j], orient_y[j], orient_z[j]}, rij, irij, iei, iej, eiej, (double [3]) {dx, dy, dz});
                total_interaction_energy  += PhiGB;
            }
    }

    return total_interaction_energy;
}
//*********************************************************************************************
//*************************    Interacción GB de una part. con todas    ***********************
double GBij_PosMod_lstnb(int particle_index, double ri[3], double ei[3], double *position_x, double *position_y, double *position_z, double *orient_x, double *orient_y, double *orient_z, int *neighbor_count_array, int *neighbor_list_array){
	int    j, k, neighbor_start_index, neighbor_end_index;
	double dx, dy, dz, r2, ei2, ej2, nei, nej, PhiGB, total_interaction_energy, rij, irij;
	double iei, iej, eiej;

	total_interaction_energy = 0.0;
	neighbor_start_index   = 0;
	if (particle_index > 0) neighbor_start_index = neighbor_count_array[particle_index-1];
	neighbor_end_index   = neighbor_count_array[particle_index];

    #pragma omp parallel for private(k, j, r2, dx, dy, dz, rij, irij, ei2, ej2, nei, nej, iei, iej, eiej, PhiGB) reduction(+: total_interaction_energy)
	for (k=neighbor_start_index; k<neighbor_end_index ; k++)    // La lista cubre vecinos j > particle_index
	{

		j  = neighbor_list_array[k];
		r2 = compute_distance_sq(position_x[particle_index], position_y[particle_index], position_z[particle_index], position_x[j], position_y[j], position_z[j], &dx, &dy, &dz);

		if (r2 < cutoff_radius_sq)
		{

			rij     = sqrt(r2);
			irij    = 1.0/rij;
			ei2     = compute_dot_product(ei[0], ei[1], ei[2], ei[0], ei[1], ei[2]);
			ej2     = compute_dot_product(orient_x[j], orient_y[j], orient_z[j], orient_x[j], orient_y[j], orient_z[j]);
			nei     = sqrt(ei2);
			nej     = sqrt(ej2);
			iei     = 1.0/nei; 
			iej     = 1.0/nej;
			eiej    = iei*iej*compute_dot_product(ei[0], ei[1], ei[2], orient_x[j], orient_y[j], orient_z[j]);
			PhiGB   =  pij_GB(ei, (double [3]) {orient_x[j], orient_y[j], orient_z[j]}, rij, irij, iei, iej, eiej, (double [3]) {dx, dy, dz});
			total_interaction_energy  += PhiGB;

		}

	}

	return total_interaction_energy;

}
//*********************************************************************************************
//****************************   Colloid-nematogen interaction    *****************************
// Colloid-rod interaction (equation (9) in Antypov & Cleaver J. Phys.: Condens. Matter 16 (2004))
// Inputs:
//   i          : particle index (unused here but kept for API compatibility)
//   ri[3]      : position of rod center (x,y,z)
//   ei[3]      : unit orientation vector of the rod (ux,uy,uz)
//   Coll_pos[3]: position of colloid center (x,y,z)
// Globals used (must be defined elsewhere in your code):
//   double colloid_radius;         // colloid radius (R in the paper)
//   double gb_sigma_sidetoside;      // sigma_0 (rod breadth / sphere diameter)
//   double gb_chi;      // chi (shape anisotropy)  (eq.4)
//   double gb_chi_prime;     // chi' (system_energy anisotropy) (eq.6)
//   double gb_mu_exponent;       // mu exponent (µ)
//   double gb_kappa_prime;       // (used here as epsilon_CR0 scale) --- adjust if you have a different variable
//
// NOTE: If your code uses a different variable for the system_energy prefactor ε_CR0, replace gb_kappa_prime below by that variable.
double GBCollRod(int i, double ri[3], double ei[3], double Coll_pos[3]){
    double d_nc, dij;          // d_nc = r_ij (center-to-center), dij = d_nc - R (distance from colloid surface)
    double position_x, position_y, position_z;
    double rhatx, rhaty, rhatz;
    double rhat_e_dot, sin2theta;
    double sigma_CR, eps_CR;
    double x;                  // shorthand for (dij - sigma_CR + sigma0)
    double sigma0;
    double term1, term2, term3, term4;
    double U;

    /* compute center-to-center distance r_ij */
    d_nc = sqrt(compute_distance_sq(ri[0], ri[1], ri[2], Coll_pos[0], Coll_pos[1], Coll_pos[2], &position_x, &position_y, &position_z));

    /* dij is di_j = r_ij - R (distance measured from colloid surface to particle center) */
    dij = d_nc - colloid_radius;

    /* guard against zero centre-to-centre distance (shouldn't happen for real systems) */
    if (d_nc <= 1e-12) {
        /* rod center coincides with colloid centre -> extremely large repulsion (return large positive system_energy) */
        return 1.0e8;
    }
    // Potential truncated
    if (dij > colloid_rod_cutoff) return 0.0; // Cutoff distance 4.0*sigma_ss

    /* unit vector from colloid center to rod center */
    position_x = ri[0] - Coll_pos[0];
    position_y = ri[1] - Coll_pos[1];
    position_z = ri[2] - Coll_pos[2];
    rhatx = position_x / d_nc;
    rhaty = position_y / d_nc;
    rhatz = position_z / d_nc;

    /* cos(theta) = rhat · e_i (rod orientation); sin^2(theta) = 1 - cos^2(theta) */
    rhat_e_dot = rhatx * ei[0] + rhaty * ei[1] + rhatz * ei[2];
    if (rhat_e_dot > 1.0) rhat_e_dot = 1.0;
    if (rhat_e_dot < -1.0) rhat_e_dot = -1.0;
    sin2theta = 1.0 - rhat_e_dot * rhat_e_dot;

    /* sigma0 (paper's sigma_0) -> use gb_sigma_sidetoside (breadth / sphere diameter) */
    sigma0 = gb_sigma_sidetoside;

    /* --- orientation-dependent range sigma_CR (eq.10) ---
       sigma_CR = sigma0 * sqrt( (1 - chi * sin^2 theta) / (1 - chi) )
       (note: denominator 1 - chi assumed > 0; if not, you should revisit parameter choices)
    */
    {
        double denom = 1.0 - gb_chi;
        double numer  = 1.0 - gb_chi * sin2theta;
        if (denom <= 0.0) denom = 1e-12;   /* defensive */
        if (numer <= 1e-12) numer = 1e-12; /* prevent negative / zero under sqrt */
        sigma_CR = sigma0 * sqrt(numer / denom);
    }

    /* --- orientation-dependent well depth eps_CR (eq.11) ---
       eps_CR = eps_CR0 * ( (1 - chi') / (1 - chi' * sin^2 theta) )^mu
       Here I use gb_kappa_prime as eps_CR0 (adjust if your code stores this in another variable).
    */
    {
        double denom_eps = 1.0 - gb_chi_prime * sin2theta;
        double numer_eps  = 1.0 - gb_chi_prime;
        if (denom_eps <= 1e-12) denom_eps = 1e-12; /* defensive */
        /* choose prefactor for epsilon: replace gb_kappa_prime below if your code uses different var for eps_CR0 */
        eps_CR = gb_kappa_prime * pow( numer_eps / denom_eps, gb_mu_exponent );
    }

    /* Now build the bracketed combination from eq.(9):
       U_CR = eps_CR * [ sigma0^9/(45 * x^9) - sigma0^3/(6 * x^3) - sigma0^9/(40 * r_ij * x^8) + sigma0^3/(4 * r_ij * x^2) ]
       where x = (dij - sigma_CR + sigma0)
    */
    x = dij - sigma_CR + sigma0;

    /* Defensive handling: if the particle is too close (x <= 0) produce a large repulsive system_energy (avoid NaN/Inf) */
    if (x <= 1e-10) {
        /* Strong overlap -> very large positive system_energy */
        return 1.0e8;
    }

    /* compute powers in a numerically stable way (use pow for clarity) */
    {
        double s3 = sigma0 * sigma0 * sigma0;             /* sigma0^3 */
        double s9 = s3 * s3 * s3;                         /* sigma0^9 */
        double x2 = x * x;
        double x3 = x2 * x;
        double x8 = x3 * x3 * x2;
        double x9 = x8 * x;

        term1 = s9 / (45.0 * x9);
        term2 = s3 / (6.0 * x3);
        term3 = s9 / (40.0 * d_nc * x8);
        term4 = s3 / (4.0 * d_nc * x2);

        U = eps_CR * ( term1 - term2 - term3 + term4 );
    }

    return U;
}
//*********************************************************************************************
//*************************     Energía total del sistema con pot GB    ***********************
double tot_ene_lstnb(double *position_x, double *position_y, double *position_z, double *orient_x, double *orient_y, double *orient_z, int *neighbor_count_array, int *neighbor_list_array){
	int    i;
	double dumv, Uwall;
	//  double dumv, Uwall, fzai[3];

	dumv  = 0.0;
	Uwall = 0.0;

	for (i=0; i<num_particles ; i++)
	{
		dumv  += GBij_lstnb(i, position_x, position_y, position_z, orient_x, orient_y, orient_z, neighbor_count_array, neighbor_list_array);  // El ciclo está paralelizado dentro de la función
	}

	if (colloid_wall_option == 1){
#pragma omp parallel for reduction(+: Uwall)
		for (i=0; i<num_particles ; i++)
		{
			Uwall += GBCollRod(i, (double [3]) {position_x[i], position_y[i], position_z[i]}, 
                                  (double [3]) {orient_x[i], orient_y[i], orient_z[i]}, 
                                  colloid_coords);
		}
	}

	return 0.5*dumv + Uwall;

}
//*********************************************************************************************
//**********   Calcula parámetro de orden del sistema                               ***********
static inline int find_largest_eigenvalue_index(float a[3])
{
	int   ip, i;
	float val;

	ip  = 1;
	val = a[0];

	for (i=1; i<3; i++)
	{
		if (a[i] >= val)
		{
			ip  = i;
			val = a[i];
		}
	}

	return ip;
}

/**
 * Diagonalizes a 3x3 symmetric matrix using the Jacobi eigenvalue algorithm
 * 
 * The Jacobi method iteratively applies plane rotations to zero out off-diagonal
 * elements until the matrix becomes diagonal. The diagonal elements then become
 * the eigenvalues, and the product of all rotations gives the eigenvectors.
 * 
 * This implementation is specifically optimized for 3x3 symmetric matrices
 * and includes a robust eigenvector calculation method.
 * 
 * @param input_matrix     INPUT: 3x3 symmetric matrix to diagonalize
 *                        OUTPUT: Destroyed during diagonalization process
 * @param eigenvectors    OUTPUT: 3x3 matrix where columns are eigenvectors
 * @param eigenvalues     OUTPUT: Array of 3 eigenvalues in same order as eigenvectors
 */
void diagonalize_symmetric_matrix(float input_matrix[3][3], 
                                  float eigenvectors[3][3], 
                                  float eigenvalues[3])
{
    int i, j, pivot_row = 0, pivot_col = 0, previous_col;
    int convergence_achieved;
    
    // Jacobi algorithm variables
    float max_iterations = 13.5;      // Safety limit for 3x3 matrix
    float iteration_count = 0.0;
    float max_off_diagonal_value;
    
    // Rotation calculation variables
    float rotation_cosine, rotation_sine;
    float off_diagonal_squared, diagonal_difference;
    float rotation_angle_parameter;
    
    // Eigenvector calculation variables  
    float denominator1, denominator2;
    float x1_over_x3, x2_over_x3;
    
    // Store original matrix for eigenvector calculation
    float original_matrix[3][3];

    // STEP 1: Preserve original matrix for eigenvector calculation
    // The diagonalization process destroys the original matrix
    memcpy(original_matrix, input_matrix, sizeof(float) * 9);

    convergence_achieved = 0;  // Flag to indicate matrix is diagonal

    // STEP 2: Jacobi iteration loop - zero out off-diagonal elements
    while ((convergence_achieved == 0) && (iteration_count < max_iterations))
    {
        iteration_count += 1.0;
        max_off_diagonal_value = 0.0;

        // Search for the largest off-diagonal element in upper triangle
        for (j = 1; j < 3; j++) 
        {
            previous_col = j - 1;

            for (i = 0; i <= previous_col; i++) 
            {
                // Find the largest absolute off-diagonal value
                float abs_value = fabs(input_matrix[i][j]);

                if ((max_off_diagonal_value - abs_value) < 0.0) 
                {
                    max_off_diagonal_value = abs_value;
                    pivot_row = i;      // Row index of largest off-diagonal
                    pivot_col = j;      // Column index of largest off-diagonal
                }
            }
        }

        // STEP 3: Check convergence criteria
        // If largest off-diagonal is negligible, matrix is effectively diagonal
        if ((max_off_diagonal_value - 1.0e-14) <= 0.0)
        {
            convergence_achieved = 1;  // Matrix is diagonal
        }
        else
        {
            // STEP 4: Compute Jacobi rotation parameters
            // We'll rotate in the (pivot_row, pivot_col) plane to zero input_matrix[pivot_row][pivot_col]
            
            off_diagonal_squared = input_matrix[pivot_row][pivot_col] * input_matrix[pivot_row][pivot_col];
            diagonal_difference = input_matrix[pivot_row][pivot_row] - input_matrix[pivot_col][pivot_col];
            
            // Compute rotation angle parameter: tan(2θ) = 2*a_ij / (a_ii - a_jj)
            rotation_angle_parameter = sqrt(fabs(diagonal_difference * diagonal_difference + 4.0 * off_diagonal_squared));
            
            // Compute cosine of rotation angle: cos(θ)
            rotation_cosine = sqrt(fabs((rotation_angle_parameter + diagonal_difference) / (2.0 * rotation_angle_parameter)));

            // STEP 5: Compute sine of rotation angle ensuring numerical stability
            if (rotation_cosine <= sqrt(2)/2.)  // If cos(θ) ≤ 1/√2 (θ ≥ 45°)
            {
                // Use identity: sin²(θ) + cos²(θ) = 1
                rotation_sine = -rotation_cosine;
                rotation_cosine = sqrt(1.0 - rotation_sine * rotation_sine);
            }
            else  // If cos(θ) > 1/√2 (θ < 45°)
            {
                rotation_sine = -sqrt(1.0 - rotation_cosine * rotation_cosine);
            }

            // Adjust sign of sine based on the ratio of diagonal difference to off-diagonal
            if (diagonal_difference / input_matrix[pivot_row][pivot_col] < 0.0) 
            {
                rotation_sine = -rotation_sine;
            }

            // STEP 6: Apply Jacobi rotation to all affected rows and columns
            // The rotation matrix is: [ cos(θ)  -sin(θ) ]
            //                        [ sin(θ)   cos(θ) ]
            for (j = 0; j < 3; j++)
            {
                if ((pivot_row != j) && (pivot_col != j))
                {
                    // Apply rotation to pivot_row and pivot_col rows
                    float temp1 = rotation_cosine * input_matrix[j][pivot_row] - rotation_sine * input_matrix[j][pivot_col];
                    float temp2 = rotation_sine * input_matrix[j][pivot_row] + rotation_cosine * input_matrix[j][pivot_col];
                    
                    input_matrix[pivot_row][j] = temp1;
                    input_matrix[pivot_col][j] = temp2;
                    
                    // Maintain symmetry
                    input_matrix[j][pivot_row] = input_matrix[pivot_row][j];
                    input_matrix[j][pivot_col] = input_matrix[pivot_col][j];
                }
            }

            // STEP 7: Update diagonal elements after rotation
            // The trace is preserved: a_kk + a_ll remains constant
            float diagonal_sum = input_matrix[pivot_row][pivot_row] + input_matrix[pivot_col][pivot_col];
            
            // New diagonal elements after rotation
            input_matrix[pivot_row][pivot_row] = rotation_cosine * rotation_cosine * input_matrix[pivot_row][pivot_row] + 
                                                rotation_sine * rotation_sine * input_matrix[pivot_col][pivot_col] - 
                                                2.0 * rotation_cosine * rotation_sine * input_matrix[pivot_row][pivot_col];
            
            input_matrix[pivot_col][pivot_col] = diagonal_sum - input_matrix[pivot_row][pivot_row];
            
            // Zero out the rotated off-diagonal elements
            input_matrix[pivot_col][pivot_row] = 0.0;
            input_matrix[pivot_row][pivot_col] = 0.0;
        }
    }

    // STEP 8: Check for convergence failure
    if (iteration_count >= max_iterations) 
    {
        printf("Error: Matrix diagonalization failed to converge in Jacobi method\n");
    }

    // STEP 9: Extract eigenvalues from diagonal elements
    // After convergence, diagonal elements are the eigenvalues
    eigenvalues[0] = input_matrix[0][0];
    eigenvalues[1] = input_matrix[1][1];
    eigenvalues[2] = input_matrix[2][2];

    // STEP 10: Compute eigenvectors using original matrix
    // For each eigenvalue λ, solve: (A - λI)v = 0 for eigenvector v
    for (i = 0; i < 3; i++)
    {
        float current_eigenvalue = eigenvalues[i];
        
        // Solve for eigenvector component ratios using Cramer's rule approach
        // We solve the system by expressing x1 and x2 in terms of x3
        
        // Denominator for x2/x3 calculation
        denominator1 = original_matrix[2][0] * original_matrix[0][1] / (current_eigenvalue - original_matrix[0][0]) + original_matrix[2][1];
        if (fabs(denominator1) < 1.0e-12) denominator1 = 1.0e-12;  // Avoid division by zero
        
        // Compute x2/x3 ratio
        x2_over_x3 = -(original_matrix[2][0] * original_matrix[0][2] / (current_eigenvalue - original_matrix[0][0]) + 
                      original_matrix[2][2] - current_eigenvalue) / denominator1;

        // Denominator for x1/x3 calculation  
        denominator2 = (original_matrix[1][0] + (current_eigenvalue - original_matrix[1][1]) * 
                       (original_matrix[0][0] - current_eigenvalue) / original_matrix[0][1]);
        if (fabs(denominator2) < 1.0e-12) denominator2 = 1.0e-12;  // Avoid division by zero
        
        // Compute x1/x3 ratio
        x1_over_x3 = -(original_matrix[1][2] + (current_eigenvalue - original_matrix[1][1]) * 
                      original_matrix[0][2] / original_matrix[0][1]) / denominator2;

        // STEP 11: Normalize eigenvector to unit length
        // Compute x3 component from normalization condition: x1² + x2² + x3² = 1
        eigenvectors[2][i] = -sqrt(1.0 / (x1_over_x3 * x1_over_x3 + x2_over_x3 * x2_over_x3 + 1.0));
        
        // Compute x2 and x1 components using the ratios
        eigenvectors[1][i] = x2_over_x3 * eigenvectors[2][i];
        eigenvectors[0][i] = x1_over_x3 * eigenvectors[2][i];
    }
}

/**
 * Computes the liquid crystal order parameters from molecular orientations
 * 
 * This function constructs the Q-tensor from molecular orientation vectors,
 * diagonalizes it to find the principal directions and degree of alignment,
 * and returns the order parameter (P2) and director vector.
 * 
 * The Q-tensor is defined as: Q = ⟨u⊗u⟩ - (1/3)I, where u are unit orientation
 * vectors, ⟨⟩ denotes ensemble average, and I is the identity matrix.
 * 
 * @param num_particles         Number of particles in the system
 * @param orientation_x         Array of x-components of orientation vectors (unit vectors)
 * @param orientation_y         Array of y-components of orientation vectors  
 * @param orientation_z         Array of z-components of orientation vectors
 * @param director_x            OUTPUT: x-component of the director (principal orientation)
 * @param director_y            OUTPUT: y-component of the director
 * @param director_z            OUTPUT: z-component of the director
 * @param order_parameter       OUTPUT: Scalar order parameter P2 (0=isotropic, 1=perfect alignment)
 */
void compute_liquid_crystal_order_parameters(int num_particles, 
                                            double *orientation_x, 
                                            double *orientation_y, 
                                            double *orientation_z, 
                                            float *director_x, 
                                            float *director_y, 
                                            float *director_z, 
                                            float *order_parameter)
{
    int i, largest_eigenvalue_index;
    
    // Q-tensor components (symmetric 3x3 matrix)
    float q_xx, q_xy, q_xz, q_yy, q_yz, q_zz;
    
    // Full Q-tensor and its eigenvalues/vectors
    float q_tensor[3][3];
    float eigenvectors[3][3];
    float eigenvalues[3];
    
    // Precompute inverse of particle count for averaging
    const float inverse_particle_count = 1.0 / ((double)num_particles);;

    // Initialize Q-tensor components to zero
    q_xx = 0.0;
    q_xy = 0.0;
    q_xz = 0.0;
    q_yy = 0.0;
    q_yz = 0.0;
    q_zz = 0.0;

    // STEP 1: Construct the raw second moment tensor ⟨u⊗u⟩
    // Sum over all particles to compute ensemble averages
    for (i = 0; i < num_particles; i++) {
        // u_x * u_x component - measures alignment along x-axis
        q_xx += orientation_x[i] * orientation_x[i];
        
        // u_x * u_y component - measures correlation between x and y alignment  
        q_xy += orientation_x[i] * orientation_y[i];
        
        // u_x * u_z component - measures correlation between x and z alignment
        q_xz += orientation_x[i] * orientation_z[i];
        
        // u_y * u_y component - measures alignment along y-axis
        q_yy += orientation_y[i] * orientation_y[i];
        
        // u_y * u_z component - measures correlation between y and z alignment
        q_yz += orientation_y[i] * orientation_z[i];
        
        // u_z * u_z component - measures alignment along z-axis
        q_zz += orientation_z[i] * orientation_z[i];
    }

    // STEP 2: Construct the traceless Q-tensor Q = ⟨u⊗u⟩ - (1/3)I
    // Diagonal elements: subtract 1/3 to make tensor traceless
    // This ensures Q = 0 for isotropic (random) orientation distribution
    
    // Q_xx = ⟨u_x²⟩ - 1/3
    q_tensor[0][0] = q_xx * inverse_particle_count - tercio;
    
    // Q_yy = ⟨u_y²⟩ - 1/3  
    q_tensor[1][1] = q_yy * inverse_particle_count - tercio;
    
    // Q_zz = ⟨u_z²⟩ - 1/3
    q_tensor[2][2] = q_zz * inverse_particle_count - tercio;
    
    // Off-diagonal elements (no subtraction needed)
    // Q_xy = ⟨u_x u_y⟩
    q_tensor[0][1] = q_xy * inverse_particle_count;
    
    // Q_xz = ⟨u_x u_z⟩
    q_tensor[0][2] = q_xz * inverse_particle_count;
    
    // Q_yz = ⟨u_y u_z⟩
    q_tensor[1][2] = q_yz * inverse_particle_count;
    
    // Enforce symmetry: Q is a symmetric tensor
    q_tensor[1][0] = q_tensor[0][1];  // Q_yx = Q_xy
    q_tensor[2][0] = q_tensor[0][2];  // Q_zx = Q_xz  
    q_tensor[2][1] = q_tensor[1][2];  // Q_zy = Q_yz

    // STEP 3: Diagonalize the Q-tensor to find principal axes
    // This finds the coordinate system where the tensor is diagonal
    // The eigenvectors represent the principal directions of alignment
    // The eigenvalues represent the degree of alignment along each principal axis
    diagonalize_symmetric_matrix(q_tensor, eigenvectors, eigenvalues);

    // STEP 4: Find the principal direction (director)
    // The largest eigenvalue corresponds to the direction of maximum alignment
    largest_eigenvalue_index = find_largest_eigenvalue_index(eigenvalues);

    // STEP 5: Compute the scalar order parameter P2
    // P2 = (3/2) * λ_max, where λ_max is the largest eigenvalue
    // Range: 0 (completely random) to 1 (perfectly aligned)
    *order_parameter = 1.5 * eigenvalues[largest_eigenvalue_index];

    // STEP 6: Extract the director vector
    // The eigenvector corresponding to the largest eigenvalue gives the average orientation
    // We take the negative to follow convention (director is defined up to a sign)
    *director_x = -eigenvectors[0][largest_eigenvalue_index];
    *director_y = -eigenvectors[1][largest_eigenvalue_index]; 
    *director_z = -eigenvectors[2][largest_eigenvalue_index];
}




//*********************************************************************************************
//**********   Calcula constantes para utilizar en potencial entre dos partículas   ***********
void   update_running_averages(int opcion, int nump, float *n_ave, float *ave_ene, float *ave_dens, float ave_pressTns[4], 
		float *ave_ordpar, float energia, float dens, float press_tns[4], 
		float ord_par)
{
	int i;

	switch ( opcion )
	{
		case 0:
			*n_ave    = 0.0;
			*ave_ene  = 0.0;
			*ave_dens = 0.0;
			for (i = 0; i < 4; i++) ave_pressTns[i] = 0.0;
			*ave_ordpar = 0.0;

			break;
		case 1:
			*n_ave    += 1.0;
			*ave_ene  += energia;
			*ave_dens += dens;
			for (i = 0; i < 4; i++) ave_pressTns[i] += press_tns[i];
			*ave_ordpar += ord_par;

			break;
		case 2:
			*ave_ene    /= (*n_ave);
			*ave_dens   /= (*n_ave);
			for (i = 0; i < 4; i++) ave_pressTns[i] /= (*n_ave);
			*ave_ordpar /= (*n_ave);

			break;
		default:
			printf("Opción de función cálculo de promedios no reconocida (tp_arre = %d)\n", opcion);
			break;
	}
}
//*********************************************************************************************
//**********   Calcula constantes para utilizar en potencial entre dos partículas   ***********
void jacobi(int np, float a[np][np], float d[np], float v[np][np])
{
  int    i, j, k, ip, iq, n, maxrot, nrot;
  float  sm, treshold, ss, c, t, theta, tau, h, gg, p;
  float  b[np], z[np];

  maxrot = 100;
  nrot   = 0;
  n      = np;

  for (ip = 0; ip < n; ip++)
  {
     for (iq=0; iq<n; iq++)
     {
	 v[ip][iq] = 0.0;  // barro todos los elementos
     }
     v[ip][ip] = 1.0;      // hago 1 a los elementos de la diagonal
  }

  for (ip = 0; ip < n; ip++)
  {
     b[ip] = a[ip][ip];
     d[ip] = b[ip];
     z[ip] = 0.0;
  }

// Rotaciones de Jacobi
  
  for (i = 0; i < maxrot; i++)
  {

      sm = 0.0;
      for (ip = 0; ip < n-1; ip++)
      {
	  for (iq = ip+1; iq < n; iq++)
	  {
	      sm += fabs(a[ip][iq]);
	  }
      }

      if (sm == 0){
	 break; }  //  Sale del ciclo

      if (i < 4){
	 treshold = 0.2*sm/((float)(n*n)); }
      else{ 
	 treshold = 0.0; }

      for (ip = 0; ip < n-1; ip++)
      {
	  for (iq = ip+1; iq < n; iq++)
	  {
	      gg = 100.0*fabs(a[ip][iq]);

	      if ((i > 4) && (gg + fabs(d[ip]) == fabs(d[ip])) && (gg + fabs(d[iq]) == fabs(d[iq])))
	      {
		 a[ip][iq] = 0.0;
	      }
	      else if (fabs(a[ip][iq]) > treshold)
	      {
		 h = d[iq] - d[ip];

		 if (gg + fabs(h) == fabs(h))
		 {
		    t = a[ip][iq] / h;
		 }
                 else
		 {
	            theta = 0.5*h / a[ip][iq];
		    t     = 1.0/(fabs(theta) + sqrt(1.0 + theta*theta));
		    if (theta < 0.0) t = -t;
		 }

		 c   = 1.0/sqrt(1.0 + t*t);
		 ss  = t*c;
		 tau = ss/(1.0 + c);
		 h   = t*a[ip][iq];
		 z[ip] -= h;
		 z[iq] += h;
		 d[ip] -= h;
		 d[iq] += h;
		 a[ip][iq] = 0.0;

		 for (j = 0; j < ip-1; j++)
		 {
		     gg = a[j][ip];
		     h  = a[j][iq];
		     a[j][ip] = gg - ss*(h  + gg*tau);
		     a[j][iq] = h  + ss*(gg - h*tau);
		 }
		 for (j = ip+1; j < iq - 1; j++)
		 {
	             gg = a[ip][j];
		     h  = a[j][iq];
                     a[ip][j] = gg - ss*(h  + gg*tau);
		     a[j][iq] = h  + ss*(gg - h*tau);
		 }
		 for (j = iq+1; j < n; j++)
		 {
		     gg = a[ip][j];
		     h  = a[iq][j];
		     a[ip][j] = gg - ss*(h  + gg*tau);
                     a[iq][j] = h  + ss*(gg - h*tau);
		 }
		 for (j = 0; j < n; j++)
		 {
	             gg = v[j][ip];
		     h  = v[j][iq];
		     v[j][ip] = gg - ss*(h  + gg*tau);
		     v[j][iq] = h  + ss*(gg - h*tau);
		 }

		 nrot += 1;
	      }
	  }
      }

      for (ip = 0; ip < n; ip++)
      {
	  b[ip] += z[ip];
	  d[ip]  = b[ip];
	  z[ip]  = 0.0;
      }
  } 

  if (nrot == maxrot-1) printf("Diagonalización de matriz no convergió (Jacobi)\n");

  for (i = 0; i < n - 1; i++)
  {
      k = i;
      p = d[i];

      for (j = i+1; j < n; j++)
      {
	  if (d[j] < p)
	  {
             k = j;
	     p = d[j];
	  }
      }

      if (k != i)
      {
	 d[k] = d[i];
	 d[i] = p;

	 for (j = 0; j < n; j++)
	 {
	     p = v[j][i];
	     v[j][i] = v[j][k];
	     v[j][k] = p;
	 }
      }
  }
}
//*********************************************************************************************
//**********   Implements periodic boundary conditions                     ********************
void apply_pbc(double *x, double *y, double *z)
{
    if (*x > box_size[0])     *x -= box_size[0];
    else if (*x < 0.0) *x += box_size[0];

    if (*y > box_size[1])     *y -= box_size[1];
    else if (*y < 0.0) *y += box_size[1];

    if (*z > box_size[2])     *z -= box_size[2];
    else if (*z < 0.0) *z += box_size[2];
}

//*********************************************************************************************
//*****   Performs a single Monte Carlo move attempt for a randomly selected particle   *******
/**
 * // IN/OUT: Particle positions (x, y, z coordinates)
 * // IN/OUT: Particle orientation vectors
 * // INPUT: Maximum displacement parameter
 * // IN/OUT: Total system system_energy (updated if move accepted)
 * // INPUT: Neighbor neighbor_list_array arrays
 * // IN/OUT: Counter for accepted moves
 * // IN/OUT: Maximum displacement tracker for neighbor_list_array updates
 * // IN/OUT: Histogram of which particles are moved
 */

void attempt_monte_carlo_move(
                              double *position_x, double *position_y, double *position_z,
                              double *orient_x, double *orient_y, double *orient_z,
                              float max_displacement,
                              double *system_energy,
                              int *neighbor_count_array, int *neighbor_list_array,
                              long *accepted_moves,
                              double *desp_max,
                              size_t *histograma
                             ) 
{

    /* local vars */
    int selected_particle;
    int displacement_index;                                             // Index for displacement magnitude
    int rotation_index;                                                 // Index for rotation vector
    double new_position_x, new_position_y, new_position_z;              // NEW trial position coordinates
    double new_orient_x = 0.0, new_orient_y = 0.0, new_orient_z = 0.0;  // NEW trial orientation vector
    
    double old_energy;                // Energy at current (old) position/orientation
    double new_energy;                // Energy at trial (new) position/orientation  
    double energy_difference;         // Energy difference: ΔE = E_new - E_old
    double rand_acceptance;           // Random number for Metropolis acceptance test

    double max_displacement_squared;  // Maximum component of rotation vector (for neighbor_list_array updates)
    double r2, dx, dy, dz;            // Distance squared and components to colloid center
    double displacement_range;        // Random displacement component (-max_displacement/2 to +max_displacement/2)
    const double NORM_EPS = 1e-12;    // Tolerance for orientation vector normalization (1e-12)

    /* 1) pick particle uniformly */
    selected_particle = (int)(rand() * I_MAX_RAND * num_particles);
    /* histogram update (not thread-safe — ensure single-threaded or use atomic) */
    histograma[selected_particle] += 1;

    /* 2) choose random rotation/translation sample indices - use randombytes_uniform to avoid rounding issues */
    if (POINTS_ON_SPHERE <= 0) {
        /* fallback to a single-sample trivial rotation */
        displacement_index = 0; rotation_index = 0;
    } else {
        // displacement_index = (int) randombytes_uniform((unsigned int)POINTS_ON_SPHERE);
        // rotation_index = (int) randombytes_uniform((unsigned int)POINTS_ON_SPHERE);
        displacement_index = (int)(rand() * I_MAX_RAND * POINTS_ON_SPHERE);
        rotation_index = (int)(rand() * I_MAX_RAND * POINTS_ON_SPHERE);
    }

    /* 3) compute old system_energy contributions (LC-LC + LC-colloid) */
    old_energy  = GBij_lstnb(selected_particle, 
                             position_x, position_y, position_z, 
                             orient_x, orient_y, orient_z, 
                             neighbor_count_array, neighbor_list_array);
    old_energy += GBCollRod(selected_particle, 
                            (double [3]) {position_x[selected_particle], position_y[selected_particle], position_z[selected_particle]}, 
                            (double [3]) {orient_x[selected_particle], orient_y[selected_particle], orient_z[selected_particle]}, 
                            colloid_coords);

    /* 4) propose new position (uniform within [-max_displacement/2, max_displacement/2] per component) */
    /* use rand() here only for doubles in (-0.5,0.5); you may replace with a high-quality double RNG if desired */
    displacement_range = max_displacement * ( (double)rand() * I_MAX_RAND - 0.5 );
    new_position_x = position_x[selected_particle] + displacement_range;
    
    displacement_range = max_displacement * ( (double)rand() * I_MAX_RAND - 0.5 );
    new_position_y = position_y[selected_particle] + displacement_range;
    
    displacement_range = max_displacement * ( (double)rand() * I_MAX_RAND - 0.5 );
    new_position_z = position_z[selected_particle] + displacement_range;

    /* wrap into box */
    apply_pbc(&new_position_x, &new_position_y, &new_position_z);

    // Compute teh squared distance between the new position and the center of the box
    // where it's placed the spherical colloid
    r2 = compute_distance_sq(new_position_x, new_position_y, new_position_z, 
                             0.5*box_size[0], 0.5*box_size[1], 0.5*box_size[2], 
                             &dx, &dy, &dz);

    if (r2 > colloid_radius*colloid_radius) {  // If new position is outside the colloid, proceed

        /* 5) propose new orientation: add small rotation vector from sample and renormalize */
        /* check bounds for rotation_index */
        if (POINTS_ON_SPHERE > 0) {
            /* rot_vec_x/rot_vec_y/rot_vec_z should be in [-1,1] type or similar */
            new_orient_x = rot_vec_x[rotation_index] * max_rotation + orient_x[selected_particle];
            new_orient_y = rot_vec_y[rotation_index] * max_rotation + orient_y[selected_particle];
            new_orient_z = rot_vec_z[rotation_index] * max_rotation + orient_z[selected_particle];
        } else {
            new_orient_x = orient_x[selected_particle];
            new_orient_y = orient_y[selected_particle];
            new_orient_z = orient_z[selected_particle];
        }

        /* normalize new orientation robustly */
        {
            double norm2 = new_orient_x*new_orient_x + new_orient_y*new_orient_y + new_orient_z*new_orient_z;

            // If the vector is too small (degenerate), keep the old orientation.
            if (norm2 <= NORM_EPS) {
                /* fallback: keep old orientation (or pick a random orientation) 
                 * This prevents division by zero */
                new_orient_x = orient_x[selected_particle]; 
                new_orient_y = orient_y[selected_particle]; 
                new_orient_z = orient_z[selected_particle];
            } else {
                new_orient_x /= sqrt(norm2); 
                new_orient_y /= sqrt(norm2); 
                new_orient_z /= sqrt(norm2);
            }
        }

        /* 6) compute new system_energy for moved particle (LC-LC and LC-colloid) */
        new_energy  = GBij_PosMod(selected_particle, 
                                  (double [3]) {new_position_x, new_position_y, new_position_z}, 
                                  (double [3]) {new_orient_x, new_orient_y, new_orient_z}, 
                                  position_x, position_y, position_z, 
                                  orient_x, orient_y, orient_z);

        new_energy += GBCollRod(selected_particle, 
                                (double [3]) {new_position_x, new_position_y, new_position_z}, 
                                (double [3]) {new_orient_x, new_orient_y, new_orient_z}, 
                                colloid_coords);

        /* 7) acceptance test (log-space to avoid overflow) */
        energy_difference = new_energy - old_energy;
        if (energy_difference <= 0.0) {
            /* always accept energetically favorable moves */
            rand_acceptance = 0.0;
        } else {
            rand_acceptance = (double)rand() * I_MAX_RAND; /* uniform (0,1) */
        }

        if (energy_difference <= 0.0 || log(rand_acceptance) <= (-gb_beta * energy_difference)) {
            /* ACCEPT move */
            position_x[selected_particle] = new_position_x; 
            position_y[selected_particle] = new_position_y; 
            position_z[selected_particle] = new_position_z;

            orient_x[selected_particle] = new_orient_x; 
            orient_y[selected_particle] = new_orient_y; 
            orient_z[selected_particle] = new_orient_z;

            /* update displacement tracker: use actual rotation sample magnitude */
            if (POINTS_ON_SPHERE > 0) {
                /* max_displacement_squared := max(|rot_vec_x[displacement_index]|,|rot_vec_y[displacement_index]|,|rot_vec_z[displacement_index]|) */
                double a = fabs(rot_vec_x[displacement_index]);
                double b = fabs(rot_vec_y[displacement_index]);
                double c = fabs(rot_vec_z[displacement_index]);

                max_displacement_squared = a > b ? a : b;
                max_displacement_squared = max_displacement_squared > c ? max_displacement_squared : c;

                *desp_max += 3.0 * max_displacement_squared * max_displacement_squared;
            }

            *system_energy += energy_difference;
            *accepted_moves += 1;

        } else {
            /* REJECT move: do nothing, everything remains unchanged */
        }
    }
}
//*********************************************************************************************
//**********   Calcula constantes para utilizar en potencial entre dos partículas   ***********
static inline double g_omega(double eiej, double amasb2, double amenb2, double omega){
   double ome_eiej, gmas, gmen;

   ome_eiej = omega*eiej;
   gmas     = amasb2/(1.0 + ome_eiej);
   gmen     = amenb2/(1.0 - ome_eiej);

   return 1.0 - 0.50*omega*(gmas + gmen);
}
//*********************************************************************************************
//*************************   Potencial de GB entre dos partículas    *************************
double pij_GB(double ei[3],    // INPUT: Orientation vector of particle i (unit vector)
              double ej[3],    // INPUT: Orientation vector of particle j (unit vector) 
              double rij,      // INPUT: Distance between particle centers
              double irij,     // INPUT: Inverse distance (1/rij)
              double iei,      // INPUT: Inverse norm of ei orientation vector
              double iej,      // INPUT: Inverse norm of ej orientation vector  
              double eiej,     // INPUT: Dot product ei·ej (cosine of angle between orientations)
              double dr[3]     // INPUT: Vector from particle i to j (dx, dy, dz)
              ) {
  
    // Orientation-Dependent Geometry
    double eirij;        // Projection: (ei · r_ij)/|r_ij| (cosine of angle between ei and separation vector)
    double ejrij;        // Projection: (ej · r_ij)/|r_ij| (cosine of angle between ej and separation vector)
    double amasb;        // Sum: eirij + ejrij
    double amenb;        // Difference: eirij - ejrij  
    double amab2;        // Squared sum: (eirij + ejrij)²
    double ameb2;        // Squared difference: (eirij - ejrij)²
    
    // Energy Scaling Factors
    double eps1;         // Energy scale factor 1: 1/√(1 - χ²(ei·ej)²)
    double ctep1;        // eps1 raised to ν power: eps1^ν
    double eps2;         // Energy scale factor 2 from g_omega function
    double ctep2;        // eps2 raised to μ power: eps2^μ
    double s_chi;        // Shape function from g_omega (range parameter)

    // Distance calculations
    double sigma;        // Orientation-dependent contact distance
    double distreal;     // Effective distance: rij - σ + σ_ss
    double erre;         // σ_ss / distreal (equivalent to σ/r in LJ potential)
    double erre2;        // erre squared
    double erre6;        // erre^6 (for LJ 12-6 potential)
    double epsil;        // Combined system_energy scaling: ctep1 * ctep2
    double resultado;    // Final potential system_energy value

    
    eirij  = irij*iei*compute_dot_product(dr[0], dr[1], dr[2], ei[0], ei[1], ei[2]);
    ejrij  = irij*iej*compute_dot_product(dr[0], dr[1], dr[2], ej[0], ej[1], ej[2]);

    amasb  = eirij + ejrij;
    amenb  = eirij - ejrij;
    amab2  = amasb*amasb;
    ameb2  = amenb*amenb;

    eps1   = 1.0/sqrt(1.0 - gb_chi_sq*eiej*eiej);
    ctep1  = pow(eps1, gb_nu_exponent);
    eps2   = g_omega(eiej, amab2, ameb2, gb_chi_prime);
    ctep2  = pow(eps2, gb_mu_exponent);
    s_chi  = g_omega(eiej, amab2, ameb2, gb_chi);

    sigma  = gb_sigma_sidetoside/sqrt(s_chi);
    distreal = (rij - sigma + gb_sigma_sidetoside);

    if (distreal < 0.001) resultado = 1000000000; // Para no desbordar número doble
    
    else {
        erre      = gb_sigma_sidetoside/distreal;
        erre2     = erre*erre;
        erre6     = erre2*erre2*erre2;
        epsil     = ctep1*ctep2;
        resultado = gb_4eps0*epsil*erre6*(erre6 - 1.0);  // Potencial de Gay-Berne
    }

    return resultado;
}

//*********************************************************************************************
//***********************       Producto punto entre dos vectores             *****************
static inline double compute_dot_product(double x1, double y1, double z1, double x2, double y2, double z2) {
  return x1*x2 + y1*y2 + z1*z2;
}

//*********************************************************************************************
static inline double compute_distance_sq(double x1, double y1, double z1,
                                         double x2, double y2, double z2,
                                         double *dx, double *dy, double *dz) 
{
    
    double x_sq, y_sq, z_sq, rij2;

    *dx = x2 - x1;
    *dy = y2 - y1;
    *dz = z2 - z1;

    // Minimum image using global boxes box_size[0], box_size[1], box_size[2]
    if (box_size[0] > 0.0) {
        if (*dx >  0.5*box_size[0]) *dx -= box_size[0];
        else if (*dx < -0.5*box_size[0]) *dx += box_size[0];
    }
    if (box_size[1] > 0.0) {
        if (*dy >  0.5*box_size[1]) *dy -= box_size[1];
        else if (*dy < -0.5*box_size[1]) *dy += box_size[1];
    }
    if (box_size[2] > 0.0) {
        if (*dz >  0.5*box_size[2]) *dz -= box_size[2];
        else if (*dz < -0.5*box_size[2]) *dz += box_size[2];
    }

    x_sq = (*dx)*(*dx);
    y_sq = (*dy)*(*dy);
    z_sq = (*dz)*(*dz);
    rij2 = x_sq + y_sq + z_sq;

    return rij2;
}

//*********************************************************************************************
//******* Estima el número de vecinos de cada partícula para poder generar las listas  ********
void estima_vecinos(int *max_neighbs, int *otra_estimacion, double *diff_SqrRadi, 
		            int *NumPart_inByt, int *neig_inByt, float *factor_lista) {
 
    // Maximum neighbors per particle (estimated)
    *max_neighbs  = (int)(1 + 2.0*(system_density + 1.0) * verlet_volume);

    // Alternative estimation method
    *otra_estimacion = (int)(1 + 2.4 * verlet_volume);

    // Squared difference between Verlet and cutoff radii
    *diff_SqrRadi    = (verlet_radius - cutoff_radius)*(verlet_radius - cutoff_radius);

    // Choose which, max_neighbs or otra_estimacion, is bigger and take it
    *max_neighbs   = ((*max_neighbs) >= (*otra_estimacion) ? (*max_neighbs) : (*otra_estimacion));

    // Memory for neighbor neighbor_list_array data (bytes)
    *NumPart_inByt   = num_particles*sizeof(int);

    // Factor for neighbor neighbor_list_array update frequency
    *neig_inByt      = (*max_neighbs)*num_particles*sizeof(int);

    if (num_particles > 100) *factor_lista = num_particles*0.01; // Supongo que en cada movimiento de MC es probable que mueva la misma partícula

    return;
}
//*********************************************************************************************
//****************************  Calcula constantes para utilizar en la sim  *******************
void get_system_const(void) {

    // Thermodynamic and Simulation Parameters
    gb_beta   = 1.0/system_temperature;                         // Inverse temperature: β = 1/(k_B*T)
    gb_4eps0  = 4.0*gb_epsilon0;                                // 4*epsilon_0 (system_energy scale factor for GB potential)

    // Cutoff and Neighbor List Radii
    cutoff_radius_sq = cutoff_radius * cutoff_radius;           // Squared cutoff radius for interactions
    r_verlet_sq      = verlet_radius * verlet_radius;           // Squared Verlet radius for neighbor lists
    verlet_volume    = (4./3.) * pi * pow(verlet_radius, 3);    // Volume of a sphere using verlet radius
    
    // Gay-Berne Shape Parameters
    gb_k_imu   = pow(gb_kappa_prime, 1.0/gb_mu_exponent);       // κ' raised to 1/μ power

    // Molecular Inertia and Dynamics
    gb_inercia = 0.05 * gb_sigma_endtoend*gb_sigma_endtoend 
                      * (1.0 + gb_kappa_sq);                    // Moment of inertia for rod-like particl
    msad       = 2.0 * system_temperature / gb_inercia;         // Mean squared angular displacement parameter

    // Gay-Berne Anisotropy Parameters
    gb_chi       = (gb_kappa_sq - 1.0) / (gb_kappa_sq + 1.0);   // Shape anisotropy parameter χ
    gb_kappa_sq  = gb_kappa*gb_kappa;                           // Squared aspect ratio κ² = (σ_ee/σ_ss)²    
    gb_chi_sq    =  gb_chi * gb_chi;                            // Squared shape anisotropy χ²
    gb_chi_prime = (gb_k_imu - 1.0) / (gb_k_imu + 1.0);         // Energy anisotropy parameter χ'
    
    colloid_rod_cutoff = 4.0 * gb_sigma_endtoend;               // Cutoff distance for colloid-rod interactions

    // Inner and outer sphere surface
    colloid_radius_sq = colloid_radius*colloid_radius;          // Squared colloid radius

    return;
}
//*********************************************************************************************
//**********************  Posiciones iniciales obtenidas de fortSphere.1    *******************
void read_initial_configuration(double *position_x, double *position_y, double *position_z, double *orient_x, double *orient_y, double *orient_z) {
    int    n_tmp, i, nval = 0;
    double basr;
    FILE   *f_fort;

    f_fort = fopen ("fortSphere.1","r");

    if ( fscanf(f_fort, "%d %lf %lf %lf %lf", &n_tmp, &box_size[0], &box_size[1], &box_size[2], &colloid_radius) == 5) {
        nval += 5;
    }

    // ==============================================================================================================
    // Calculate some constants
    colloid_volume = (4./3.) * pi * pow(colloid_radius, 3);              // Volume of the colloidal sphere particle
    box_volume     = box_size[0] * box_size[1] * box_size[2];            // Total simulation box volume
    system_density = num_particles / (box_volume - colloid_volume);      // Number density inside the box, outside the colloid
    colloid_coords[0] = 0.5 * box_size[0];                                       // x coordinate of the center of the colloid
    colloid_coords[1] = 0.5 * box_size[1];                                       // y coordinate of the center of the colloid
    colloid_coords[2] = 0.5 * box_size[2];                                       // z coordinate of the center of the colloid
    // ==============================================================================================================

    printf("================================================\n");
    printf("Box size    = %3.5lf x %3.5lf x %3.5lf\n", box_size[0], box_size[1], box_size[2]);
    printf("Volume      = %.5lf\n", box_volume);
    printf("Density (ρ) = %.5lf \n", system_density);
    printf("================================================\n\n");

    if (n_tmp >= num_particles)
    {
        for (i=0; i<num_particles; i++)
        {
            if (fscanf (f_fort, "%lf %lf %lf", &position_x[i], &position_y[i], &position_z[i]) == 3) nval += 3;
            if (fscanf (f_fort, "%lf %lf %lf", &basr,  &basr,  &basr) == 3) nval += 3;
            if (fscanf (f_fort, "%lf %lf %lf", &orient_x[i], &orient_y[i], &orient_z[i]) == 3) nval += 3;
            if (fscanf (f_fort, "%lf %lf %lf", &basr,  &basr,  &basr) == 3) nval += 3;
        }
    }
    else
    {
        printf("Configuración de partículas en fortSphere.1 no son suficientes (N_fort.1 = %d y N_run = %d \n", n_tmp, num_particles);
    }

    if (nval != 12*num_particles + 5) printf ("Error en la lectura de archivo fortSphere.1 \n");

    fclose(f_fort);
}

//*********************************************************************************************
//***************      Posiciones iniciales al azar para todas las partículas       ***********
void initialize_random_positions(double *position_x, double *position_y, double *position_z, double *orient_x, double *orient_y, double *orient_z) {
    
    int    i, j, traslape;
    float  xi, yi, zi;
    double dx, dy, dz, r2;

    // ==============================================================================================================
    // Calculate some constants
    colloid_volume = (4./3.) * pi * pow(colloid_radius, 3);              // Volume of the colloidal sphere particle
    box_volume     = (num_particles / system_density) + colloid_volume;  // Volume of the simulation box
    
    box_size[0] = pow(box_volume, 1.0/3.0);                              // x length of the simulation box
    box_size[1] = box_size[0];                                           // y length of the simulation box
    box_size[2] = box_size[0];                                           // z length of the simulation box

    colloid_coords[0] = 0.5 * box_size[0];                                       // x coordinate of the center of the colloid
    colloid_coords[1] = 0.5 * box_size[1];                                       // y coordinate of the center of the colloid
    colloid_coords[2] = 0.5 * box_size[2];                                       // z coordinate of the center of the colloid
    // ==============================================================================================================

    printf("================================================\n");
    printf("Box size    = %3.5lf x %3.5lf x %3.5lf\n", box_size[0], box_size[1], box_size[2]);
    printf("Volume      = %.5lf\n", box_volume);
    printf("Density (ρ) = %.5lf \n", system_density);
    printf("================================================\n\n");

    for (i = 0; i < num_particles; i++)
    {
        do
        {
            position_x[i] = rand()*box_size[0]*I_MAX_RAND;  
            position_y[i] = rand()*box_size[1]*I_MAX_RAND;  
            position_z[i] = rand()*box_size[2]*I_MAX_RAND;  

            pto_ransphr(&xi, &yi, &zi, 1.0);
            
            orient_x[i] = xi; 
            orient_y[i] = yi; 
            orient_z[i] = zi;

            traslape = 0;

            for ( j = 0; j < i; j++ ) 
            {   
                r2 = compute_distance_sq(position_x[i], position_y[i], position_z[i], 
                                         position_x[j], position_y[j], position_z[j], 
                                         &dx, &dy, &dz);

                if (r2 < gb_sigma_endtoend*gb_sigma_endtoend)  
                {
                    traslape += 1;
                    break;
                }
            }

            r2 = compute_distance_sq(position_x[i], position_y[i], position_z[i], 
                                     0.5*box_size[0], 0.5*box_size[1], 0.5*box_size[2], 
                                     &dx, &dy, &dz);

            if (r2 < colloid_radius*colloid_radius) traslape += 1;     // colloid_radius_sq not yet defined

        } while ( traslape != 0 );
        
        if ((i+1)%500 == 0) {
            printf("\rArranging particles: %d %% ", 100*num_particles/(i+1));
            fflush(stdout);
        }
    }
    printf("\n\n");
    
    // Check if all particles are outside the inner sphere and inside the box
    for (i = 0; i < num_particles; i++)
    {
        r2 = compute_distance_sq(position_x[i], position_y[i], position_z[i], 
                                 0.5*box_size[0], 0.5*box_size[1], 0.5*box_size[2], 
                                 &dx, &dy, &dz);

        if (r2 < colloid_radius*colloid_radius) printf("Error: Partícula %d dentro de la esfera interna\n", i);
        
        if (position_x[i] < 0.0 || position_x[i] > box_size[0]) printf("Error: Partícula %d fuera del box en x. \n", i);
        if (position_y[i] < 0.0 || position_y[i] > box_size[1]) printf("Error: Partícula %d fuera del box en y. \n", i);
        if (position_z[i] < 0.0 || position_z[i] > box_size[2]) printf("Error: Partícula %d fuera del box en z. \n", i);
    }

    return;
}

//*********************************************************************************************
//***************      Posiciones iniciales en cubo para todas las partículas       ***********
void initialize_cubic_lattice_positions(double *position_x, double *position_y, double *position_z, double *orient_x, double *orient_y, double *orient_z){
    
    int    i, n_s, iix, iiy, iiz;
    float  exi, eyi, ezi;
    double deltax, deltay, deltaz;
    double dx, dy, dz, r2;
    double tempboxx, tempboxy, tempboxz;

    // ==============================================================================================================
    // Calculate some constants
    colloid_volume = (4./3.) * pi * pow(colloid_radius, 3);              // Volume of the colloidal sphere particle
    box_volume     = (num_particles / system_density) + colloid_volume;  // Volume of the simulation box
    
    box_size[0] = pow(box_volume, 1.0/3.0);                              // x length of the simulation box
    box_size[1] = box_size[0];                                           // y length of the simulation box
    box_size[2] = box_size[0];                                           // z length of the simulation box

    colloid_coords[0] = 0.5 * box_size[0];                                       // x coordinate of the center of the colloid
    colloid_coords[1] = 0.5 * box_size[1];                                       // y coordinate of the center of the colloid
    colloid_coords[2] = 0.5 * box_size[2];                                       // z coordinate of the center of the colloid
    // ==============================================================================================================

    printf("================================================\n");
    printf("Box size    = %3.5lf x %3.5lf x %3.5lf\n", box_size[0], box_size[1], box_size[2]);
    printf("Volume      = %.5lf\n", box_volume);
    printf("Density (ρ) = %.5lf \n", system_density);
    printf("================================================\n\n");

    n_s  = 2;

    while ((n_s*n_s*n_s) < num_particles) n_s++;
    
    deltax = box_size[0]/((double)n_s);
    if (deltax < 1.1*gb_sigma_endtoend) deltax = gb_sigma_endtoend;   // Minimum distance between particles
    
    deltay = box_size[1]/((double)n_s);
    if (deltay < 1.1*gb_sigma_endtoend) deltay = gb_sigma_endtoend;
    
    deltaz = box_size[2]/((double)n_s);
    if (deltaz < 1.1*gb_sigma_endtoend) deltaz = gb_sigma_endtoend;

    iix=iiy=iiz=0;
    for (i=0; i<num_particles; i++) 
    {
        r2 = 0.0;

        while (r2 < colloid_radius*colloid_radius) // while inside the inner sphere, move to the next position
        {
            tempboxx = (double)(iix)*deltax;
            tempboxy = (double)(iiy)*deltay;
            tempboxz = (double)(iiz)*deltaz;

            r2 = compute_distance_sq(tempboxx, tempboxy, tempboxz, 
                                     colloid_coords[0], colloid_coords[1], colloid_coords[2], 
                                     &dx, &dy, &dz);

            if (r2 <= colloid_radius*colloid_radius)
            {
                ++iix;
                if (iix == n_s ) 
                { 
                    iix = 0;
                    ++iiy;
                
                    if (iiy == n_s ) 
                    {
                        iiy = 0;
                        ++iiz;
                    }
                }
            }
        }

        position_x[i] = (double)(iix)*deltax;
        position_y[i] = (double)(iiy)*deltay;
        position_z[i] = (double)(iiz)*deltaz;

        pto_ransphr(&exi, &eyi, &ezi, 1.0);
        orient_x[i] = exi;
        orient_y[i] = eyi;
        orient_z[i] = ezi;

        ++iix;
        
        if ( iix == n_s ) { 
            iix = 0;
            ++iiy;
        
            if ( iiy == n_s ) {
                iiy = 0;
                ++iiz;
            }
        }
    }

    // Check if all particles are outside the inner sphere and inside the box
    for (i = 0; i < num_particles; i++)
    {
        r2 = compute_distance_sq(position_x[i], position_y[i], position_z[i], 
                                 0.5*box_size[0], 0.5*box_size[1], 0.5*box_size[2], 
                                 &dx, &dy, &dz);

        if (r2 < colloid_radius*colloid_radius) printf("Error: Partícula %d dentro de la esfera interna\n", i);
        if (position_x[i] < 0.0 || position_x[i] > box_size[0]) printf("Error: Partícula %d fuera del box en x\n", i);
        if (position_y[i] < 0.0 || position_y[i] > box_size[1]) printf("Error: Partícula %d fuera del box en y\n", i);
        if (position_z[i] < 0.0 || position_z[i] > box_size[2]) printf("Error: Partícula %d fuera del box en z\n", i);
    }

    return;
}
//*********************************************************************************************
//*** Genera un conjunto de puntos de forma aleatoria dentro de una esfera cierto radio  ******
void Marsaglia4MC(float *arr_x, float *arr_y, float *arr_z, float radio){
	
    const long int HalfNumPoints = POINTS_ON_SPHERE/2;

    for (size_t i = 0; i < HalfNumPoints; i++) {
        pto_ransphr(&rot_vec_x[i], &rot_vec_y[i], &rot_vec_z[i], radio);
        rot_vec_x[i + HalfNumPoints] = -rot_vec_x[i];
        rot_vec_y[i + HalfNumPoints] = -rot_vec_y[i];
        rot_vec_z[i + HalfNumPoints] = -rot_vec_z[i];
    }

}
//*********************************************************************************************
//*************   Genera un punto aleatoria dentro de una esfera cierto radio   ***************
static inline void pto_ransphr(float *x_m, float *y_m, float *z_m, float frac){
  int   contador;
  float x1, x2, x1_cu, x2_cu, sum_cu;

  contador = 0;
  
  while (contador == 0) {
	x1     = 1.0 - rand()*2.0*I_MAX_RAND;
	x2     = 1.0 - rand()*2.0*I_MAX_RAND;
    x1_cu  = x1 * x1;
    x2_cu  = x2 * x2;
    sum_cu = x1_cu + x2_cu;
    
    if ( sum_cu < 1) {
        // Generate the z, y, z points on the unitary sphere surface
        *x_m  = 2 * x1 * sqrt(1 - sum_cu);
        *y_m  = 2 * x2 * sqrt(1 - sum_cu);
        *z_m  = 1 - 2 * sum_cu;

        // Rescale the sphere points to the desired radius
        *x_m *= frac;
        *y_m *= frac;
        *z_m *= frac;

        // Raise the flag to geot out of while loop
        contador += 1;
    }
  }
}
//*********************************************************************************************
//*******************    Escribo trayectoria en archivo Configuraciones   *********************
void print_Trayectoria(double *position_x, double *position_y, double *position_z, double *orient_x, double *orient_y, double *orient_z,
		       int option){
  char file_name[] = "Configuraciones.dat";
  FILE *f_Config;
  switch(option){
	 case 0: // Inicio de escritura
             f_Config = fopen(file_name,"w");   // Inicia escritura por 1ra vez
	     fprintf(f_Config, "%d \n", num_particles);     // Save snapshot
 	     fprintf(f_Config, "%3.5lf %3.5lf %3.5lf \n", box_size[0], box_size[1], box_size[2]);
	     for (int i = 0; i < num_particles; i++){
		  fprintf(f_Config, "%5.7lf \t %5.7lf \t %5.7lf \t %5.7lf \t %5.7lf \t %5.7lf \n",
			  position_x[i], position_y[i], position_z[i], orient_x[i], orient_y[i], orient_z[i]);
	     }
             fclose(f_Config);
             break;
	 case 1: // Escritura de configuración intermedia
             f_Config = fopen(file_name,"a");       // Continúa escritura
	     fprintf(f_Config, "%d \n", num_particles);     // Save snapshot
 	     fprintf(f_Config, "%3.5lf %3.5lf %3.5lf \n", box_size[0], box_size[1], box_size[2]);
	     for (int i = 0; i < num_particles; i++){
		  fprintf(f_Config, "%5.7lf \t %5.7lf \t %5.7lf \t %5.7lf \t %5.7lf \t %5.7lf \n",
			  position_x[i], position_y[i], position_z[i], orient_x[i], orient_y[i], orient_z[i]);
	     }
             fclose(f_Config);
	     break;
  }
}
//*********************************************************************************************
//*******************    Escribo trayectoria en archivo Configuraciones   *********************
void print_Qmga(double *position_x, double *position_y, double *position_z, double *orient_x, double *orient_y, double *orient_z,
		int option){
  char file_name[] = "qmga_mcnvtGB.cnf";
  FILE *f_qmga;
  switch(option){
	 case 0: // Inicio de escritura
	     f_qmga = fopen (file_name,"w");
             fprintf (f_qmga, "%d\n", num_particles);
             fprintf (f_qmga, "%lf\n", box_size[0]);
             fprintf (f_qmga, "%lf\n", box_size[1]);
             fprintf (f_qmga, "%lf\n", box_size[2]);
             fprintf (f_qmga, "%lf %lf\n", 0.0, 0.0);
             for (size_t i = 0; i < num_particles; i++){   // qmga uses a box centered at the origin
                 fprintf (f_qmga, "%15.8lf %15.8lf %15.8lf \
                                   %15.8lf %15.8lf %15.8lf \
                                   %15.8lf %15.8lf %15.8lf \
                                   %15.8lf %15.8lf %15.8lf 0 0\n", 
                                                                   position_x[i] - 0.5*box_size[0], 
                                                                   position_y[i] - 0.5*box_size[1], 
                                                                   position_z[i] - 0.5*box_size[2], 
                                                                   0.0, 0.0, 0.0, 
                                                                   orient_x[i], orient_y[i], orient_z[i], 
                                                                   0.0, 0.0, 0.0);
             }
	     fclose(f_qmga);
             break;
	 case 1: // Escritura de configuración intermedia
	     f_qmga = fopen (file_name,"a");      // Continúa escritura
             fprintf (f_qmga, "%d\n", num_particles);
             fprintf (f_qmga, "%lf\n", box_size[0]);
             fprintf (f_qmga, "%lf\n", box_size[1]);
             fprintf (f_qmga, "%lf\n", box_size[2]);
             fprintf (f_qmga, "%lf %lf\n", 0.0, 0.0);
             for (size_t i = 0; i < num_particles; i++){
                 fprintf (f_qmga, "%15.8lf %15.8lf %15.8lf \
                                   %15.8lf %15.8lf %15.8lf \
                                   %15.8lf %15.8lf %15.8lf \
                                   %15.8lf %15.8lf %15.8lf 0 0\n", position_x[i] - 0.5*box_size[0], 
                                                                   position_y[i] - 0.5*box_size[1], 
                                                                   position_z[i] - 0.5*box_size[2], 
                                                                   0.0, 0.0, 0.0, 
                                                                   orient_x[i], orient_y[i], orient_z[i], 
                                                                   0.0, 0.0, 0.0);
             }
	     fclose(f_qmga);
	     break;
  }
}
//*********************************************************************************************
//*********************    Escribo posiciones a archivo en formato xyz   **********************
void print_pos_xyz(double *r_x, double *r_y, double *r_z, int option){
  int  i;
  char file_name[] = "DiscosEsfera.xyz";
  FILE *file_xyz;
  switch(option){
	 case 0: // Inicio de escritura
             file_xyz = fopen(file_name,"w");   // Inicia escritura por 1ra vez
             fprintf (file_xyz, "%10d %15lf %15lf %15lf \n", num_particles, box_size[0], box_size[1], box_size[2]);
             fprintf (file_xyz, "Hola, cara de bola  \n");
             fclose(file_xyz);
         break;

	 case 1: // Escritura de configuración intermedia
             file_xyz = fopen(file_name,"a");   // Continúa escritura
             for (i = 0; i < num_particles; i++){
                  fprintf(file_xyz, "H %15lf %15lf %15lf \n", r_x[i], r_y[i], r_z[i]);
             }
             fclose(file_xyz);
	 break;
  }
}
//*********************************************************************************************
//*******************          Free the memory of 3 linear arrays           *******************
void free_3_arrays(double **v1, double **v2, double **v3) {

    free(*v1);  *v1 = NULL;
    free(*v2);  *v2 = NULL;
    free(*v3);  *v3 = NULL;

    return;
}
//*********************************************************************************************
//**********************    Escribe conf final a archivo fortSphere.1    **********************
void esc_fort1(double *position_x, double *position_y, double *position_z, double *orient_x, double *orient_y, double *orient_z) {
  int    i;
  FILE   *f_fort;

  f_fort = fopen ("fortSphere.1","w");
  fprintf (f_fort, "%d %15.8lf %15.8lf %15.8lf %15.8lf\n", num_particles, box_size[0], box_size[1], box_size[2], colloid_radius);

  for (i=0; i<num_particles; i++)
  {
      fprintf (f_fort, "%15.8lf %15.8lf %15.8lf\n", position_x[i], position_y[i], position_z[i]);
      fprintf (f_fort, "%15.8lf %15.8lf %15.8lf\n", 0.0,    0.0,    0.0);
      fprintf (f_fort, "%15.8lf %15.8lf %15.8lf\n", orient_x[i], orient_y[i], orient_z[i]);
      fprintf (f_fort, "%15.8lf %15.8lf %15.8lf\n", 0.0,    0.0,    0.0);
  }

  fclose(f_fort);

}
//*********************************************************************************************
//**********************    Escribe conf final a archivo fortSphere.1    **********************
void backup_restart_file(double *position_x, double *position_y, double *position_z, double *orient_x, double *orient_y, double *orient_z) {
  int    i;
  FILE   *f_fort;

  f_fort = fopen ("fortSphere.1.BAK","w");
  fprintf (f_fort, "%d %15.8lf %15.8lf %15.8lf %15.8lf\n", num_particles, box_size[0], box_size[1], box_size[2], colloid_radius);

  for (i=0; i<num_particles; i++)
  {
      fprintf (f_fort, "%15.8lf %15.8lf %15.8lf\n", position_x[i], position_y[i], position_z[i]);
      fprintf (f_fort, "%15.8lf %15.8lf %15.8lf\n", 0.0,    0.0,    0.0);
      fprintf (f_fort, "%15.8lf %15.8lf %15.8lf\n", orient_x[i], orient_y[i], orient_z[i]);
      fprintf (f_fort, "%15.8lf %15.8lf %15.8lf\n", 0.0,    0.0,    0.0);
  }

  fclose(f_fort);
}
//*********************************************************************************************
//*********************************************************************************************
//*********************************************************************************************
//*********************************************************************************************
//*********************************************************************************************
//*********************************************************************************************

void get_3D_rank_grid_dims(int* n_ranks, int* grid_dims) {

    // Get the number of ranks
    MPI_Comm_size(MPI_COMM_WORLD, n_ranks);

    // Let MPI choose the optimal decomposition
    MPI_Dims_create(*n_ranks, 3, grid_dims);  // The 3 indicates 3D

    return;
}

void get_block_size(double* block_size, double* box_size, int* grid_dims) {

    block_size[0] = box_size[0] / grid_dims[0];
    block_size[1] = box_size[1] / grid_dims[1];
    block_size[2] = box_size[2] / grid_dims[2];

    return;
}

void get_block_coords(int block_coords[][3], int* grid_dims) {

    int i, j, k;
    int index;

    for (k=0; k < grid_dims[2]; k++) {
        for (j=0; j < grid_dims[1]; j++) {
            for (i=0; i < grid_dims[0]; i++) {
                
                // Use this instead of index++ because
                // this is thread-safe
                index = i + 
                        j*grid_dims[0] + 
                        k*grid_dims[0]*grid_dims[1];

                block_coords[index][0] = i;
                block_coords[index][1] = j;
                block_coords[index][2] = k;
            }
        }
    }

    return;
}


void block_init(Block* block, int rank, int capacity, int* block_coords, double* block_size) {
    
    int i;
    int x_coords, y_coords, z_coords;
    int cell_capacity = get_cell_capacity(capacity);
    
    block->rank = rank;
    block->capacity = capacity;
    block->num_particles = 0;

    memcpy(block->coords, block_coords, 3*sizeof(int));
    memcpy(block->size,   block_size,   3*sizeof(double));
    
    // Fill indexes of each cell
    for (i = 0; i < 8; i++) {
        block->cells_index[i] = i;
    }

    // Calculate domain boundaries and cell size
    for (i = 0; i < 3; i++) {
        block->domain_min[i] = block_coords[i] * block_size[i];
        block->domain_max[i] = (block_coords[i] + 1) * block_size[i];

        block->cell_size[i] = (block->domain_max[i] - block->domain_min[i]) / 2.0;
    }

    // Initialize local subcells A-H
    for (z_coords = 0; z_coords < 2; z_coords++) {
        for (y_coords = 0; y_coords < 2; y_coords++) {
            for (x_coords = 0; x_coords < 2; x_coords++) {

                // Pass the address of each local_cell to a cell struct
                // The local cells will remain untouch unless updated by update_cell function
                Cell *cell = &block->local_cells[x_coords][y_coords][z_coords];
                cell_init(cell, 
                          block->rank, 
                          cell_capacity, 
                          (int [3]) {x_coords, y_coords, z_coords},
                          block->cell_size, 
                          block->domain_min);
            }
        }
    }

    // Initialize extended cell grid (local and ghost cells)
    for (z_coords = 0; z_coords < 3; z_coords++) {
        for (y_coords = 0; y_coords < 3; y_coords++) {
            for (x_coords = 0; x_coords < 3; x_coords++) {

                // Pass the address of each local and ghost cell to a cell struct
                Cell *cell = &block->extended_grid[x_coords][y_coords][z_coords];
                cell_init(cell, 
                          block->rank, 
                          cell_capacity, 
                          (int [3]) {0, 0, 0},
                          block->cell_size, 
                          block->domain_min);
            }
        }
    }

    block->are_cells_reset = 0;

    block->batch = (Particle *) calloc(block->capacity, sizeof(Particle));
    
    // Build neighbor rank mapping for extended grid
    // build_neighbor_rank_map(block, grid_dims);

    return;
}

void cell_init(Cell *cell, int block_owner, int capacity, int* cell_coords, double* cell_size, double block_domain_min[3]) {
    
    int i;

    // Set index and owners
    cell->block_owner = block_owner;
    cell->index       = cell_coords_to_index(cell_coords[0], 
                                             cell_coords[1], 
                                             cell_coords[2]);

    // Set initial capacity
    cell->capacity      = capacity;
    cell->num_particles = 0;

    memcpy(cell->coords, cell_coords, 3*sizeof(int));
    memcpy(cell->size,   cell_size,   3*sizeof(double));

    // Calculate cell boundaries
    for (i = 0; i < 3; i++) {
        cell->domain_min[i] = cell_coords[i] * cell_size[i] + block_domain_min[i];
        cell->domain_max[i] = (cell_coords[i] + 1) * cell_size[i] + block_domain_min[i];
    }

    // Allocate memory
    cell->particles_index = (int *) calloc(cell->capacity, sizeof(int));
    cell->particles = (Particle *) malloc(cell->capacity * sizeof(Particle));
    
    for (i = 0; i < cell->capacity; i++) {
        particle_plain_init(&cell->particles[i]);
    }
    
    cell->is_reset = 1;

    return;
}

void particle_init(Particle* particle, int index, double position[3], double orient[3], int max_neighbs) {
    
    particle->global_index = index;
    particle->local_index = 0;

    particle->cell_owner = 0;
    particle->block_owner = 0;

    particle->position[0] = position[0];
    particle->position[1] = position[1];
    particle->position[2] = position[2];

    particle->orient[0] = orient[0];
    particle->orient[1] = orient[1];
    particle->orient[2] = orient[2];

    particle->num_neighbs = 0;
    particle->max_neighbs = max_neighbs;

    particle->neighbs_index = (int *)calloc(particle->max_neighbs, sizeof(int));
    
    return;
}


void particle_plain_init(Particle* particle) {
    
    particle->global_index = 0;
    particle->local_index = 0;

    particle->position[0] = 0;
    particle->position[1] = 0;
    particle->position[2] = 0;

    particle->orient[0] = 0;
    particle->orient[1] = 0;
    particle->orient[2] = 0;

    particle->num_neighbs = 0;
    particle->max_neighbs = 0;

    particle->neighbs_index = NULL;
    
    return;
}


void block_free(Block* block) {

    int i;
    int x_coords, y_coords, z_coords;

    // Free the allocated memory for the batch of particles
    for (i = 0; i < block->capacity; i++) {
        if (block->batch[i].neighbs_index != NULL) {
            particle_free(&block->batch[i]);
        }
    }

    free(block->batch); block->batch = NULL;

    // Set values to zero
    block->rank = 0;
    block->capacity = 0;
    block->num_particles = 0;

    // Set array values to zero
    for (i = 0; i < 8; i++) {
        block->cells_index[i] = 0;
    }

    // Set array values to zero
    for (i = 0; i < 3; i++) {
        block->coords[i]      = 0;
        block->size[i]        = 0;
        
        block->domain_min[i]  = 0;
        block->domain_max[i]  = 0;

        block->cells_index[i] = 0;
        block->cell_size[i]   = 0;
    }

    // Free local subcells A-H
    for (z_coords = 0; z_coords < 2; z_coords++) {
        for (y_coords = 0; y_coords < 2; y_coords++) {
            for (x_coords = 0; x_coords < 2; x_coords++) {

                // Pass the address of each local_cell to a cell struct
                Cell *cell = &block->local_cells[x_coords][y_coords][z_coords];
                cell_free(cell);
            }
        }
    }

    // Free extended cell grid (local and ghost cells)
    for (z_coords = 0; z_coords < 3; z_coords++) {
        for (y_coords = 0; y_coords < 3; y_coords++) {
            for (x_coords = 0; x_coords < 3; x_coords++) {

                // Pass the address of each local and ghost cell to a cell struct
                Cell *cell = &block->extended_grid[x_coords][y_coords][z_coords];
                cell_free(cell);
            }
        }
    }

    // Set True that cells were reset
    block->are_cells_reset = 1;

    return;

}

void cell_free(Cell *cell) {
    
    int i;

    free_particle_array(cell->particles, cell->capacity);
    free(cell->particles_index); cell->particles_index = NULL;
    
    // Set values to zero
    cell->block_owner = 0;
    cell->index       = 0;
    cell->capacity      = 0;
    cell->num_particles = 0;

    // Set array values to zero
    for (i = 0; i < 3; i++) {
        cell->coords[i]     = 0;
        cell->size[i]       = 0;
        cell->domain_min[i] = 0;
        cell->domain_max[i] = 0;
    }

    cell->is_reset = 0;

    return;
}

void particle_free(Particle* particle) {
    
    int i;

    // Set values to zero
    particle->global_index = 0;
    particle->local_index  = 0;
    particle->cell_owner   = 0;
    particle->block_owner  = 0;
    particle->num_neighbs  = 0;
    particle->max_neighbs  = 0;

    // Set array values to zero
    for (i = 0; i < 3; i++) {
        particle->position[i]  = 0;
        particle->orient[i]    = 0;
    }
    
    // Free memory
    free(particle->neighbs_index); particle->neighbs_index = NULL;
    
    return;
}


/* MATH ****************************************************************************************/
/**
 * @brief Helper function to allocate memory for a 2D array
 */
int** allocate_int_matrix(int rows, int cols) {

    // Allocate memory for 'rows' number of pointers (for each row)
    int** matrix = (int **) calloc(rows, sizeof(int*));

    if (matrix == NULL) {
        printf("Memory allocation failed for matrix rows\n");
        // return 1;
    }

    // Allocate memory for each row
    for (int i = 0; i < rows; i++) {
        matrix[i] = (int *) calloc(cols, sizeof(int));
        if (matrix[i] == NULL) {
            printf("Memory allocation failed for row %d\n", i);
            // return 1;
        }
    }

    return matrix;
}

double** allocate_dbl_matrix(int rows, int cols) {

    // Allocate memory for 'rows' number of pointers (for each row)
    double** matrix = (double **) calloc(rows, sizeof(double*));

    if (matrix == NULL) {
        printf("Memory allocation failed for matrix rows\n");
        // return 1;
    }

    // Allocate memory for each row
    for (int i = 0; i < rows; i++) {
        matrix[i] = (double *) calloc(cols, sizeof(double));
        if (matrix[i] == NULL) {
            printf("Memory allocation failed for row %d\n", i);
            // return 1;
        }
    }

    return matrix;
}

Particle** allocate_particle_matrix(int rows, int cols) {

    // Allocate memory for 'rows' number of pointers (for each row)
    Particle** matrix = (Particle **) calloc(rows, sizeof(Particle*));

    if (matrix == NULL) {
        printf("Memory allocation failed for matrix rows\n");
        // return 1;
    }

    // Allocate memory for each row
    for (int i = 0; i < rows; i++) {
        matrix[i] = (Particle *) calloc(cols, sizeof(Particle));
        if (matrix[i] == NULL) {
            printf("Memory allocation failed for row %d\n", i);
            // return 1;
        }
    }

    return matrix;
}

/**
 * @brief Helper function to free memory for a 2D array
 */
void free_int_matrix(int** matrix, int rows) {

    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }

    free(matrix); matrix = NULL;

    return;
}

/**
 * @brief Helper function to free memory for a 2D array
 */
void free_dbl_matrix(double** matrix, int rows) {

    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }

    free(matrix); matrix = NULL;

    return;
}


/**
 * @brief Helper function to free memory for a 2D array
 */
void free_particle_matrix(Particle** matrix, int rows, int cols) {

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (matrix[i][j].neighbs_index != NULL) {
               particle_free(&matrix[i][j]);
            }
        }
        free(matrix[i]);
    }

    free(matrix); matrix = NULL;

    return;
}

/**
 * @brief Helper function to free memory for a 2D array
 */
void free_particle_array(Particle* array, int rows) {

    for (int i = 0; i < rows; i++) {
        if (array[i].neighbs_index != NULL) {
            particle_free(&array[i]);
        }
    }

    free(array); array = NULL;

    return;
}

/** 
 * @brief Calculate the cell index given the cell coordinates
 */
int cell_coords_to_index(int x, int y, int z) {

    return (x + 2*y + 4*z);
}

/** 
 * @brief Calculate the cell coordinates from the cell index in a 2x2x2 array 
 */
void index_to_cell_coords(int index, int* x, int* y, int* z) {

    int local_coords[3];

    for (int i=0; i < 3; i++) {        
        local_coords[i] = index % 2;
        index = index / 2;
    }

    *x = local_coords[0];
    *y = local_coords[1];
    *z = local_coords[2];

    return;
}

/** 
 * @brief This algorithm shuffles the entries of an int array of size n using the Fisher-Yates algorithm
 * @param array   array of integers to shuffle
 * @param n       length of array
*/
void shuffle_int_array(int* array, int n) {

    for (int i = n-1; i > 0; i--) {

        unsigned int limit = RAND_MAX - (RAND_MAX % (i + 1));
        unsigned int r;
        do {
            r = rand();
        } while (r > limit);

        int j = r % (i + 1);

        swap(&array[i], &array[j]);
    }

    return; 
}

/** 
 * @brief Swap the values of a and b
 */
void swap(int* a, int* b) {
    
    int temp = *a;
    *a = *b;
    *b = temp;
}

/**
 * Finds which MPI rank owns a ghost cell given its relative coordinates to the active cell
 * and returns the coordinates of the owner block and the offset for the 3D coordinates (x,y,z) 
 * of the block owner block
 * 
 * @param cell_rel_coords       [in] 3D cell coordinates (x,y,z) relative to the active cell to look for its owner
 * @param current_block_coords  [in] 3D coordinates (x,y,z) of the current block
 * @param owner_block_coords    [out] 3D coordinates (x,y,z) of the owner block
 * @param block_offset          [out] offset for the 3D coordinates (x,y,z) of the block owner block
 * @param grid_dims             [in] 3D dimensions of the MPI Cartesian grid
 * @param local_grid_dim        [in] Size of local grid in each block (usually 2, indicating a 2x2x2 grid)
 * @return                      Rank number that owns the specified block coordinates
 */
int get_cell_owner_full(int cell_rel_coords[3], 
                        int current_block_coords[3], 
                        int owner_block_coords[3],
                        int block_offset[3] 
                        int grid_dims[3], 
                        int local_grid_dim) {
    
    int block_owner;

    block_offset[0] = cell_rel_coords[0] < 0 ? -1 : (cell_rel_coords[0] >= local_grid_dim ? 1 : 0);
    block_offset[1] = cell_rel_coords[1] < 0 ? -1 : (cell_rel_coords[1] >= local_grid_dim ? 1 : 0);
    block_offset[2] = cell_rel_coords[2] < 0 ? -1 : (cell_rel_coords[2] >= local_grid_dim ? 1 : 0);
    
    owner_block_coords[0] = mod(current_block_coords[0] + block_offset[0], grid_dims[0]);
    owner_block_coords[1] = mod(current_block_coords[1] + block_offset[1], grid_dims[1]);
    owner_block_coords[2] = mod(current_block_coords[2] + block_offset[2], grid_dims[2]);

    block_owner = get_block_index(owner_block_coords, grid_dims);
    
    return block_owner;
}


/**
 * Finds which MPI rank owns a ghost cell given its local coordinates
 * 
 * @param cell_coords           [in] 3D local coordinates (x,y,z) of the ghost cell to look for its owner
 * @param current_block_coords  [in] 3D coordinates (x,y,z) of the current block
 * @param grid_dims             [in] 3D dimensions of the MPI Cartesian grid
 * @param grid_dims             [in] 3D dimensions of the MPI Cartesian grid
 * @param local_grid_dim        [in] Size of local grid in each block (usually 2, indicating a 2x2x2 grid)
 * @return                      Rank number that owns the specified block coordinates
 */
int get_cell_owner(int cell_coords[3], 
                   int current_block_coords[3], 
                   int grid_dims[3], 
                   int local_grid_dim) {
    
    int block_owner;
    int owner_block_coords[3] = {0, 0, 0};
    int offset[3] = {0, 0, 0};

    offset[0] = cell_coords[0] < 0 ? -1 : (cell_coords[0] >= local_grid_dim ? 1 : 0);
    offset[1] = cell_coords[1] < 0 ? -1 : (cell_coords[1] >= local_grid_dim ? 1 : 0);
    offset[2] = cell_coords[2] < 0 ? -1 : (cell_coords[2] >= local_grid_dim ? 1 : 0);
    
    owner_block_coords[0] = mod(current_block_coords[0] + offset[0], grid_dims[0]);
    owner_block_coords[1] = mod(current_block_coords[1] + offset[1], grid_dims[1]);
    owner_block_coords[2] = mod(current_block_coords[2] + offset[2], grid_dims[2]);

    block_owner = get_block_index(owner_block_coords, grid_dims);
    
    return block_owner;
}


/**
 * Gets the rank owner of a block given its 3D coordinates
 * 
 * @param block_coords  [in] 3D coordinates (x,y,z) of the block
 * @param grid_dims     [in] 3D dimensions of the MPI Cartesian grid
 * @return              Rank number that owns the specified block coordinates
 */
int get_block_index(int block_coords[3], int grid_dims[3]) {
    
    // Apply periodic boundary conditions to coordinates
    int x = mod(block_coords[0], grid_dims[0]);
    int y = mod(block_coords[1], grid_dims[1]);
    int z = mod(block_coords[2], grid_dims[2]);
    
    // Convert 3D coordinates to linear rank using row-major ordering
    return x + y*grid_dims[0] + z*grid_dims[0]*grid_dims[1];
}

/**
 * @brief Calculates the python-like module operation -> a % b
 */
static inline int mod(int a, int b) {
    return (a % b + b) % b;
}


/*
 * activate cell
 * build 

*/



//################################################################
//################################################################

/**
 * @brief Check the result of a parameter read operation.
 *
 * @param result      Number of values successfully read by sscanf().
 * @param expected    Required number of successfully parsed values.
 * @param paramName   Name of the parameter being parsed (for reporting).
 */
void error_checker(int result, int expected, const char* paramName) {
    if (result != expected) {
        fprintf(stderr, "Error reading parameter: %s\n", paramName);
        exit(1);
    }
}

/**
 * @brief Read the next non-empty, non-comment line from a parameter file.
 *
 * This helper function reads lines from @p file, ignoring:
 * - empty lines
 * - lines starting with '#'
 *
 * Once a valid line is found, it tries to parse:
 *
 *     <paramName> <value...>
 *
 * where <value...> may contain spaces until the end of the line.
 *
 * Example accepted line:
 *     system_temperature   1.25
 *
 * @param file        [in] Handle to an already-opened configuration file.
 * @param paramName   [out] Buffer where the parameter name will be stored.
 * @param value       [out] Buffer where the parameter value(s) will be stored.
 *
 * @return 1 if a parameter line was successfully read,  
 *         0 if end-of-file is reached.
 */
int read_next_parameter(FILE *file, char *paramName, char *value) {
    char line[256];
    
    while (fgets(line, sizeof(line), file)) {
        // Skip empty lines and comment lines
        if (line[0] == '\n' || line[0] == '#' || line[0] == '\0')
            continue;
        
        // Try to parse parameter name and value
        if (sscanf(line, "%255s %255[^\n]", paramName, value) >= 1) {
            return 1; // Successfully read a parameter
        }
    }
    return 0; // End of file or error
}

/**
 * @brief Read all initial conditions and simulation parameters from a file.
 *
 * This function loads 23 required Monte Carlo simulation parameters from
 * a text configuration file. The file can contain:
 *   - parameter lines of the form: `<name> <value>`
 *   - comment lines starting with '#'
 *   - empty lines
 *
 * Each parameter is read, parsed with sscanf(), stored in the
 * corresponding global variable, and validated using error_checker().
 *
 * If one or more required parameters are missing, the function prints
 * the missing names and terminates the program.
 *
 * After successfully reading all parameters, the function computes the 
 * derived quantity:
 *
 *     gb_sigma_endtoend = gb_sigma_sidetoside * gb_kappa;
 *
 * @param filename  Path to the configuration file.
 *
 * @note The function terminates the program on any error (missing
 *       parameter, unreadable file, parsing failure).
 */
void load_initial_conditions(const char* filename) {
    
    FILE *ini_file;
    char paramName[256];
    char value[256];
    int errorCheck;
    
    // Track which parameters we've read
    bool paramsRead[23] = {false};
    int totalParamsRead = 0;

    ini_file = fopen(filename, "r");
    if (ini_file == NULL) {
        perror("Error opening the parameters file:");
        printf("%s", filename);
        printf("\nCRITICAL ERROR: Make sure the file exists in your project's root folder.\n");
        exit(1);
    }

    while (read_next_parameter(ini_file, paramName, value)) {
        
        if (strcmp(paramName, "mc_ensemble_type") == 0) {
            errorCheck = sscanf(value, "%d", &mc_ensemble_type);
            error_checker(errorCheck, 1, "mc_ensemble_type");
            paramsRead[0] = true;
            totalParamsRead++;

        } else if (strcmp(paramName, "num_particles") == 0) {
            errorCheck = sscanf(value, "%d", &num_particles);
            error_checker(errorCheck, 1, "num_particles");
            paramsRead[1] = true;
            totalParamsRead++;

        } else if (strcmp(paramName, "system_density") == 0) {
            errorCheck = sscanf(value, "%lf", &system_density);
            error_checker(errorCheck, 1, "system_density");
            paramsRead[2] = true;
            totalParamsRead++;

        } else if (strcmp(paramName, "system_temperature") == 0) {
            errorCheck = sscanf(value, "%lf", &system_temperature);
            error_checker(errorCheck, 1, "system_temperature");
            paramsRead[3] = true;
            totalParamsRead++;

        } else if (strcmp(paramName, "system_pressure") == 0) {
            errorCheck = sscanf(value, "%lf", &system_pressure);
            error_checker(errorCheck, 1, "system_pressure");
            paramsRead[4] = true;
            totalParamsRead++;

        } else if (strcmp(paramName, "gb_sigma_sidetoside") == 0) {
            errorCheck = sscanf(value, "%lf", &gb_sigma_sidetoside);
            error_checker(errorCheck, 1, "gb_sigma_sidetoside");
            paramsRead[5] = true;
            totalParamsRead++;

        } else if (strcmp(paramName, "gb_epsilon0") == 0) {
            errorCheck = sscanf(value, "%lf", &gb_epsilon0);
            error_checker(errorCheck, 1, "gb_epsilon0");
            paramsRead[6] = true;
            totalParamsRead++;

        } else if (strcmp(paramName, "gb_kappa") == 0) {
            errorCheck = sscanf(value, "%lf", &gb_kappa);
            error_checker(errorCheck, 1, "gb_kappa");
            paramsRead[7] = true;
            totalParamsRead++;

        } else if (strcmp(paramName, "gb_kappa_prime") == 0) {
            errorCheck = sscanf(value, "%lf", &gb_kappa_prime);
            error_checker(errorCheck, 1, "gb_kappa_prime");
            paramsRead[8] = true;
            totalParamsRead++;

        } else if (strcmp(paramName, "gb_mu_exponent") == 0) {
            errorCheck = sscanf(value, "%lf", &gb_mu_exponent);
            error_checker(errorCheck, 1, "gb_mu_exponent");
            paramsRead[9] = true;
            totalParamsRead++;

        } else if (strcmp(paramName, "gb_nu_exponent") == 0) {
            errorCheck = sscanf(value, "%lf", &gb_nu_exponent);
            error_checker(errorCheck, 1, "gb_nu_exponent");
            paramsRead[10] = true;
            totalParamsRead++;

        } else if (strcmp(paramName, "max_displacement") == 0) {
            errorCheck = sscanf(value, "%f", &max_displacement);
            error_checker(errorCheck, 1, "max_displacement");
            paramsRead[11] = true;
            totalParamsRead++;

        } else if (strcmp(paramName, "max_rotation") == 0) {
            errorCheck = sscanf(value, "%f", &max_rotation);
            error_checker(errorCheck, 1, "max_rotation");
            paramsRead[12] = true;
            totalParamsRead++;

        } else if (strcmp(paramName, "initial_configuration_type") == 0) {
            errorCheck = sscanf(value, "%d", &initial_configuration_type);
            error_checker(errorCheck, 1, "initial_configuration_type");
            paramsRead[13] = true;
            totalParamsRead++;

        } else if (strcmp(paramName, "cutoff_radius") == 0) {
            errorCheck = sscanf(value, "%lf", &cutoff_radius);
            error_checker(errorCheck, 1, "cutoff_radius");
            paramsRead[14] = true;
            totalParamsRead++;

        } else if (strcmp(paramName, "verlet_radius") == 0) {
            errorCheck = sscanf(value, "%lf", &verlet_radius);
            error_checker(errorCheck, 1, "verlet_radius");
            paramsRead[15] = true;
            totalParamsRead++;

        } else if (strcmp(paramName, "total_mc_cycles") == 0) {
            errorCheck = sscanf(value, "%d", &total_mc_cycles);
            error_checker(errorCheck, 1, "total_mc_cycles");
            paramsRead[16] = true;
            totalParamsRead++;

        } else if (strcmp(paramName, "configuration_save_interval") == 0) {
            errorCheck = sscanf(value, "%d", &configuration_save_interval);
            error_checker(errorCheck, 1, "configuration_save_interval");
            paramsRead[17] = true;
            totalParamsRead++;

        } else if (strcmp(paramName, "statistics_print_interval") == 0) {
            errorCheck = sscanf(value, "%d", &statistics_print_interval);
            error_checker(errorCheck, 1, "statistics_print_interval");
            paramsRead[18] = true;
            totalParamsRead++;

        } else if (strcmp(paramName, "colloid_wall_option") == 0) {
            errorCheck = sscanf(value, "%d", &colloid_wall_option);
            error_checker(errorCheck, 1, "colloid_wall_option");
            paramsRead[19] = true;
            totalParamsRead++;

        } else if (strcmp(paramName, "colloid_interaction_epsilon") == 0) {
            errorCheck = sscanf(value, "%lf", &colloid_interaction_epsilon);
            error_checker(errorCheck, 1, "colloid_interaction_epsilon");
            paramsRead[20] = true;
            totalParamsRead++;

        } else if (strcmp(paramName, "colloid_interaction_amplitude") == 0) {
            errorCheck = sscanf(value, "%lf", &colloid_interaction_amplitude);
            error_checker(errorCheck, 1, "colloid_interaction_amplitude");
            paramsRead[21] = true;
            totalParamsRead++;

        } else if (strcmp(paramName, "colloid_radius") == 0) {
            errorCheck = sscanf(value, "%lf", &colloid_radius);
            error_checker(errorCheck, 1, "colloid_radius");
            paramsRead[22] = true;
            totalParamsRead++;

        } else {
            printf("Warning: Unknown parameter '%s' in configuration file\n", paramName);
        }
    }

    fclose(ini_file);

    // Check if all required parameters were read
    if (totalParamsRead != 23) {
        printf("Error: Not all parameters were read correctly. Expected 23, got %d\n", totalParamsRead);
        printf("Missing parameters:\n");
        const char *paramNames[] = {
            "mc_ensemble_type", "num_particles", "system_density", "system_temperature",
            "system_pressure", "gb_sigma_sidetoside", "gb_epsilon0", "gb_kappa",
            "gb_kappa_prime", "gb_mu_exponent", "gb_nu_exponent", "max_displacement",
            "max_rotation", "initial_configuration_type", "cutoff_radius",
            "verlet_radius", "total_mc_cycles", "configuration_save_interval",
            "statistics_print_interval", "colloid_wall_option", "colloid_interaction_epsilon",
            "colloid_interaction_amplitude", "colloid_radius"
        };
        
        for (int i = 0; i < 23; i++) {
            
            if (!paramsRead[i]) {

                printf("  - %s", paramNames[i]);

                if ((i == 5) || (i == 7)) {
                    printf(" cannot generate gb_sigma_endtoend variable.\n");
                } else {
                    printf("\n");
                }
            }
        }
        exit(1);
    }

    // Calculate derived value
    gb_sigma_endtoend = gb_sigma_sidetoside * gb_kappa;
    
    // printf("All parameters read successfully!\n");
}


void broadcast_global_vars(int master_rank) {

    MPI_Bcast(&num_particles,                 1, MPI_INT,    master_rank, MPI_COMM_WORLD);
    MPI_Bcast(&total_mc_cycles,               1, MPI_INT,    master_rank, MPI_COMM_WORLD);
    MPI_Bcast(&configuration_save_interval,   1, MPI_INT,    master_rank, MPI_COMM_WORLD);
    MPI_Bcast(&initial_configuration_type,    1, MPI_INT,    master_rank, MPI_COMM_WORLD);
    MPI_Bcast(&statistics_print_interval,     1, MPI_INT,    master_rank, MPI_COMM_WORLD);
    MPI_Bcast(&colloid_wall_option,           1, MPI_INT,    master_rank, MPI_COMM_WORLD);
    MPI_Bcast(&mc_ensemble_type,              1, MPI_INT,    master_rank, MPI_COMM_WORLD);

    MPI_Bcast(&max_displacement,              1, MPI_FLOAT,  master_rank, MPI_COMM_WORLD);
    MPI_Bcast(&max_rotation,                  1, MPI_FLOAT,  master_rank, MPI_COMM_WORLD);

    MPI_Bcast(&system_pressure,               1, MPI_DOUBLE, master_rank, MPI_COMM_WORLD);
    MPI_Bcast(&system_temperature,            1, MPI_DOUBLE, master_rank, MPI_COMM_WORLD);
    MPI_Bcast(&system_density,                1, MPI_DOUBLE, master_rank, MPI_COMM_WORLD);

    MPI_Bcast(&gb_epsilon0,                   1, MPI_DOUBLE, master_rank, MPI_COMM_WORLD);
    MPI_Bcast(&gb_kappa,                      1, MPI_DOUBLE, master_rank, MPI_COMM_WORLD);
    MPI_Bcast(&gb_kappa_prime,                1, MPI_DOUBLE, master_rank, MPI_COMM_WORLD);
    MPI_Bcast(&gb_mu_exponent,                1, MPI_DOUBLE, master_rank, MPI_COMM_WORLD);
    MPI_Bcast(&gb_nu_exponent,                1, MPI_DOUBLE, master_rank, MPI_COMM_WORLD);
    MPI_Bcast(&gb_sigma_sidetoside,           1, MPI_DOUBLE, master_rank, MPI_COMM_WORLD);
    MPI_Bcast(&gb_sigma_endtoend,             1, MPI_DOUBLE, master_rank, MPI_COMM_WORLD);

    MPI_Bcast(&cutoff_radius,                 1, MPI_DOUBLE, master_rank, MPI_COMM_WORLD);
    MPI_Bcast(&verlet_radius,                 1, MPI_DOUBLE, master_rank, MPI_COMM_WORLD);
    
    MPI_Bcast(&colloid_interaction_amplitude, 1, MPI_DOUBLE, master_rank, MPI_COMM_WORLD);
    MPI_Bcast(&colloid_interaction_epsilon,   1, MPI_DOUBLE, master_rank, MPI_COMM_WORLD);

    MPI_Bcast(&colloid_radius,                1, MPI_DOUBLE, master_rank, MPI_COMM_WORLD);
    MPI_Bcast(&colloid_volume,                1, MPI_DOUBLE, master_rank, MPI_COMM_WORLD);
    MPI_Bcast(&box_volume,                    1, MPI_DOUBLE, master_rank, MPI_COMM_WORLD);
    MPI_Bcast(&system_density,                1, MPI_DOUBLE, master_rank, MPI_COMM_WORLD);

    MPI_Bcast(box_size,                       3, MPI_DOUBLE, master_rank, MPI_COMM_WORLD);
    MPI_Bcast(colloid_coords,                 3, MPI_DOUBLE, master_rank, MPI_COMM_WORLD);

    MPI_Bcast(rot_vec_x,       POINTS_ON_SPHERE, MPI_FLOAT,  master_rank, MPI_COMM_WORLD);
    MPI_Bcast(rot_vec_y,       POINTS_ON_SPHERE, MPI_FLOAT,  master_rank, MPI_COMM_WORLD);
    MPI_Bcast(rot_vec_z,       POINTS_ON_SPHERE, MPI_FLOAT,  master_rank, MPI_COMM_WORLD);

    // Broadcast constants ###############################################################

    MPI_Bcast(&gb_beta,                       1, MPI_DOUBLE, master_rank, MPI_COMM_WORLD);
    MPI_Bcast(&gb_4eps0,                      1, MPI_DOUBLE, master_rank, MPI_COMM_WORLD);
    MPI_Bcast(&cutoff_radius_sq,              1, MPI_DOUBLE, master_rank, MPI_COMM_WORLD);
    MPI_Bcast(&r_verlet_sq,                   1, MPI_DOUBLE, master_rank, MPI_COMM_WORLD);
    MPI_Bcast(&verlet_volume,                 1, MPI_DOUBLE, master_rank, MPI_COMM_WORLD);
    MPI_Bcast(&gb_k_imu,                      1, MPI_DOUBLE, master_rank, MPI_COMM_WORLD);
    MPI_Bcast(&gb_inercia,                    1, MPI_DOUBLE, master_rank, MPI_COMM_WORLD);
    MPI_Bcast(&msad,                          1, MPI_DOUBLE, master_rank, MPI_COMM_WORLD);
    MPI_Bcast(&gb_chi,                        1, MPI_DOUBLE, master_rank, MPI_COMM_WORLD);
    MPI_Bcast(&gb_chi_sq,                     1, MPI_DOUBLE, master_rank, MPI_COMM_WORLD);
    MPI_Bcast(&gb_chi_prime,                  1, MPI_DOUBLE, master_rank, MPI_COMM_WORLD);
    MPI_Bcast(&gb_kappa_sq,                   1, MPI_DOUBLE, master_rank, MPI_COMM_WORLD);
    MPI_Bcast(&colloid_rod_cutoff,            1, MPI_DOUBLE, master_rank, MPI_COMM_WORLD);
    MPI_Bcast(&colloid_radius_sq,             1, MPI_DOUBLE, master_rank, MPI_COMM_WORLD);

    return;
}

void broadcast_block_params(int master_rank, double block_size[3], int block_coords[][3], int n_ranks) {

    MPI_Bcast(&block_capacity,          1, MPI_INT,    master_rank, MPI_COMM_WORLD);
    MPI_Bcast(block_size,               3, MPI_DOUBLE, master_rank, MPI_COMM_WORLD);
    MPI_Bcast(block_coords,   n_ranks * 3, MPI_INT,    master_rank, MPI_COMM_WORLD);

    return;
}

void broadcast_cell_params(int master_rank, int cell_capacity) {

    MPI_Bcast(&cell_capacity,   1, MPI_INT,    master_rank, MPI_COMM_WORLD);

    return;
}

void get_initial_config(int initial_configuration_type,
                        double *position_x, double *position_y, double *position_z, 
                        double *orient_x, double *orient_y, double *orient_z) {

    // Select how to start the initial positions and orientations of each particle
    switch (initial_configuration_type) {
        case 0:
            read_initial_configuration(position_x, position_y, position_z, orient_x, orient_y, orient_z);
            break;
        case 1:
            initialize_random_positions(position_x, position_y, position_z, orient_x, orient_y, orient_z);
            break;
        case 2:
            initialize_cubic_lattice_positions(position_x, position_y, position_z, orient_x, orient_y, orient_z);
            break;
        default:
            printf("The initial configuration type selector has invalid value\n\n");
    }

    return;
}

/**
 * @brief Allocate and initialize to zero three double arrays.
 *
 * Each array is allocated using `calloc`, ensuring all elements start at zero.
 *
 * @param[out] arr1   Pointer to the first array pointer.
 * @param[out] arr2   Pointer to the second array pointer.
 * @param[out] arr3   Pointer to the third array pointer.
 * @param[in]  length Number of elements in each array.
 */
void callocate_3_dbl_arrays(double **arr1, double **arr2, double **arr3, int length) {
    *arr1 = (double *)calloc(length, sizeof(double));
    *arr2 = (double *)calloc(length, sizeof(double));
    *arr3 = (double *)calloc(length, sizeof(double));

    if (!*arr1 || !*arr2 || !*arr3) {
        fprintf(stderr, "Error: failed to allocate memory for double arrays.\n");
        exit(EXIT_FAILURE);
    }
}

/**
 * @brief Allocate and initialize to zero three float arrays.
 *
 * @param[out] arr1   Pointer to the first array pointer.
 * @param[out] arr2   Pointer to the second array pointer.
 * @param[out] arr3   Pointer to the third array pointer.
 * @param[in]  length Number of elements in each array.
 */
void callocate_3_float_arrays(float **arr1, float **arr2, float **arr3, int length) {
    *arr1 = (float *)calloc(length, sizeof(float));
    *arr2 = (float *)calloc(length, sizeof(float));
    *arr3 = (float *)calloc(length, sizeof(float));

    if (!*arr1 || !*arr2 || !*arr3) {
        fprintf(stderr, "Error: failed to allocate memory for float arrays.\n");
        exit(EXIT_FAILURE);
    }
}


int get_block_capacity(double block_size[3]) {

    double block_volume = block_size[0] * block_size[1] * block_size[2];

    // We will use two methods to stimate the max particles per block
    int method1, method2;
    int block_max_particles;

    // If system_density < 0.2, use method2.
    method1 = (int)(1 + 2.0*(system_density + 1.0) * block_volume);
    method2 = (int)(1 + 2.4 * block_volume);

    // Choose the biggest value between method1 and method2
    block_max_particles = fmax(method1, method2);
    block_max_particles = fmin(block_max_particles, num_particles);

    // return block_max_particles;
    return 1000;
}


int get_max_num_neighb(double verlet_volume) {

    // We will use two methods to stimate the max particles per block
    int method1, method2;
    int max_neighb_particles;

    // If system_density < 0.2, use method2.
    method1 = (int)(1 + 2.0*(system_density + 1.0) * verlet_volume);
    method2 = (int)(1 + 2.4 * verlet_volume);

    // Choose the biggest value between method1 and method2
    max_neighb_particles = fmax(method1, method2);

    return max_neighb_particles;
}

int get_cell_capacity(int block_capacity) {

    return (int) (block_capacity/8 + 1); // The one is to ensure a minimum of 1 particle
}

void initialize_global_vars(const char* parameters_file,
                            double **position_x,  double **position_y, double **position_z,
                            double **orient_x,    double **orient_y,  double **orient_z,
                            double **rxi, double **ryi, double **rzi,
                            float *rot_vec_x, float *rot_vec_y, float *rot_vec_z,
                            float unitary_radius) {

    // read_simulation_parameters();
    load_initial_conditions(parameters_file);
    // print_resum();

    // Start random seed
    srand(time(0));

    /* Allocate memory for 3 1D arrays
    * position_x,  position_y,  position_z  are the particle positions (x, y, z coordinates)
    * rx1, ryi, rzi are the reference positions for neighbor neighbor_list_array updates
    * orient_x,  orient_y,  orient_z  are the particle orientation vectors (unit vectors)
    * 
    * Only rank 0 need to allocate this large arrays and then
    * broadcast all the constants and respective particles to each block
    */
    callocate_3_dbl_arrays(position_x, position_y, position_z, num_particles);
    callocate_3_dbl_arrays(orient_x, orient_y, orient_z, num_particles);
    callocate_3_dbl_arrays(rxi, ryi, rzi, num_particles);

    // Select how to start the initial positions and orientations of each particle
    get_initial_config(initial_configuration_type,
                       *position_x, *position_y, *position_z,
                       *orient_x, *orient_y, *orient_z);

    // Generate uniformly POINTS_ON_SPHERE random points on the unitary sphere
    // Note that in Marsaglia4MC we use POINTS_ON_SPHERE / 2
    // because it generates a random 3D point and its negative
    Marsaglia4MC(rot_vec_x, rot_vec_y, rot_vec_z, unitary_radius);

    return;
}

void get_particle_owners(double position[3], int* block_owner, int* cell_index, double block_size[3], int grid_dims[3]) {

    int block_coords[3];
    int cell_coords[3];

    *block_owner = get_owner_block_index(position, block_size, grid_dims, block_coords);
    *cell_index = get_owner_cell_index(position, block_size, block_coords, cell_coords);

    return;
}


int get_owner_block_index(double position[3], double block_size[3], int grid_dims[3], int block_coords[3]) {

    int rank;

    for (int i = 0; i < 3; i++) {
        block_coords[i] = (int) (position[i] / block_size[i]);
    }

    rank = get_block_index(block_coords, grid_dims);

    return rank;
}


int get_owner_cell_index(double position[3], double block_size[3], int block_coords[3], int cell_coords[3]) {

    int cell_index;
    double norm_position[3];    // Normalized particle position

    for (int i = 0; i < 3; i++) {
        norm_position[i] = position[i] - (block_size[i] * block_coords[i]);
        cell_coords[i] = ( (int) (2 * norm_position[i] / block_size[i]));
        // printf("cell_coord[%d] = %d, block_coord[%d] = %d\n", i, cell_coords[i], i, block_coords[i]);
    }

    cell_index = cell_coords_to_index(cell_coords[0], cell_coords[1], cell_coords[2]);

    return cell_index;
}

// void fill_blocks_and_cells(int block_index, int cell_index) {

//     for (int i = 0; i < num_particles; i++) {
//         double block_index;
//         double cell_index;

//         if (rank ==  rank_master) {
//             MPI_Send(particle, 1, sizeof(Particle), rank = block_index, tag = i);
//         } else if (rank = block_index) {
//             MPI_Recv(particle, 1, sizeof(Particle), rank = 0, tag = i);
//         }
//     }




//     return;
// }


MPI_Datatype create_particle_mpi_type(void) {
    
    MPI_Datatype particle_type;
    
    const int n_args = 7;

    int n_elements[]     = {1, 1, 
                            3, 3, 
                            1, 1, 
                            max_neighbs};
    
    MPI_Datatype types[] = {MPI_INT, MPI_INT, 
                            MPI_DOUBLE, MPI_DOUBLE, 
                            MPI_INT, MPI_INT,
                            MPI_INT};
    
    MPI_Aint displacements[n_args];
    
    Particle dummy;
    
    // Get addresses for each distinct block
    MPI_Get_address(&dummy.global_index,     &displacements[0]);
    MPI_Get_address(&dummy.local_index,      &displacements[1]);
    MPI_Get_address(&dummy.position[0],      &displacements[2]);
    MPI_Get_address(&dummy.orient[0],        &displacements[3]);
    MPI_Get_address(&dummy.num_neighbs,      &displacements[4]);
    MPI_Get_address(&dummy.max_neighbs,      &displacements[5]);
    MPI_Get_address(&dummy.neighbs_index[0], &displacements[6]); // Dynamic array
    
    // Make displacements relative to the first element
    MPI_Aint base = displacements[0];
    
    for (int i = 0; i < n_args; i++) {
        displacements[i] = MPI_Aint_diff(displacements[i], base);
    }
    
    MPI_Type_create_struct(n_args, n_elements, displacements, types, &particle_type);
    
    MPI_Type_commit(&particle_type);
    
    return particle_type;
}


MPI_Datatype create_cell_mpi_type(MPI_Datatype Particle_MPI) {
    
    MPI_Datatype cell_type;

    const int n_args = 11;
    
    int n_elements[]   = {1, 1, 1, 1, 
                          3, 3, 3, 3, 
                          1, 
                          cell_capacity, cell_capacity};

    MPI_Datatype types[] = {MPI_INT, MPI_INT, MPI_INT, MPI_INT, 
                            MPI_DOUBLE, MPI_INT, MPI_DOUBLE, MPI_DOUBLE, 
                            MPI_INT, 
                            Particle_MPI, MPI_INT};

    MPI_Aint displacements[n_args];
    
    Cell dummy;
    
    MPI_Get_address(&dummy.index,              &displacements[0]);
    MPI_Get_address(&dummy.block_owner,         &displacements[1]);
    MPI_Get_address(&dummy.capacity,           &displacements[2]);
    MPI_Get_address(&dummy.num_particles,      &displacements[3]);
    MPI_Get_address(&dummy.size[0],            &displacements[4]);
    MPI_Get_address(&dummy.coords[0],          &displacements[5]);  
    MPI_Get_address(&dummy.domain_min[0],      &displacements[6]);
    MPI_Get_address(&dummy.domain_max[0],      &displacements[7]);
    MPI_Get_address(&dummy.is_reset,           &displacements[8]);
    MPI_Get_address(&dummy.particles[0],       &displacements[9]);
    MPI_Get_address(&dummy.particles_index[0], &displacements[10]);
    
    // Make displacements relative
    MPI_Aint base = displacements[0];
    
    for (int i = 0; i < n_args; i++) {
        displacements[i] = MPI_Aint_diff(displacements[i], base);
    }
    
    MPI_Type_create_struct(n_args, n_elements, displacements, types, &cell_type);
    
    MPI_Type_commit(&cell_type);
    
    return cell_type;
}

void mpi_datatypes_init(MPI_Datatype* Particle_MPI, MPI_Datatype* Cell_MPI) {
    
    // Create particle type first (needed for cell type)
    *Particle_MPI = create_particle_mpi_type();
    *Cell_MPI = create_cell_mpi_type(*Particle_MPI);

    return;
}


void send_particle(const Particle *p, int dest, int tag, MPI_Comm comm) {
    
    int pos = 0;
    int bufsize = 0;
    int tmp;

    // ---- Compute buffer size ----
    MPI_Pack_size(1, MPI_INT, comm, &tmp); bufsize += tmp; // global_index
    MPI_Pack_size(1, MPI_INT, comm, &tmp); bufsize += tmp; // local_index

    MPI_Pack_size(3, MPI_DOUBLE, comm, &tmp); bufsize += tmp; // position
    MPI_Pack_size(3, MPI_DOUBLE, comm, &tmp); bufsize += tmp; // orient

    MPI_Pack_size(1, MPI_INT, comm, &tmp); bufsize += tmp; // num_neighbs
    MPI_Pack_size(1, MPI_INT, comm, &tmp); bufsize += tmp; // max_neighbs

    MPI_Pack_size(p->max_neighbs, MPI_INT, comm, &tmp);
    bufsize += tmp;  // neigh_list

    char *buffer = malloc(bufsize);

    // ---- Pack fields ----
    MPI_Pack(&p->global_index, 1, MPI_INT, buffer, bufsize, &pos, comm);
    MPI_Pack(&p->local_index,  1, MPI_INT, buffer, bufsize, &pos, comm);

    MPI_Pack(p->position, 3, MPI_DOUBLE, buffer, bufsize, &pos, comm);
    MPI_Pack(p->orient,   3, MPI_DOUBLE, buffer, bufsize, &pos, comm);

    MPI_Pack(&p->num_neighbs, 1, MPI_INT, buffer, bufsize, &pos, comm);
    MPI_Pack(&p->max_neighbs, 1, MPI_INT, buffer, bufsize, &pos, comm);

    MPI_Pack(p->neighbs_index, p->max_neighbs, MPI_INT,
             buffer, bufsize, &pos, comm);

    // ---- Send packed data ----
    MPI_Send(buffer, pos, MPI_PACKED, dest, tag, comm);
    free(buffer);
}


void recv_particle(Particle *particle, int source, int tag, MPI_Comm comm) {
    
    MPI_Status st;

    // Probe to get message size
    int count;
    MPI_Probe(source, tag, comm, &st);
    MPI_Get_count(&st, MPI_PACKED, &count);

    char *buffer = malloc(count);
    MPI_Recv(buffer, count, MPI_PACKED, source, tag, comm, MPI_STATUS_IGNORE);

    int pos = 0;

    MPI_Unpack(buffer, count, &pos, &particle->global_index, 1, MPI_INT, comm);
    MPI_Unpack(buffer, count, &pos, &particle->local_index, 1, MPI_INT, comm);

    MPI_Unpack(buffer, count, &pos, particle->position, 3, MPI_DOUBLE, comm);
    MPI_Unpack(buffer, count, &pos, particle->orient,   3, MPI_DOUBLE, comm);

    MPI_Unpack(buffer, count, &pos, &particle->num_neighbs, 1, MPI_INT, comm);
    MPI_Unpack(buffer, count, &pos, &particle->max_neighbs, 1, MPI_INT, comm);

    // allocate the dynamic neighbour array
    particle->neighbs_index = malloc(particle->max_neighbs * sizeof(int));

    MPI_Unpack(buffer, count, &pos, particle->neighbs_index,
               particle->max_neighbs, MPI_INT, comm);

    free(buffer);
}

char* pack_particle(const Particle* p, int* out_bytes) {
    
    int bytes =
        4*sizeof(int) +
        6*sizeof(double) +
        2*sizeof(int) +
        p->max_neighbs*sizeof(int);

    *out_bytes = bytes;

    // Allocate buffer
    char* buf = malloc(bytes);
    if (!buf) {
        printf("\nERROR in 'pack_particle': buffer malloc failed\n");
        return NULL;
    }

    // Buffer pointer
    char* ptr = buf;

    // Pack particle data
    memcpy(ptr, &p->global_index, sizeof(int));     ptr += sizeof(int);
    memcpy(ptr, &p->local_index,  sizeof(int));     ptr += sizeof(int);
   
    memcpy(ptr, &p->cell_owner,   sizeof(int));     ptr += sizeof(int);
    memcpy(ptr, &p->block_owner,  sizeof(int));     ptr += sizeof(int);

    memcpy(ptr, p->position,    3*sizeof(double));  ptr += 3*sizeof(double);
    memcpy(ptr, p->orient,      3*sizeof(double));  ptr += 3*sizeof(double);

    memcpy(ptr, &p->num_neighbs, sizeof(int));      ptr += sizeof(int);
    memcpy(ptr, &p->max_neighbs, sizeof(int));      ptr += sizeof(int);

    memcpy(ptr, p->neighbs_index, p->max_neighbs*sizeof(int));

    return buf;
}


int unpack_particle(Particle* p, const char* buf) {
    
    // Buffer pointer
    const char* ptr = buf;

    // Unpack particle data
    memcpy(&p->global_index, ptr, sizeof(int));
    ptr += sizeof(int);
    memcpy(&p->local_index,  ptr, sizeof(int));
    ptr += sizeof(int);

    memcpy(&p->cell_owner,   ptr, sizeof(int));
    ptr += sizeof(int);
    memcpy(&p->block_owner,  ptr, sizeof(int));
    ptr += sizeof(int);

    memcpy(p->position,      ptr, 3*sizeof(double));
    ptr += 3*sizeof(double);
    memcpy(p->orient,        ptr, 3*sizeof(double));
    ptr += 3*sizeof(double);

    memcpy(&p->num_neighbs,  ptr, sizeof(int));
    ptr += sizeof(int);
    
    memcpy(&p->max_neighbs,  ptr, sizeof(int));
    ptr += sizeof(int);

    // Allocate neighbor list
    if (p->neighbs_index != NULL) {
        free(p->neighbs_index);
        p->neighbs_index = NULL; 
    }

    p->neighbs_index = malloc(p->max_neighbs * sizeof(int));
    if (!p->neighbs_index) {
        printf("\nERROR at doing 'unpack_particle' when 'malloc'\n");
        return -1;
    }

    memcpy(p->neighbs_index, ptr, p->max_neighbs*sizeof(int)); 
    ptr += p->max_neighbs * sizeof(int);

    // Return total bytes consumed
    return (int)(ptr - buf);
}

/* Copy particle info from src to dst */
void copy_particle(Particle* dst, Particle* src) {

    // Pack source particle
    int bytes;
    char* buf = pack_particle(src, &bytes);

    // Unpack into destination particle
    unpack_particle(dst, buf);

    // Free buffer
    free(buf);

    return;
}


/* Distribute particles to batches */
void distr_particles_to_batches(Particle* box_particles, 
                                Particle** batch_matrix, 
                                int* particles_per_batch, 
                                double* block_size, int* grid_dims, int n_ranks) {

    int i;

    int block_owner;
    int cell_owner;

    // Initialize the number of particles per batch to zero
    for (int i = 0; i < n_ranks; i++) particles_per_batch[i] = 0;

    // Distribute particles to each batch
    for (int i = 0; i < num_particles; i++) {

        // Get particle owners
        get_particle_owners(box_particles[i].position, 
                            &block_owner, &cell_owner, 
                            block_size, grid_dims);

        // if (i < 30) {
        //     printf("Particle %d positions       (%.2lf, %.2lf, %.2lf)\n", i, box_particles[i].position[0], box_particles[i].position[1], box_particles[i].position[2]);
        //     printf("Particle %d belongs to rank %d\n", i, block_owner);
        // }

        box_particles[i].block_owner = block_owner;
        box_particles[i].cell_owner  = cell_owner;

        copy_particle(&batch_matrix[block_owner][particles_per_batch[block_owner]], &box_particles[i]);

        particles_per_batch[block_owner]++;
    }

    return;
}


/* Pack particle array
 * @param particle_array: array of particles to pack
 * @param n_particles: number of particles in the array
 * @param total_bytes_out: pointer to store the total number of bytes
 * @return: pointer to the packed particle array
*/
char* pack_particle_array(Particle* particle_array, int n_particles, int* total_bytes_out) {
    
    int i;
    int total_bytes = 0;

    // First pack each particle individually to compute total_bytes size
    int* particle_bytes = calloc(n_particles, sizeof(int));
    if (!particle_bytes) {
        printf("\nERROR at doing 'pack_particle_array' when 'calloc particle_bytes'\n");
        return NULL;
    }

    // Allocate particle buffers for each particle
    char** particle_buffers = malloc(n_particles * sizeof(char*));
    if (!particle_buffers) {
        printf("\nERROR at doing 'pack_particle_array' when 'malloc particle_buffers'\n");
        
        // Free particle_bytes if it was allocated
        free(particle_bytes); particle_bytes = NULL;
        
        return NULL;
    }

    // Pack each particle and compute total_bytes
    for (i = 0; i < n_particles; i++) {
        
        particle_buffers[i] = pack_particle(&particle_array[i], &particle_bytes[i]);
        
        if (!particle_buffers[i]) {
            printf("\nERROR at doing 'pack_particle_array' at 'particle_buffers[%d]'\n", i);
            
            // Free particle_bytes if it was allocated
            free(particle_bytes); particle_bytes = NULL;
            // Free particle_buffers if it was allocated
            free(particle_buffers); particle_buffers = NULL;
            
            return NULL;
        }

        total_bytes += particle_bytes[i];
    }

    // Allocate the combined buffer
    char* buffer = malloc(total_bytes);
    if (!buffer) {
        printf("\nERROR at doing 'pack_particle_array' when 'malloc buffer'\n");
        
        // Free particle_bytes if it was allocated
        free(particle_bytes); particle_bytes = NULL;
        // Free particle_buffers if it was allocated
        free(particle_buffers); particle_buffers = NULL;
        
        return NULL;
    }

    // Copy all packed particles into it
    int offset = 0;
    for (i = 0; i < n_particles; i++) {
        memcpy(buffer + offset, particle_buffers[i], particle_bytes[i]);
        offset += particle_bytes[i];
        free(particle_buffers[i]);
    }

    // Free particle buffers
    free(particle_buffers); particle_buffers = NULL;
    free(particle_bytes); particle_bytes = NULL;

    *total_bytes_out = total_bytes;
    
    return buffer;
}

/* Unpack particle array 
 * @param particle_array: array of particles to unpack
 * @param n_particles: number of particles in the array
 * @param buffer: buffer containing the packed particle array
*/
int unpack_particle_array(Particle* particle_array, int n_particles, char* buffer) {
    
    int offset = 0;

    // Unpack each particle
    for (int i = 0; i < n_particles; i++) {
        int bytes = unpack_particle(&particle_array[i], buffer + offset);
        offset += bytes;
    }

    return offset;
}

/* Pack cell 
 * @param c: cell to pack
 * @param out_bytes: pointer to store the total number of bytes
 * @return: pointer to the packed cell
*/
char* pack_cell(const Cell* c, int* out_bytes) {

    int metadata_size = 
        2 * sizeof(int) +       // index, block_owner
        2 * sizeof(int) +       // capacity, num_particles
        3 * sizeof(double) +    // size[3]
        3 * sizeof(int) +       // coords[3]
        6 * sizeof(double) +    // domain_min[3], domain_max[3]
        1 * sizeof(int);        // is_reset

    // Particles array size calculation
    int particles_array_bytes = 0;
    
    // We need a temporary buffer for packed particles to know the size
    char* packed_particles = NULL;
    
    // Pack particles data
    // We pack only VALID particles (num_particles), not the whole capacity
    if (c->num_particles > 0) {
        packed_particles = pack_particle_array(c->particles, c->num_particles, &particles_array_bytes);
        if (!packed_particles) {
            printf("\nERROR in 'pack_cell': pack_particle_array failed\n");
            return NULL;
        }
    }

    // Particles index size
    int particles_index_bytes = c->num_particles * sizeof(int);

    int total_bytes = metadata_size + particles_array_bytes + particles_index_bytes;
    *out_bytes = total_bytes;

    char* buf = malloc(total_bytes);
    if (!buf) {
        printf("\nERROR in 'pack_cell': buffer malloc failed\n");
        
        // Free packed particles if they were allocated
        if (packed_particles) free(packed_particles);
        
        return NULL;
    }

    // Buffer pointer (to keep track of where we are in the buffer)
    char* ptr = buf;

    // Pack metadata
    memcpy(ptr, &c->index,         sizeof(int));     ptr += sizeof(int);
    memcpy(ptr, &c->block_owner,   sizeof(int));     ptr += sizeof(int);
    memcpy(ptr, &c->capacity,      sizeof(int));     ptr += sizeof(int);
    memcpy(ptr, &c->num_particles, sizeof(int));     ptr += sizeof(int);
    
    memcpy(ptr, c->size,       3 * sizeof(double));  ptr += 3 * sizeof(double);
    memcpy(ptr, c->coords,     3 * sizeof(int));     ptr += 3 * sizeof(int);
    memcpy(ptr, c->domain_min, 3 * sizeof(double));  ptr += 3 * sizeof(double);
    memcpy(ptr, c->domain_max, 3 * sizeof(double));  ptr += 3 * sizeof(double);
    
    memcpy(ptr, &c->is_reset,      sizeof(int));     ptr += sizeof(int);

    // Pack particles data
    if (c->num_particles > 0) {
        memcpy(ptr, c->particles_index, particles_index_bytes);
        ptr += particles_index_bytes;
        
        memcpy(ptr, packed_particles, particles_array_bytes);
        ptr += particles_array_bytes;

        free(packed_particles);
    }

    return buf;
}

int unpack_cell(Cell* c, const char* buf) {

    const char* ptr = buf;

    // Save old number of particles to free memory if needed
    if (!c->is_reset) {
        reset_cell(&c);
    }

    // Unpack metadata
    memcpy(&c->index,         ptr,     sizeof(int));     ptr += sizeof(int);
    memcpy(&c->block_owner,   ptr,     sizeof(int));     ptr += sizeof(int);
    memcpy(&c->capacity,      ptr,     sizeof(int));     ptr += sizeof(int);
    memcpy(&c->num_particles, ptr,     sizeof(int));     ptr += sizeof(int);

    memcpy(c->size,           ptr, 3 * sizeof(double));  ptr += 3 * sizeof(double);
    memcpy(c->coords,         ptr, 3 * sizeof(int));     ptr += 3 * sizeof(int);
    memcpy(c->domain_min,     ptr, 3 * sizeof(double));  ptr += 3 * sizeof(double);
    memcpy(c->domain_max,     ptr, 3 * sizeof(double));  ptr += 3 * sizeof(double);

    memcpy(&c->is_reset,     ptr,      sizeof(int));     ptr += sizeof(int);

    // Unpack particles data and particles index
    if (c->num_particles > 0) {
        int particles_index_bytes = c->num_particles * sizeof(int);
        memcpy(c->particles_index, ptr, particles_index_bytes);
        ptr += particles_index_bytes;

        int particles_bytes = unpack_particle_array(c->particles, c->num_particles, (char*)ptr);
        ptr += particles_bytes;
    }
    
    return (int)(ptr - buf);
}

void dist_batch_to_cells(Block* block) {

    int i, j, k;

    reset_block_cells(block);

    for (i = 0; i < block->num_particles; i++) {

        int cx, cy, cz;

        index_to_cell_coords(block->batch[i].cell_owner, &cx, &cy, &cz);

        Cell* cell =  &block->local_cells[cx][cy][cz];

        copy_particle(&cell->particles[cell->num_particles], 
                      &block->batch[i]);        

        cell->particles_index[cell->num_particles] = block->batch[i].global_index;
        cell->num_particles++;
    }

    return;
}

void reset_block_cells(Block* block) {

    int i;
    int x_coords, y_coords, z_coords;
    
    // Initialize local subcells A-H
    for (z_coords = 0; z_coords < 2; z_coords++) {
        for (y_coords = 0; y_coords < 2; y_coords++) {
            for (x_coords = 0; x_coords < 2; x_coords++) {

                // Pass the address of each local_cell to a cell struct
                // The local cells will remain untouch unless updated by update_cell function
                // Cell *cell = &block->local_cells[x_coords][y_coords][z_coords];
                reset_cell(&block->local_cells[x_coords][y_coords][z_coords]);
            }
        }
    }

    // Initialize extended cell grid (local and ghost cells)
    for (z_coords = 0; z_coords < 3; z_coords++) {
        for (y_coords = 0; y_coords < 3; y_coords++) {
            for (x_coords = 0; x_coords < 3; x_coords++) {

                // Pass the address of each local and ghost cell to a cell struct
                // Cell *cell = &block->extended_grid[x_coords][y_coords][z_coords];
                reset_cell(&block->extended_grid[x_coords][y_coords][z_coords]);
            }
        }
    }

}

void reset_cell(Cell* cell) {

    cell->num_particles = 0;

    // Set the particle index array to zeros
    memset(cell->particles_index, 0, cell->capacity * sizeof(int));

    // Free and allocate new memory to particle array
    // Free memory
    for (int i = 0; i < cell->capacity; i++) {
        if (&cell->particles[i].neighbs_index != NULL) {
            particle_free(&cell->particles[i]);
            particle_plain_init(&cell->particles[i]);
        }
    }

    // Active cell = False
    cell->is_reset = 1;

    return;
}

void copy_cell(Cell* dest, const Cell* src) {

    // Pack source cell
    int bytes;
    char* buf = pack_cell(src, &bytes);

    // Unpack into destination cell
    unpack_cell(dst, buf);

    // Free buffer
    free(buf);

    return;
}

void clean_neighbor_cells(Block* block) {
    // Free extended cell grid (local and ghost cells)
    for (int z = 0; z < 3; z++) {
        for (int y = 0; y < 3; y++) {
            for (int x = 0; x < 3; x++) {
                Cell *cell = &block->extended_grid[x][y][z];
                // Use reset_cell which handles freeing and zeroing
                reset_cell(cell);
            }
        }
    }
}

void build_neighbor_cells(Block* block, int active_cell_index, int grid_dims[3]) {
    
    // 1. Identify active coordinates in local grid (0 or 1)
    int active_cell_x, active_cell_y, active_cell_z;
    index_to_cell_coords(active_cell_index, &active_cell_x, &active_cell_y, &active_cell_z);
    
    // We map 3x3x3 extended grid indices [0..2] to relative directions [-1..1]
    // extended[1][1][1] is always the center (active cell location in extended grid)

    // Buffers and Request handling
    // 27 sends + 27 recvs, we know that there will less but let's keep it simple
    MPI_Request size_reqs[54];
    MPI_Request data_reqs[54];
    
    // Request counters
    int n_size_reqs = 0;
    int n_data_reqs = 0;

    // Send/Receive sizes
    // This arrays indicate the number of bytes to send/receive for each neighbor cell
    int send_sizes[26];
    int recv_sizes[26];
    
    // Send/Receive buffers
    // These arrays store the actual data to be sent/received
    char* send_buffers[26];
    char* recv_buffers[26];
    
    // Pointers to cells we are sending
    // These arrays store the actual data to be sent/received
    Cell* send_cells[26]; 

    // Initialize buffers and cells to NULL
    for(int i=0; i<26; i++) {
        send_buffers[i] = NULL;
        recv_buffers[i] = NULL;
        send_cells[i]   = NULL;
    }

    // Loop over 26 neighbors to send/receive cells to/from neighbors 
    for (int offset_z = -1; offset_z <= 1; offset_z++) {
        for (int offset_y = -1; offset_y <= 1; offset_y++) {
             for (int offset_x = -1; offset_x <= 1; offset_x++) {
                 
                // Skip the active cell
                if (offset_x == 0 && offset_y == 0 && offset_z == 0) continue;
   
                // Index for each cell in the 3x3x3 extended grid
                int neighbor_idx = (offset_x+1) + (offset_y+1)*3 + (offset_z+1)*9;
                
                // Target coordinates (relative to active cell) in LOCAL BLOCK frame
                int target_rel_cell_coords[3] = {active_cell_x + offset_x,
                                                 active_cell_y + offset_y,
                                                 active_cell_z + offset_z};
                
                // Target coordinates (local block frame) in LOCAL BLOCK frame
                int target_loc_cell_coords[3] = {mod(target_rel_cell_coords[0], 2),
                                                 mod(target_rel_cell_coords[1], 2),
                                                 mod(target_rel_cell_coords[2], 2)};
                // --- RECV LOGIC ---
                // If target is outside local block [0,1], it's a Ghost -> RECV from neighbor block (rank)
                int is_ghost = (target_rel_cell_coords[0] < 0 || target_rel_cell_coords[0] > 1 || 
                                target_rel_cell_coords[1] < 0 || target_rel_cell_coords[1] > 1 || 
                                target_rel_cell_coords[2] < 0 || target_rel_cell_coords[2] > 1);

                // Initialize owner block coordinates and block offset
                int owner_block_coords[3] = {0, 0, 0};
                int block_offset[3] = {0, 0, 0};
                
                if (is_ghost) {

                    // Determine owner block (rank) of this ghost cell
                    int block_owner = get_cell_owner_full(target_rel_cell_coords, 
                                                          block->coords, 
                                                          owner_block_coords,
                                                          block_offset,
                                                          grid_dims, 
                                                          2);

                    
                     
                    // Irecv Size from neighbor block (rank)
                    MPI_Irecv(&recv_sizes[neighbor_idx], 
                              1, 
                              MPI_INT, 
                              block_owner, 
                              0, 
                              MPI_COMM_WORLD, 
                              &size_reqs[n_size_reqs++]);
                }
                
                // --- SEND LOGIC ---
                // If I needed the cell [x, y, z] (ghost cell) from block (x+offset_x, y+offset_y, z+offset_z)
                // then the block (x-offset_x, y-offset_y, z-offset_z) needs my cell [x, y, z].
                
                
                
                int source_cell_x = active_cell_x - offset_x;
                int source_cell_y = active_cell_y - offset_y;
                int source_cell_z = active_cell_z - offset_z;
                
                int neighbor_needs_me = (source_cell_x < 0 || source_cell_x > 1 || 
                                         source_cell_y < 0 || source_cell_y > 1 || 
                                         source_cell_z < 0 || source_cell_z > 1);
                
                if (neighbor_needs_me) {
                    // Rank of neighbor in direction (dx, dy, dz)
                    int neighbor_block_coords[3] = {
                        mod(block->coords[0] + source_cell_x, grid_dims[0]),
                        mod(block->coords[1] + source_cell_y, grid_dims[1]),
                        mod(block->coords[2] + source_cell_z, grid_dims[2])
                    };
                    int neighbor_rank = get_block_index(neighbor_block_coords, grid_dims);
                    
                    // Cell to send is at (active + d).
                    // This guarantees local index is [0,1] because width=2.
                    Cell* cell_to_send = &block->local_cells[active_cell_x+offset_x][active_cell_y+offset_y][active_cell_z+offset_z];
                    send_cells[neighbor_idx] = cell_to_send;
                    
                    // Pack Size
                    send_buffers[neighbor_idx] = pack_cell(cell_to_send, &send_sizes[neighbor_idx]);
                    
                    // Isend Size
                    MPI_Isend(&send_sizes[neighbor_idx], 1, MPI_INT, neighbor_rank, 0, MPI_COMM_WORLD, &size_reqs[n_size_reqs++]);
                }
                
                neighbor_idx++;
             }
        }
    }
    
    // Wait for Size Exchange
    if (n_size_reqs > 0) {
        MPI_Waitall(n_size_reqs, size_reqs, MPI_STATUSES_IGNORE);
    }
    
    // Phase 2: Data Exchange and Local Copy
    neighbor_idx = 0;
    
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
             for (int dx = -1; dx <= 1; dx++) {
                 if (dx == 0 && dy == 0 && dz == 0) {
                     // CENTER CELL: Copy Local Active Cell
                     // extended_grid[1][1][1] = local_cells[ax][ay][az]
                     copy_cell(&block->extended_grid[1][1][1], &block->local_cells[active_cell_x][active_cell_y][active_cell_z]);
                     continue;
                 }
                 
                 int tx = active_cell_x + dx;
                 int ty = active_cell_y + dy;
                 int tz = active_cell_z + dz;
                 int is_ghost = (tx < 0 || tx > 1 || ty < 0 || ty > 1 || tz < 0 || tz > 1);
                 
                 // RECV DATA
                 if (is_ghost) {
                     int target_cell_coords[3] = {tx, ty, tz};
                     int owner_rank = get_cell_owner(target_cell_coords, block->coords, grid_dims, 2);
                     
                     recv_buffers[neighbor_idx] = malloc(recv_sizes[neighbor_idx]);
                     MPI_Irecv(recv_buffers[neighbor_idx], recv_sizes[neighbor_idx], MPI_BYTE, owner_rank, 1, MPI_COMM_WORLD, &data_reqs[n_data_reqs++]);
                 } else {
                     // Local Copy for Neighbors inside block
                     // extended_grid[dx+1][dy+1][dz+1] gets local_cells[tx][ty][tz]
                     copy_cell(&block->extended_grid[dx+1][dy+1][dz+1], &block->local_cells[tx][ty][tz]);
                 }
                 
                 // SEND DATA
                 if (send_cells[neighbor_idx] != NULL) {
                      int neighbor_block_coords[3] = {
                         mod(block->coords[0] + dx, grid_dims[0]),
                         mod(block->coords[1] + dy, grid_dims[1]),
                         mod(block->coords[2] + dz, grid_dims[2])
                     };
                     int neighbor_rank = get_block_index(neighbor_block_coords, grid_dims);
                     
                     MPI_Isend(send_buffers[neighbor_idx], send_sizes[neighbor_idx], MPI_BYTE, neighbor_rank, 1, MPI_COMM_WORLD, &data_reqs[n_data_reqs++]);
                 }
                 
                 neighbor_idx++;
             }
        }
    }
    
    // Wait for Data Exchange
    if (n_data_reqs > 0) {
        MPI_Waitall(n_data_reqs, data_reqs, MPI_STATUSES_IGNORE);
    }
    
    // Phase 3: Unpack Received Data
    neighbor_idx = 0;
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
             for (int dx = -1; dx <= 1; dx++) {
                 if (dx == 0 && dy == 0 && dz == 0) continue;
                 
                 int tx = active_cell_x + dx;
                 int ty = active_cell_y + dy;
                 int tz = active_cell_z + dz;
                 int is_ghost = (tx < 0 || tx > 1 || ty < 0 || ty > 1 || tz < 0 || tz > 1);
                 
                 if (is_ghost) {
                     Cell* ghost_cell = &block->extended_grid[dx+1][dy+1][dz+1];
                     unpack_cell(ghost_cell, recv_buffers[neighbor_idx]);
                     
                     free(recv_buffers[neighbor_idx]);
                 }
                 
                 if (send_buffers[neighbor_idx] != NULL) {
                     free(send_buffers[neighbor_idx]);
                 }
                 neighbor_idx++;
             }
        }
    }

}

// typedef struct {

//     int rank;

//     // MPI block information
//     int capacity;

//     double size[3];             // Size of the block
//     int coords[3];              // Coordinates in the MPI grid
//     double domain_min[3];       // Minimum coordinates of this block
//     double domain_max[3];       // Maximum coordinates of this block
    
//     // Cells information
//     int cells_index[8];         // Index of each cell, it indicates the order to perform each sweep
//     double cell_size[3];        // Size of each cell in x,y,z

//     int are_cells_free;         // Flag to indicate if cells are correctly allocated and initialized

//     // The 2×2×2 local cells (0, 1, ..., 7)
//     // Indexed by [x][y][z] where each is 0 or 1
//     Cell local_cells[2][2][2];  
    
//     // The 3×3×3 structure including ghost neighbors
//     // Indexed by [x][y][z] where each is -1, 0 or 1
//     Cell extended_grid[3][3][3];

// } Block;


// typedef struct {

//     // Subcell metadata
//     int index;                 // Local index within MPI block (0-7 for A-H)
//     int block_owner;            // Rank which this cell belongs

//     int capacity;              // Maximum number of particles per cell
//     int num_particles;         // Current number of particles

//     double size[3];             // Size of the cell
//     int    coords[3];           // Local coordinates in the block
//     double domain_min[3];       // Minimum coordinates of this block
//     double domain_max[3];       // Maximum coordinates of this block

//     int is_reset;              // Is this the currently active cell?

//     // Particle data for THIS cell
//     Particle* particles;       // Dynamic array for particles in the cell
//     int* particles_index;      // Particle indexes of the particles in the cell

// } Cell;

// typedef struct {
//     // Particle data for a single particle
//     int global_index;   // Particle index in the whole system
//     int local_index;    // Particle index inside its block
    
//     double position[3]; // Particle position (x, y, z)
//     double orient[3];   // Particle orientation (x, y, z)e

//     int num_neighbs;    // number of neighbours
//     int max_neighbs;    // maximum number of neighbours per particle

//     int* neighbs_index; // neighbours indexes

// } Particle;