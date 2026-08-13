&ctl_nl
vthreads          = 1
NThreads          = 1
partmethod        = 4
topology          = "cube"
test_case         = "jw_baroclinic"
u_perturb         = 1
rel_perturb       = 0
rotate_grid       = 0
ne                = 2
qsize             = 4
nmax              = 1
statefreq         = 9999
restartfreq       = 43200
restartfile       = "./R0001"
runtype           = 0
mesh_file         = '/dev/null'
tstep             = 300
rsplit            = 2
qsplit            = 1
integration       = "explicit"
smooth            = 0
nu                = 7e15
nu_div            = 1e15
nu_p              = 7e15
nu_q              = 7e15
nu_s              =-1
nu_top            = 2.5e5
se_ftype          = 0
limiter_option    = 9
vert_remap_q_alg  = 1
hypervis_scaling  = 0
hypervis_order    = 2
hypervis_subcycle = 1
hypervis_subcycle_tom  = 0
theta_hydrostatic_mode = false
theta_advect_form = 1
tstep_type        = 10
/
&solver_nl
precon_method = "identity"
maxits        = 500
tol           = 1.e-9
/
&filter_nl
filter_type   = "taylor"
transfer_type = "bv"
filter_freq   = 0
filter_mu     = 0.04D0
p_bv          = 12.0D0
s_bv          = .666666666666666666D0
wght_fm       = 0.10D0
kcut_fm       = 2
/
&vert_nl
vfile_mid = './vcoord/sabm-128.ascii'
vfile_int = './vcoord/sabi-128.ascii'
/

&prof_inparm
profile_outpe_num   = 100
profile_single_file = .true.
/

!  timunits: 0= steps, 1=days, 2=hours
&analysis_nl
 interp_gridtype   = 2
 output_dir        = 'output/'
 output_prefix     = 'base_'
 output_timeunits  = 0,0
 output_frequency  = 1,1
 output_start_time = 0,0
 output_end_time   = 30000,30000
 output_varnames1  = 'u','v','Th'
 output_varnames2  = 'u_sens','v_sens','Th_sens'
 io_stride         = 8
 output_type       = 'netcdf'
/
