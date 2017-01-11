# Constants used in the package.

# Filter initial weights options.
init_weight_opts = {'random': 0, 'zeros': 1}

# LMS adaptive filter
MU_LMS = 0.01
MU_LMS_MIN = 0.0
MU_LMS_MAX = 1000.0

# RLS adaptive filter
DELTA_RLS = 0.004
DELTA_RLS_MIN = 0.0
DELTA_RLS_MAX = 1.0

LAMDA_RLS = 0.99
LAMDA_RLS_MIN = 0.0
LAMDA_RLS_MAX = 1.0
