from scipy.special import jv, iv
from scipy.special import jn_zeros

# Precompute Bessel function zeros and values for first-passage time distribution
zeros_0 = jn_zeros(0, 100)
JVZ1 = jv(1, zeros_0)

zeros_1 = jn_zeros(1, 100)
JVZ2 = jv(2, zeros_1)