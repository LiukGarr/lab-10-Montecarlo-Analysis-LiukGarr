
# Use this file to define all your input parameters (e.g. lengths, loss coefficient, etc.)
# c = speed of light
c = 3e8

# BER = Bit Error Rate
BER_t = 10e-3

# R_s = Symbol Rate
R_s = 32e9  # Hz

# B_n = Noise Bandwidth
B_n = 12.5e9  # Hz

# f = Frequency
f = 193.414e12

# Space_fact = Space from an amplifier to another
Space_fact = 80e3

# Loss_dB
# alpha_dB = 0.2  # dB/km
alpha_dB = 2e-4  # dB/m

# Beta square
# b2 = 2.13e-26   # ps^2/km
b2 = abs(2.13e-27)  # ps^2/m

# Gamma
gam = 1.27e-3     # (Wm)^-1

# df = frequency spacing
df = 50e9

# CT_coef = Crosstalk coefficient
CT_coef = (R_s/df) * 1e-4
