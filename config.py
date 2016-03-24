# from low mass and below XCEN = 0.3 for MS_TMIN
low_mass = 1.25

# inte_mass is where SG_MAXL becomes truly meaningless
inte_mass = 12.

# from high mass and above find MS_BEG in this code
high_mass = 19.

# for low mass stars with no MS_TO listed, where to place it (and ms_tmin)
max_age = 10e10

# file column names. Classic PARSEC
#XCEN = 'XCEN'
#YCEN = 'YCEN'
#MODE = 'MODE'

# PARSEC online (Sandro's website)
XCEN = 'H_CEN'
YCEN = 'HE_CEN'
MODE = 'MODELL'

# only used in tracks.track.calc_core_mu
XC_cen = 'XC_cen'
XO_cen = 'XO_cen'

# plot extension
EXT = '.png'
