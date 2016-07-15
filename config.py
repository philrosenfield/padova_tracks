# from low mass and below XCEN = 0.3 for MS_TMIN
low_mass = 1.25

# inte_mass is where SG_MAXL becomes truly meaningless
inte_mass = 12.

# from high mass and above find MS_BEG in this code
high_mass = 19.

# for low mass stars with no MS_TO listed, where to place it (and ms_tmin)
max_age = 10e10

# file column names. Classic PARSEC
xcen = u'XCEN'
ycen = u'YCEN'
MODE = u'MODE'

# PARSEC online (Sandro's website)
xcen = u'H_CEN'
ycen = u'HE_CEN'
MODE = u'MODELL'

# only used in tracks.track.calc_core_mu
xc_cen = u'XC_cen'
xo_cen = u'XO_cen'

# plot extension
EXT = u'.png'

# what you would like them to be:
logL = u'logL'
logT = u'logT'
age = u'age'
mass = u'mass'
