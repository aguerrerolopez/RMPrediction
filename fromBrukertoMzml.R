###############################################################################
##
## MALDI-TOF Translator from Bruker to mzML
##
## Alejandro Guerrero-López
##
###############################################################################
args <- commandArgs(trailingOnly = TRUE)
###############################################################################
## Load libraries
## MALDIquant()
## MALDIquantForeign()
##
# install.packages(c("MALDIquant","MALDIquantForeign"))
###############################################################################

library("MALDIquant")
library("MALDIquantForeign")

###############################################################################
## Load data
###############################################################################
path <- "/Users/alexjorguer/Downloads/cepas"
path_load <- paste0(path, '/Cepas_Analizadas_Sensibles')
# path_load <- paste0(path, args[1])
spectra1 <- importBrukerFlex(path_load)

###############################################################################
## Save data
###############################################################################
path_save <- paste0(path, '/mzml/')
# path_save <- paste0(path, args[2])
## Save Rdata object
save(spectra1, file='mzml')
# save(spectra1, file=args[2])

## Export
exportMzMl(spectra1, path=path_save)


