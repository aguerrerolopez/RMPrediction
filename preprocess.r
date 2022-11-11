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

##### PREPROCESS

#Step 1: the measured intensity is transformed with a square-root method to stabilize the variance
spectra <- transformIntensity(spectra1, method="sqrt"),
# Step 2: smoothing using the Savitzky–Golay algorithm with half-window-size 5 is applied
spectra <- smoothIntensity(spectra, method="SavitzkyGolay", halfWindowSize=5)
# Step 3: an estimate of the baseline is removed in 20 iterations of the SNIP algorithm
spectra <- removeBaseline(spectra, method="SNIP", iterations=20)
# Step 4: the intensity is calibrated using the total ion current (TIC)
spectra <- calibrateIntensity(spectra, method="TIC")

# peaks <- detectPeaks(spectra, method="MAD", halfWindowSize=20, SNR=2)
# peaks <- binPeaks(peaks, tolerance=0.004)
# peaks <- filterPeaks(peaks, minFrequency=0.9)


###############################################################################
## Save data
###############################################################################
path_save <- paste0(path, '/mzml/')
# path_save <- paste0(path, args[2])
## Save Rdata object
save(spectra, file='mzml')
# save(spectra1, file=args[2])

## Export
exportMzMl(spectra, path=path_save)
