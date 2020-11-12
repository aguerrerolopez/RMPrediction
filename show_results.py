import utils
import glob
import preprocess as pp

# Load data
data_path = "./data/old_data/"
data = pp.load_data(path=data_path, format="zip")

# Preprocess data
x, x_axis, y, labels_name = pp.prepare_data(data, drop=True, columns_drop=["AMPICILINA"],
                                            impute=True, limits=[2005,19805], n_labels=18)


# ======== Parameters to plot and store ========
# If you want the results to be plot
plot = True
# If you want to store the plots
store = True
# If you want to print the results
print_r = True

for file in glob.glob("./Results/*.pkl"):
    results = utils.load_results(main_folder=file, labels=labels_name.tolist(), file=file, plot=plot, print_r=print_r, store=store)

