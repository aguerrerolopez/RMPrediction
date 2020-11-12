import numpy as np

def load_data(path, format=None, train_test_split=True):


    if format == "mzml":
        from pyteomics import mzml
        from os import listdir
        files = [f for f in listdir(path)]
        data_mz = [None] * len(files)
        data_int = [None] * len(files)
        data_file = [None] * len(files)
        for i, file in enumerate(files):
            print(file)
            t = mzml.read(path + file)
            a = next(t)
            data_mz[i] = a["m/z array"]
            data_int[i] = a["intensity array"]
            data_file[i] = file

    if format == "zip":
        import zipfile
        import pickle

        # zf = zipfile.ZipFile(path + 'zipped_TrainData.zip', 'r')
        # df_train = pickle.loads(zf.open('TrainData.pkl').read())
        # zf.close()
        #
        # # Read Test data
        # zf = zipfile.ZipFile(path + 'zipped_TestData.zip', 'r')
        # df_test = pickle.loads(zf.open('TestData.pkl').read())
        # zf.close()
        zf = zipfile.ZipFile(path + 'zipped_Data.zip', 'r')
        df_data = pickle.loads(zf.open('Data.pkl').read())
        zf.close()

        return df_data


def prepare_data(df, impute=False, drop=False, columns_drop=None, n_labels=0, remove_baseline=False, normalize=None,
                      limits = [0, 20000]):
    """
    :param df: Dataframe object. Data to clean and prepare.
    :param impute: boolean. Indicates if the missing labels have to be imputed.
    :param drop: boolean. Indicates if some columns have to be dropped.
    :param columns_drop: list. Only needed if "drop" is True. List of columns to drop.
    :param n_labels: int. Number of labels on the dataframe.
    :param remove_baseline: boolean. Indicates if the baseline noise has to be removed.
    :param normalize: str. If "TIC" the Total Ion Current method is used. If "median" the median normalization
    is computed. If None just a standard scaler is used.
    :param limits: tuple of ints. m/z limits to keep. Default [0, 20k].
    :return: ndarray, ndarray, ndarray, the spectrum of all the data, a common x axis for all of them, labels of each spectrum.
    """

    from sklearn.preprocessing import StandardScaler

    # Extract the labels values and drop the ones you dont need.
    labels_name = df.columns[-(n_labels + 1):-1]
    y = df[labels_name]
    if drop:
        y = y.drop(columns=columns_drop)
        labels_name = y.columns
        y = y.to_numpy()
    else:
        labels_name = y.columns
        y = y.to_numpy()

    if impute:
        from impyute.imputation.cs import fast_knn
        imputed_knn = fast_knn(y, 30)
        imputed_labels = imputed_knn > 0.95
        y = imputed_labels.astype(int)


    # Extract the intensity values and align them all.
    spectra_raw, coord_mz = equal_mz(df, samples=1, limits=limits)
    spectra_norm = spectra_raw.copy()
    if normalize == "TIC":
        for i in range(spectra_norm.shape[0]):
            TIC = np.sum(spectra_raw[i])
            spectra_norm[i] = spectra_raw[i]/TIC
    if normalize == "median":
        for i in range(spectra_raw.shape[0]):
            spectra_norm[i] /= np.median(spectra_raw[i])

    if remove_baseline:
        from scipy.signal import savgol_filter
        from BaselineRemoval import BaselineRemoval
        polynomial_degree = 15
        intensity_to_align_array = np.zeros(spectra_norm.shape)
        for line in range(spectra_norm.shape[0]):
            # Smoothing
            yhat = savgol_filter(spectra_norm[line, :], 51, 3)  # window size 51, polynomial order 3
            # Baseline removal
            baseObj = BaselineRemoval(yhat)
            intensity_to_align_array[line, :] = baseObj.IModPoly(polynomial_degree)

        spectra_raw = peak_alineation(intensity_to_align_array, coord_mz)

    scaler = StandardScaler()
    spectra = scaler.fit_transform(spectra_raw.copy())

    return spectra, coord_mz, y, labels_name

# TODO:eliminar fors de esta función, son innecesarios
def peak_alineation(spectra, new_coord_mz_axis):
    from msalign import Aligner
    from scipy.signal import find_peaks
    # compute mean spectrum
    mean_spectrum = np.mean(spectra, axis=0)
    # compute variance along spectra
    var_spectrum = np.var(spectra, axis=0)
    # Find peaks in mean spectrum
    peaks, _ = find_peaks(mean_spectrum, height = 0.00001,  distance=70)
    # Select those peaks with low variance along spectra
    no_var_peaks = []
    # TODO: Este for seguro que se puede quitar por un np.where
    for peak_index in peaks:
        if var_spectrum[peak_index]<0.0007:
            no_var_peaks.append(peak_index)

    selected_pos = [no_var_peaks[i] for i in np.argsort(var_spectrum[no_var_peaks])[:4]]
    selected_peaks = new_coord_mz_axis[selected_pos]
    # Align the spectra
    # instantiate aligner object
    aligner = Aligner(
        new_coord_mz_axis,
        spectra.T,
        np.sort(selected_peaks),
        return_shifts=True,
        align_by_index=True,
        only_shift= False,
        method="cubic", ## Cubic es más rápido que pchip
        iterations=5
    ).run()
    aligned_array, shifts_out = aligner.align()

    return aligned_array

# TODO:eliminar fors de esta función, son innecesarios
def equal_mz(df, samples=1, limits=[0, 20000]):
    from scipy import interpolate as interp

    spectra = df['intensity'].values
    coord_mz = df['mz_coords'].values
    # Check if all the spectra can be interpolated to the nes limits
    all_values = []
    max_val = 20000
    min_val = 0
    for i in range(0, spectra.shape[0]):
        arr = np.round(coord_mz[i], decimals=2)
        aux = np.concatenate((all_values, arr))
        all_values = np.unique(aux)
        if np.amax(arr) < max_val:  # Find the smallest maximun value
            max_val = np.amax(arr)
        if np.amin(arr) > min_val:  # Find the bigger minimun value
            min_val = np.amin(arr)

    if (min_val > limits[0]) | (max_val < limits[1]):
        limits[0] = np.max(min_val, limits[0])
        limits[1] = np.max(max_val, limits[1])
        print('Interpolation in the limits selected cannot be accomplished')
        print('New interpolation limits are', limits)

    # Create the new axis and interpolate
    new_mz_axis = np.arange(round(limits[0]), round(limits[1]), samples)
    interpolated_spectra = np.zeros((spectra.shape[0], new_mz_axis.shape[0]))
    for i in range(0, spectra.shape[0]):
        my_interp = interp.interp1d(coord_mz[i], spectra[i], 'zero')
        interpolated_spectra[i, :] = my_interp(new_mz_axis)

    X = interpolated_spectra
    return X, new_mz_axis



# df_train, df_test = load_data(path = "./data/old_data/", format="zip")
# x_tr, x_axis, y_tr = prepare_data(df=df_train, impute=True, n_labels=18, limits=[2005, 19805])