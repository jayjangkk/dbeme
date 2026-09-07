import pickle
import importlib.util
from itertools import product
from copy import deepcopy
from collections import OrderedDict
import os

import numpy as np
from tqdm import tqdm

from . import overlap_calculation_tool as oct
from .dataset_identity import fingerprint, verify
from ..fde.assemble import assemble, overlap_matrix, prop_axis_index

# ray.init(object_store_memory=2e9, ignore_reinit_error=True)
class TestModeError(Exception):
    pass

class DataUpdater:
    r"""Update and retrieve precomputed electromagnetic datasets.

    The updater drives an :class:`~em_simulation.fde.base.FDEBackend` (by
    default emepy's finite-difference mode solver) to populate a set of pickled
    dictionaries.  The upstream implementation used the Lumerical/ANSYS MODE
    API here; the dataset format and every consumer of it are unchanged.

    The dictionaries follow the structure below:

    - ``overlap``: maps a parameter point (ordered tuple of parameter values)
      to a dictionary where each key is an adjacent parameter point and each
      value is the complex overlap matrix
      :math:`\langle E_{\text{point}}, H_{\text{adj}} \rangle` with shape
      ``(2*mode_numbers, 2*mode_numbers)``.
    - ``neff``: maps a parameter point to a complex array of effective indices
      with shape ``(2*mode_numbers,)``.
    - ``TE_pol``: maps a parameter point to the TE polarization fraction with
      shape ``(2*mode_numbers,)``.

    :param data_directory: Path that contains the dataset assets and
        ``dataset_info.py`` module.
    :type data_directory: str
    :param backend: Mode solver to use.  Defaults to
        ``DatasetInfo.get_fde_backend()``, which is how a dataset declares its
        cross section (the role ``wg_crosssection.lms`` played for Lumerical).
    :type backend: em_simulation.fde.base.FDEBackend
    :param is_testmode: Disables all solver calls when ``True`` so the class can
        be instantiated in unit tests or to read an existing dataset.
    :type is_testmode: bool
    :param cache_size: Number of solved parameter points kept in memory.  Each
        point is adjacent to several others, so caching avoids re-solving the
        same cross section a handful of times during a sweep.
    :type cache_size: int
    """

    def __init__(self, data_directory, backend=None, is_testmode=False, cache_size=64):
        """Build the updater, load dataset metadata and attach a mode solver."""
        self.data_directory = data_directory
        self._is_testmode = is_testmode

        self.data_info = _load_dataset_info(data_directory)

        self.overlap_filename = os.path.join(data_directory, "overlap.pkl")
        self.neff_filename = os.path.join(data_directory, "neff.pkl")
        self.TE_pol_filename = os.path.join(data_directory, "TE_pol.pkl")

        self.parameter_names = self.get_parameter_names()
        self.parameter_types = self.get_parameter_types()
        self.parameter_grid = self.get_parameter_grid()
        self.mode_numbers = self.get_mode_numbers()
        self.crosssection_x = self.get_crosssection_x()
        self.crosssection_y = self.get_crosssection_y()
        self.wavelength = self.get_wavelength()
        self.prop_axis = prop_axis_index(self.crosssection_x, self.crosssection_y)

        self.backend = None
        if not is_testmode:
            self.backend = backend if backend is not None else self._build_backend()

        self._mode_cache = OrderedDict()
        self._cache_size = int(cache_size)

        self.verify_data()

        self.overlap = self.load_dict_from_pickle(self.overlap_filename)
        self.neff = self.load_dict_from_pickle(self.neff_filename)
        self.TE_pol = self.load_dict_from_pickle(self.TE_pol_filename)

        # Refuse to serve a cache that was solved for a different cross
        # section, wavelength, mesh, window or grid - see dataset_identity.
        verify(
            data_directory,
            fingerprint(self.data_info, self.backend),
            has_cached_data=bool(self.neff),
        )

    def _build_backend(self):
        """Ask the dataset for its solver."""
        if not hasattr(self.data_info, "get_fde_backend"):
            raise AttributeError(
                f"{self.data_directory}/dataset_info.py must define "
                "`get_fde_backend()` returning an FDEBackend, or a backend must "
                "be passed to DataUpdater(..., backend=...)"
            )
        return self.data_info.get_fde_backend()

    # ------------------------------------------------------------ mode solving

    def solve_point(self, parameter_point):
        """Solve one parameter point, memoised.

        :param parameter_point: Tuple ordered as :attr:`parameter_names`.
        :rtype: em_simulation.fde.base.ModeData
        """
        if self._is_testmode:
            raise TestModeError("solve_point() was called in test mode")
        key = tuple(parameter_point)
        cached = self._mode_cache.get(key)
        if cached is not None:
            self._mode_cache.move_to_end(key)
            return cached
        modes = self.backend.solve(key)
        self._mode_cache[key] = modes
        while len(self._mode_cache) > self._cache_size:
            self._mode_cache.popitem(last=False)
        return modes

    def clear_mode_cache(self):
        """Drop cached mode fields (they are the memory-hungry part)."""
        self._mode_cache.clear()

    def update_dataset(self, parameter_names, parameter_values):
        """
        parameter_names: list of str
        parameter_values: list of list which min, max values of the parameter
            e.g) [[0, 50000], [1e-6, 1.5e-6]]
        """
        parameter_names_ref = self.get_parameter_names()
        parameter_grid = self.get_parameter_grid()

        # Validity check
        ## Check len of parameter names and values
        if len(parameter_names) != len(parameter_values):
            print("Number of values and parameter names are different")
            return

        ## Check if parameter name is valid
        param_dict = dict()
        for param_name, parameter_value in zip(parameter_names, parameter_values):
            if param_name not in parameter_names_ref:
                print(param_name, " is not in parameters")
                print("Parameter names are ", *parameter_names_ref)
                return
            param_dict[param_name] = [parameter_value]

        # Find nearest parameter value
        nearest_values = []
        for parameter_name, parameter_value in zip(parameter_names, parameter_values):
            reference = deepcopy(parameter_grid[parameter_name])
            nearest_value = reference[np.argmin(np.abs(reference - parameter_value))]
            nearest_values.append(nearest_value)
            print("nearest point: ", parameter_name, parameter_value)

        # Extract proper values of tuples into list
        ## Find input parameter order in the tuple
        parameter_indices = []
        for parameter_name in parameter_names:
            parameter_indices.append(parameter_names_ref.index(parameter_name))

        ## Initiate tuple list
        param_list = []
        for param_name in parameter_names_ref:
            if param_name not in parameter_names:
                param_list.append(parameter_grid[param_name])
            else:
                param_list.append(param_dict[param_name])

        updating_parameters = list(product(*param_list))

        # Update parameters
        num_point = 0
        for point in tqdm(updating_parameters):
            if not self.check_data_point_in_dataset(point):
                self.calc_data_point(point)
                self.save_data()
                num_point += 1
        self.save_data()
        print("Update", num_point, "Parameters")

        return
    
    def populate_dataframe(self, parameter_names, parameter_ranges, confirm=True):
        """Compute data for all points spanned by the given parameter ranges.

        :param parameter_names: Names of the parameters that define the sweep.
        :type parameter_names: list[str]
        :param parameter_ranges: Inclusive ``[min, max]`` bounds for each
            parameter. Bounds are mapped onto the discrete grid in the dataset.
        :type parameter_ranges: list[list[float]]
        :param confirm: Ask interactively before running.  Set ``False`` for
            scripts and notebooks that should just go ahead.
        :type confirm: bool
        :returns: ``None``.
        :rtype: None
        """
        parameter_names_ref = self.get_parameter_names()
        parameter_grid = self.get_parameter_grid()

        # Validity check
        ## Check len of parameter names and values
        if len(parameter_names) != len(parameter_ranges):
            print("Number of values and parameter names are different")
            return
        ## Check if the parameter ranges
        for parameter_range in parameter_ranges:
            if len(parameter_range) != 2:
                print("Each range of parmeter should be two values. (min & max)")
                print(parameter_range)
                return
            if parameter_range[0] > parameter_range[1]:
                print("The order of the range should be min, max")
                return
        ## Check if parameter name is valid
        param_dict = dict()
        for param_name, parameter_range in zip(parameter_names, parameter_ranges):
            if param_name not in parameter_names_ref:
                print(param_name, " is not in parameters")
                print("Parameter names are ", *parameter_names_ref)
                return
        
        # add missing parameter dimension (with full parameters)
        missing_params = list(set(parameter_names_ref) - set(parameter_names))
        if len(missing_params): # if there is any missing parmaeters
            for missing_param in missing_params:
                parameter_names.append(missing_param)

                missing_param_values = deepcopy(parameter_grid[missing_param])
                parameter_ranges.append([missing_param_values[0], missing_param_values[-1]])

        # reorder param_names & param_values according to the dataset (parameter_names_ref)
        parameter_ranges_reorder= []
        for param_name in parameter_names_ref:
            param_index = index = parameter_names.index(param_name)
            parameter_ranges_reorder.append(parameter_ranges[param_index])


        # mapping parmater range to values
        value_list_from_range = []
        for parameter_name, parameter_range in zip(parameter_names_ref, parameter_ranges_reorder):
            reference = deepcopy(parameter_grid[parameter_name])
            num_trial_points = 3 * len(reference)   # this number can be adjusted it it does not work well
            trial_value_list = np.linspace(parameter_range[0], parameter_range[1], num_trial_points)
            nearest_values = []
            for trial_value in trial_value_list:
                nearest_values.append(reference[np.argmin(np.abs(reference - trial_value))])
            value_list_from_range.append(np.unique(nearest_values))

        #generate list of tuples (product of dimensions)
        updating_parameters = list(product(*value_list_from_range))

        # Update parameters
        for parameter_name, parameter_range in zip(parameter_names_ref, parameter_ranges_reorder):
            print("Parameter name: ", parameter_name, "&  Parameter ranges: ", parameter_range)
        # print("Sample parameter point : ", updating_parameters[0],  )
        if len(updating_parameters)>3:
            print("Sample parameter point : {}, {}, ... , {}".format(updating_parameters[0], updating_parameters[1], updating_parameters[-1]))
        else:
            print("Sample parameter point : {}, {}".format(updating_parameters[0], updating_parameters[-1]))

        print("The total number of points in the given range : ", len(updating_parameters))
        num_to_update = 0

        parameters_to_process = []
        for point in updating_parameters:
            # if not self.check_data_point_in_dataset(point):
            if not self.deepcheck_data_point_in_dataset(point):
                num_to_update += 1
                parameters_to_process.append(point)
        print("The total number of points to be updated : ", num_to_update)

        if confirm:
            user_input = input("Do you want to continue? (y/n): ").strip().lower()
            if user_input != 'y':
                print("Skipping the function...")
                return

        num_point = 0
        save_every = max(1, len(parameters_to_process) // 20)
        for point in tqdm(parameters_to_process):
            if not self.deepcheck_data_point_in_dataset(point):
                self.calc_data_point_modified(point)
                num_point += 1
                # Checkpoint periodically rather than after every point: the
                # pickles grow to tens of MB and rewriting them each time
                # dominates the runtime once the solver is fast.
                if num_point % save_every == 0:
                    self.save_data()
        self.save_data()
        print("Update", num_point, "Parameters")


    def save_data(self):
        if self._is_testmode:
            return
        self.save_dict_to_pickle(self.overlap, self.overlap_filename)
        self.save_dict_to_pickle(self.neff, self.neff_filename)
        self.save_dict_to_pickle(self.TE_pol, self.TE_pol_filename)
        pass

    def load_data(self):
        self.overlap = self.load_dict_from_pickle(self.overlap_filename)
        self.neff = self.load_dict_from_pickle(self.neff_filename)
        self.TE_pol = self.load_dict_from_pickle(self.TE_pol_filename)

    def check_data_point_in_dataset(self, param_point):
        test1 = (param_point in self.overlap)
        test2 = (param_point in self.neff)
        test3 = (param_point in self.TE_pol)
        return (test1 and  test2 and test3)
    
    def deepcheck_data_point_in_dataset(self, param_point):
        test123 = self.check_data_point_in_dataset(param_point)
        adj_points = self.get_adjacent_points(param_point)
        adj_points = self.filter_out_calculated_points(param_point, adj_points)
        test4 = not len(adj_points)
        return (test123 and test4)

    def verify_data(self):
        param_grid = self.get_parameter_grid()
        param_types = self.get_parameter_types()
        for param_name in self.parameter_names:
            if param_name in param_grid and param_name in param_types:
                continue
            print("Dataset parameter names and parameter gird are not matched")
            return
        
        # check parameters in lms files
        pass

    def calc_data_point(self, parameter_point, neighbours=None):
        """Deprecated alias for :meth:`calc_data_point_modified`."""
        return self.calc_data_point_modified(parameter_point, neighbours)

    def calc_data_point_modified(self, parameter_point, neighbours=None):
        """Solve a parameter point and its not-yet-linked neighbours.

        Adds ``neff``/``TE_pol`` for every newly seen point and the overlap
        matrices for every new (point, neighbour) pair, in both directions.

        :param neighbours: The grid points this one must be linked to.  A
            geometry passes the point's neighbours *on its path*; the cascade
            needs overlaps only between consecutive path points, and solving
            every grid neighbour instead costs up to ``2d + 1`` solves per
            visited point on a ``d``-axis dataset - on the two-axis Kocabas
            converter that was three solves per path point, two of them for
            points no device ever visits.  ``None`` keeps the original
            behaviour (all grid neighbours), which ``populate_dataframe`` and
            a bare ``update_dataset`` still rely on.
        :returns: ``1`` if anything was computed, ``0`` if already complete.
        :rtype: int
        """
        if neighbours is None:
            adj_points = self.get_adjacent_points(parameter_point)
        else:
            adj_points = [tuple(q) for q in neighbours if tuple(q) != tuple(parameter_point)]
        adj_points = self.filter_out_calculated_points(parameter_point, adj_points)
        if len(adj_points) == 0:
            return 0
        if self._is_testmode:
            raise TestModeError("The dataset update was attempted in test mode")

        points = [tuple(parameter_point)] + [tuple(p) for p in adj_points]
        mode_data = [self.solve_point(p) for p in points]
        neff_dict, TE_pol_dict, overlaps = self._post_process(points, mode_data)

        if parameter_point not in self.overlap:
            self.overlap[parameter_point] = dict()
            self.neff[parameter_point] = neff_dict[parameter_point]
            self.TE_pol[parameter_point] = TE_pol_dict[parameter_point]

        for adj_point in adj_points:
            self.overlap[parameter_point][adj_point] = overlaps[parameter_point][adj_point]
            if adj_point not in self.overlap:
                self.overlap[adj_point] = dict()
                self.neff[adj_point] = neff_dict[adj_point]
                self.TE_pol[adj_point] = TE_pol_dict[adj_point]
            self.overlap[adj_point][parameter_point] = overlaps[adj_point][parameter_point]

        return 1

    def _post_process(self, points, mode_data):
        """Normalise fields and build the overlap matrices for ``points``.

        ``points[0]`` is the centre point; ``points[1:]`` are its neighbours.
        Overlaps are computed centre<->neighbour only, which is all the EME
        path ever needs.

        :returns: ``(neff_dict, TE_pol_dict, overlaps)`` keyed by parameter point.
        """
        x, y, neff, TE_pol, E, H = assemble(
            mode_data, self.mode_numbers, self.prop_axis,
            lossless=self._is_lossless(),
        )

        centre = points[0]
        neff_dict = {pt: neff[i] for i, pt in enumerate(points)}
        TE_pol_dict = {pt: TE_pol[i] for i, pt in enumerate(points)}

        overlaps = {pt: dict() for pt in points}
        for i, adj in enumerate(points[1:], start=1):
            overlaps[centre][adj] = overlap_matrix(E[0], H[i], x, y, self.prop_axis)
            overlaps[adj][centre] = overlap_matrix(E[i], H[0], x, y, self.prop_axis)

        return neff_dict, TE_pol_dict, overlaps

    def get_adjacent_points(self, parameter_point):
        adj_values = []
        # get adj parameters
        for i in range(len(self.parameter_names)):
            param_name = self.parameter_names[i]
            adj_value = []
            # if error occurs, point is not exact point int param grid (check round operation)
            point_index = np.where(self.parameter_grid[param_name] == parameter_point[i])[0][0]
            if not point_index == 0:
                adj_value.append(self.parameter_grid[param_name][point_index-1])
            if point_index < len(self.parameter_grid[param_name]) - 1:
                adj_value.append(self.parameter_grid[param_name][point_index+1])
            adj_values.append(adj_value)
        
        # get adj one param adj points
        adj_points = []
        for i in range(len(adj_values)):
            temp = list(parameter_point)
            for adj in adj_values[i]:
                temp[i] = adj
                adj_points.append(tuple(temp))

        #adj_points = list(product(*adj_values))
        return adj_points
    
    def filter_out_calculated_points(self, parameter_point, adj_points):
        filtered_adj_points = []
        if not parameter_point in self.overlap:
            return adj_points
        
        for adj_point in adj_points:
            if not adj_point in self.overlap[parameter_point]:
                filtered_adj_points.append(adj_point)
        return filtered_adj_points

    #region get simple data from dataset
    def get_parameter_grid(self):
        return self.data_info.get_parameter_grid()

    def get_parameter_names(self):
        return self.data_info.get_parameter_names()

    def get_parameter_types(self):
        return self.data_info.get_parameter_types()

    def get_mode_numbers(self):
        return self.data_info.get_mode_numbers()

    def get_crosssection_x(self):
        return self.data_info.get_crosssection_x()

    def get_crosssection_y(self):
        return self.data_info.get_crosssection_y()

    def get_wavelength(self):
        return self.data_info.get_wavelength()

    def get_cladding_index(self):
        return self.data_info.get_cladding_index()

    def _is_lossless(self):
        """Whether the backend's modes are lossless by construction."""
        return bool(getattr(self.backend, "lossless", True))
    
    # def get_if_varaible_FDE(self):
    #     return self.data_info._is_variable_FDE()
    #endregion get simple data from dataset

    #region get data point from datset (overlap, neff, TE_pol)

    def get_overlap(self, pt1, pt2):
        """
        Parameters:
            -pt1, pt2: parameter point tuple
        Returns:
            overlap_ab (ndarray): <E_a, H_b> 
            overlap_ba (ndarray): <E_b, H_a> where a is section nearer to the input port and b is section nearer to output port
        """
        if pt1 == pt2:
            overlap_ab = self.same_mode_overlap_matrix()
            overlap_ba = self.same_mode_overlap_matrix()
            return overlap_ab, overlap_ba
        overlap_ab = deepcopy(self.overlap[pt1][pt2])
        overlap_ba = deepcopy(self.overlap[pt2][pt1])
        return overlap_ab, overlap_ba

    def get_overlaps(self, simul_params):
        """
        Parameters:
            -simul_params: list of tuple where each tuple is parameter point
        Returns:
            overlap_ab: <E_a, H_b>
            overlap_ba: <E_b, H_a> where a is section nearer to the input port and b is section nearer to output port
        """
        overlap_ab = np.zeros(shape=(len(simul_params)-1, 2*self.mode_numbers, 2*self.mode_numbers), dtype =np.complex64)
        overlap_ba = np.zeros(shape=(len(simul_params)-1, 2*self.mode_numbers, 2*self.mode_numbers), dtype =np.complex64)

        for i in range(len(simul_params)-1):
            if simul_params[i] == simul_params[i+1]:
                overlap_ab[i] = self.same_mode_overlap_matrix()
                overlap_ba[i] = self.same_mode_overlap_matrix()
                continue
            overlap_ab[i] = deepcopy(self.overlap[simul_params[i]][simul_params[i+1]])
            overlap_ba[i] = deepcopy(self.overlap[simul_params[i+1]][simul_params[i]])
        
        return overlap_ab, overlap_ba

    def get_neffs(self, simul_params):
        neffs = np.zeros(shape = (len(simul_params), 2*self.mode_numbers), dtype = np.complex64)
        for i in range(len(simul_params)):
            point = simul_params[i]
            neffs[i] = deepcopy(self.neff[point])
        return neffs

    def get_TE_pols(self, simul_params):
        TE_pols = np.zeros(shape = (len(simul_params), 2*self.mode_numbers), dtype =np.float16)
        for i in range(len(simul_params)):
            point = simul_params[i]
            TE_pols[i] = deepcopy(self.TE_pol[point])
        return TE_pols
    
    def get_overlaps_modified(self, simul_params, additional_param_dict):
        """
        Parameters:
            -simul_params: list of tuple where each tuple is parameter point
            -additional_param_dict: additional tuple parameter points to calcualte multi_adj_point coupling coeff
                keys: point (tuple)
                values: list of points
        Returns:
            overlap_ab (dict): <E_a, H_b>
            overlap_ba (dict): <E_b, H_a> where a is section nearer to the input port and b is section nearer to output port
        """
        overlap_ab = dict()
        overlap_ba = dict()

        for i in range(len(simul_params)-1):
            overlap_ab[simul_params[i]] = dict()
            overlap_ba[simul_params[i+1]] = dict()

        for i in range(len(simul_params)-1):
            if simul_params[i] == simul_params[i+1]:
                overlap_ab[simul_params[i]][simul_params[i+1]] = self.same_mode_overlap_matrix()
                overlap_ba[simul_params[i+1]][simul_params[i]] = self.same_mode_overlap_matrix()
                continue
            try:
                overlap_ab[simul_params[i]][simul_params[i+1]] = deepcopy(self.overlap[simul_params[i]][simul_params[i+1]])
                overlap_ba[simul_params[i+1]][simul_params[i]] = deepcopy(self.overlap[simul_params[i+1]][simul_params[i]])
            except: # multi adj pts (calculated at below for loop)
                pass
        
        for key in additional_param_dict.keys():    # multi adj pts
            for additional_pt in additional_param_dict[key]:
                if not additional_pt in overlap_ba[additional_pt]: overlap_ba[additional_pt] = dict()
                overlap_ab[key][additional_pt] = deepcopy(self.overlap_ab[key][additional_pt])
                overlap_ba[additional_pt][key] = deepcopy(self.overlap_ba[additional_pt][key])
        
        return overlap_ab, overlap_ba
    
    def get_neffs_modified(self, simul_params):
        neffs = dict()
        for point in simul_params:
            neffs[point] = deepcopy(self.neff[point])
        return neffs

    def get_TE_pols_modified(self, simul_params):
        TE_pols = dict()
        for point in simul_params:
            TE_pols[point] = deepcopy(self.TE_pol[point])
        return TE_pols
        
    
    def same_mode_overlap_matrix(self):
        overlap_matrix = np.zeros(shape=(2*self.mode_numbers, 2*self.mode_numbers))
        I = np.eye(self.mode_numbers)
        overlap_matrix[:self.mode_numbers,:self.mode_numbers] = deepcopy(I)
        overlap_matrix[:self.mode_numbers,self.mode_numbers:] = -deepcopy(I)
        overlap_matrix[self.mode_numbers:,:self.mode_numbers] = deepcopy(I)
        overlap_matrix[self.mode_numbers:,self.mode_numbers:] = -deepcopy(I)
        return overlap_matrix

    #endregion get data point from datset (overlap, neff, TE_pol)

    #region load and save pickle file
    def save_dict_to_pickle(self, data, filename):
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(filename), exist_ok=True)

            with open(filename, 'wb') as file:
                pickle.dump(data, file)
        except Exception as e:
            print(f"Failed to save data to {filename}: {str(e)}")


    def load_dict_from_pickle(self, filename):
        try: # if file exists
            with open(filename, 'rb') as file:
                return pickle.load(file)
        except: # if file does not exists, generate and load file
            with open(filename, 'wb') as file:
                data = dict()
                pickle.dump(data, file)
            with open(filename, 'rb') as file:
                return pickle.load(file)
    #endregion load and save pickle file      

    def find_mode_fields(self, parameter_point):
        """Solve a parameter point and return its raw (un-normalised) modes.

        Kept for inspection and plotting; the EME path uses the cached dataset.

        :returns: ``(Efields, Hfields, x, y)`` with the field arrays shaped
            ``(mode_numbers, len(x), len(y), 3)``.
        """
        md = self.solve_point(tuple(parameter_point))
        # (modes, 3, nx, ny) -> (modes, nx, ny, 3), matching the old FDE layout
        Efields = np.transpose(md.E, (0, 2, 3, 1))
        Hfields = np.transpose(md.H, (0, 2, 3, 1))
        return Efields, Hfields, md.x, md.y

    def get_index_profile(self, parameter_point):
        """Refractive index ``n(x, y)`` of a cross section, for plotting.

        :returns: ``(n, x, y)`` with ``n`` of shape ``(len(x), len(y))``.
        """
        cs = self.backend.cross_section
        x, y = self.backend.solve_grid()
        params = dict(zip(self.parameter_names, parameter_point))
        return cs.index(x, y, params), x, y




def _load_dataset_info(data_directory):
    """Load ``dataset_info.py`` from a dataset directory.

    Loaded by path rather than by ``sys.path`` insertion so that several
    datasets can be open at once without shadowing each other.
    """
    path = os.path.join(data_directory, "dataset_info.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no dataset_info.py in {data_directory}")
    spec = importlib.util.spec_from_file_location("dataset_info", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DatasetInfo()
