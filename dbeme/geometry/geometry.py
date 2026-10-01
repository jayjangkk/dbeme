from ..data_updater.data_updater import DataUpdater
from copy import deepcopy
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from .mode_tracking import hungarian_mode_links



#: Loss above which a mode of a *lossless* model is numerical junk (100 dB/cm).
RADIATION_IMAG_NEFF = 0.000284


def _data_is_lossless(data):
    """Mirror of ``propagator._geometry_is_lossless``: the dataset knows."""
    probe = getattr(data, "_is_lossless", None)
    return bool(probe()) if callable(probe) else True


def radiation_mode_mask(neffs, cladding_index, lossless=True):
    """Which modes are not guided: ``Re(n_eff)`` below the cladding.

    In a lossless model any imaginary part at all is round-off or a window
    artefact, so a second rule - ``|Im(n_eff)| > RADIATION_IMAG_NEFF`` - also
    rejects those.  On a lossy basis that rule would reject every physical
    mode (a Si wire 20 nm from gold sits at ``Im ~ 5e-3``, a gap plasmon at
    ``2e-2``), so there only the cutoff applies.
    """
    neffs = np.asarray(neffs)
    mask = neffs.real < float(cladding_index)
    if lossless:
        mask = mask | (np.abs(neffs.imag) > RADIATION_IMAG_NEFF)
    return mask


class Geometry():
    def __init__(self, data_updater:DataUpdater, limit_mode_number = 0, verbose=True):
        self._check_data_validity()
        self.data = data_updater
        self.simulation_parameters = None
        self.output_data = None
        self._limit_mode_number = limit_mode_number
        self._tracking_mode_names = None
        self.parameter_names = self.data.parameter_names
        self.parameter_grid = self.data.parameter_grid
        self.wavelength = self.data.wavelength
        self._is_composite_geometry = False
        self._verbose = verbose

    
    def calc_output_data(self):
        """Assemble modal data required by the propagator workflow.

        The method calls :meth:`calc_simulation_parameters`, which must be
        implemented by subclasses, pulls the necessary overlap and effective
        index information from the dataset, performs mode-order tracking, and
        stores the result on ``self.output_data``.

        :returns: Dictionary containing overlap matrices, propagation steps,
            effective indices, propagation constants, radiation masks, mode
            names, and additional overlap corrections.
        :rtype: dict
        """
        # calculate simulation parameter and update datset
        simul_params, delta_zs = self.calc_simulation_parameters()
        delta_zs = np.array(delta_zs) # wrap the list into np array
        self.simulation_parameters = simul_params
        extended_simul_params, additional_param_dict = self.get_extended_simul_params(simul_params)

        # Expand the path first: `interp_multi_adj_pts` inserts intermediate
        # grid points wherever the path moves by more than one step, and those
        # points need to be in the dataset too.
        EME_path, EME_delta_zs, multi_adj_index, index_mapping = self.interp_multi_adj_pts(simul_params, delta_zs)
        links = self._path_links(EME_path, simul_params, additional_param_dict)
        self.update_dataset(list(dict.fromkeys(list(extended_simul_params) + list(EME_path))), links)

        # get data from dataset
        overlap_ab, overlap_ba = self._get_overlaps(EME_path)
        neff = self._get_neffs(EME_path)
        neff_clmt = self._get_neffs(simul_params)
        TE_pol = self._get_TE_pols(EME_path)
        additional_overlap_dict =  self._get_additional_overlap_dict(simul_params, multi_adj_index)

        # if mode number is limited, do limiting process
        num_sections, mode_numbers = neff.shape
        mode_numbers = int(mode_numbers/2)
        if self._limit_mode_number and mode_numbers > self._limit_mode_number:
            neff, TE_pol, overlap_ab, overlap_ba = self._reduce_mode_number(neff, TE_pol, overlap_ab, overlap_ba)
            additional_overlap_dict = self._reduce_mode_number_overlap_dict(additional_overlap_dict)

        # reorder data by best overlap modes
        mode_links = self._generate_mode_links(overlap_ab)
        tracking_mode_names = self._generate_tracking_mode_names(mode_links)

        self._tracking_mode_names = tracking_mode_names

        neff = self._reorder_data(tracking_mode_names, neff)
        TE_pol = self._reorder_data(tracking_mode_names, TE_pol)
        overlap_ab, overlap_ba = self._reorder_overlap(tracking_mode_names, overlap_ab, overlap_ba)

        # reorder additional overlap
        additional_overlap_dict = self.match_mutual_adj_overlap(additional_overlap_dict, EME_path, tracking_mode_names, index_mapping)
        # equalize overlap phase
        overlap_ab, overlap_ba, additional_overlap_dict = self._equalize_overlap_phase(overlap_ab, overlap_ba, additional_overlap_dict, index_mapping)
        

        # additional data
        radiation_mode_mask = self._get_radiation_mode_mask(neff)
        beta = (2*np.pi)*neff/self.wavelength


        # generate output_data
        output_data = dict()
        output_data["overlap_ab"] = overlap_ab  # EME path overlap
        output_data["overlap_ba"] = overlap_ba
        output_data["delta_zs"] = delta_zs
        output_data["EME_delta_zs"] = EME_delta_zs
        output_data["neff"] = neff
        output_data["beta"] = beta
        output_data["radiation_mode_mask"] = radiation_mode_mask
        output_data["TE_pol"] = TE_pol
        output_data["path"] = simul_params
        output_data["EME_path"] = EME_path
        output_data["additional_overlap_dict"] = additional_overlap_dict
        output_data["index_mapping"] = index_mapping    # path index mapping from EME_path to simul_params path
        self.output_data = output_data
        return output_data
    
    def calc_simulation_parameters(self):
        """
        Returns: 
            -simul_params: list of tuple where each tuple is parameter point.\n
                each parameter point should be matched in the grid point in dataset
            -delta_zs: list of length between parameter points. len(delta_zs) == len(simul_params)-1
        """
        pass

    def _get_radiation_mode_mask(self, neffs):
        return radiation_mode_mask(neffs, self._get_cladding_index(),
                                   _data_is_lossless(self.data))
    
    #region plots
    def plot_neffs(self, show_radiation_mode = False):
        """Plot real effective indices along the propagation path.

        :param show_radiation_mode: When ``True`` include radiation modes in the
            plot; otherwise they are omitted.
        :type show_radiation_mode: bool
        """
        if self.output_data == None:
            self.calc_output_data()
        # delta_zs = deepcopy(self.output_data["delta_zs"])
        delta_zs = deepcopy(self.output_data["EME_delta_zs"])
        delta_zs = np.insert(delta_zs, 0, [0])
        propagation_length = delta_zs.cumsum()

        neffs = deepcopy(self.output_data["neff"])
        mode_numbers = int(neffs.shape[1]/2)
        neffs = neffs[:,:mode_numbers]
        threshold = 1
        neffs = np.where(neffs < threshold, np.nan, neffs)

        if not show_radiation_mode:
            radiation_mode_mask = deepcopy(self.output_data["radiation_mode_mask"])[:,:mode_numbers]
            neffs = np.where(radiation_mode_mask, np.nan, neffs)

        for i in range(mode_numbers):
            mode_name = "mode " + str(i+1)
            plt.plot(propagation_length, neffs[:,i], label = mode_name)

        plt.xlabel("propagation length")
        plt.ylabel("neff")
        plt.legend()
        plt.title("neffs along the propagation")
        plt.show()
    

    def plot_TE_polarization_fraction(self, show_radiation_mode = False):
        """Plot the TE polarization fraction along the propagation path.

        :param show_radiation_mode: When ``True`` include radiation modes in the
            plot; otherwise they are omitted.
        :type show_radiation_mode: bool
        """
        if self.output_data == None:
            self.calc_output_data()
        delta_zs = deepcopy(self.output_data["EME_delta_zs"])
        delta_zs = np.insert(delta_zs, 0, [0])
        propagation_length = delta_zs.cumsum()

        TE_pols = deepcopy(self.output_data["TE_pol"])
        neffs = deepcopy(self.output_data["neff"])
        mode_numbers = int(neffs.shape[1]/2)
        TE_pols = TE_pols[:,:mode_numbers]
        neffs = neffs[:,:mode_numbers]
        threshold = 1
        TE_pols = np.where(neffs < threshold, np.nan, TE_pols)

        if not show_radiation_mode:
            radiation_mode_mask = deepcopy(self.output_data["radiation_mode_mask"])[:,:mode_numbers]
            TE_pols = np.where(radiation_mode_mask, np.nan, TE_pols)

        for i in range(mode_numbers):
            mode_name = "mode " + str(i+1)
            plt.plot(propagation_length, TE_pols[:,i], label = mode_name)

        plt.xlabel("propagation length")
        plt.ylabel("TE polarization fraction")
        plt.legend()
        plt.title("TE polarization fraction along the propagation")
        plt.show()
    
    def plot_structure_parameters(self, resolution = 500):
        pass

    def plot_2D_structure(self):
        pass

        #endregion plots

    #region update dataset
    def update_dataset(self, simulation_parameters, links=None):
        """Solve what the dataset lacks for these points.

        :param links: ``{point: {neighbour, ...}}`` from :meth:`_path_links`.
            With it, each point is linked only to its neighbours on the path;
            without it, to every grid neighbour (the original behaviour).
        """
        iterator = tqdm(simulation_parameters) if self._verbose else simulation_parameters
        for point in iterator:
            neighbours = None if links is None else sorted(links.get(tuple(point), ()))
            is_calculated = self.calc_data_point(point, neighbours)
            if is_calculated:
                self.data.save_data()
        self.data.save_data()
        pass

    @staticmethod
    def _path_links(eme_path, simul_params=(), additional_param_dict=None):
        """Which (point, neighbour) overlaps a path actually consumes.

        Consecutive points of the expanded EME path, both directions, plus the
        mutual-neighbour detours ``get_extended_simul_params`` records for a
        diagonal step (``p_i -> m -> p_{i+1}``).  Nothing else: on a
        multi-axis dataset the off-path grid neighbours would otherwise be
        solved too, and never used.
        """
        links = {}

        def link(a, b):
            a, b = tuple(a), tuple(b)
            if a == b:
                return
            links.setdefault(a, set()).add(b)
            links.setdefault(b, set()).add(a)

        path = [tuple(p) for p in eme_path]
        for a, b in zip(path[:-1], path[1:]):
            link(a, b)
        params = [tuple(p) for p in simul_params]
        for i, p in enumerate(params[:-1]):
            for m in (additional_param_dict or {}).get(p, ()):
                link(p, m)
                link(m, params[i + 1])
        return links

    def check_if_multi_adj_pt(self, pt1, pt2):
        diff_param = 0
        for i in range(len(pt1)):
            if pt1[i] == pt2[i]: continue
            diff_param += 1
        
        if diff_param > 1: return 1
        else: return 0

    def find_mutual_adj_pt(self, pt1, pt2):
        adj_pt1 = self.data.get_adjacent_points(pt1)
        adj_pt2 = self.data.get_adjacent_points(pt2)
        
        intersection_set = set(adj_pt1) & set(adj_pt2)
        mutual_adj_pts = list(intersection_set)

        return mutual_adj_pts
    
    def get_extended_simul_params(self, simulation_params):
        additional_param_points = []
        additional_param_dict = dict()
        for i in range(len(simulation_params)-1):
            if self.check_if_multi_adj_pt(simulation_params[i], simulation_params[i+1]):
                temp = self.find_mutual_adj_pt(simulation_params[i], simulation_params[i+1])
                additional_param_dict[simulation_params[i]] = deepcopy(temp)
                additional_param_points = list(set(additional_param_points + temp))
        extended_simul_params = list(set(simulation_params + additional_param_points))
        return extended_simul_params, additional_param_dict
    
    def _grid_steps(self, param_index, start, stop):
        """Grid values from just after ``start`` through to ``stop``.

        :param param_index: Position of the parameter in the point tuple.
        :param start: Current value; must be on the grid.
        :param stop: Target value; must be on the grid.
        :returns: The intervening grid values in order, ending at ``stop``.
        """
        grid = np.asarray(self.parameter_grid[self.parameter_names[param_index]])
        i_start = int(np.argmin(np.abs(grid - start)))
        i_stop = int(np.argmin(np.abs(grid - stop)))
        step = 1 if i_stop >= i_start else -1
        return [grid[k] for k in range(i_start + step, i_stop + step, step)]

    def interp_multi_adj_pts(self, pts, delta_zs):
        pts = list(deepcopy(pts))
        delta_zs = list(delta_zs)
        delta_z_copy = deepcopy(delta_zs)
        multi_adj_index = [] # multi_adj index in original path
        added_points = 0
        index_mapping = dict()

        for i, delta_z in enumerate(delta_z_copy):
            pt1, pt2 = pts[i + added_points], pts[i + added_points + 1]
            path = []
            current_point = list(pt1)
            
            for j in range(len(pt1)): # number of parameters in a tuple
                if current_point[j] == pt2[j]:
                    continue
                # Step along the grid rather than jumping straight to pt2[j]:
                # the dataset only holds overlaps between *adjacent* grid
                # points, so a parameter that moves by more than one step has
                # to be walked.  Continuous geometries never do that, but a
                # discontinuous one does - a two-arc S-bend flips its curvature
                # from +1/R to -1/R at the join.
                for value in self._grid_steps(j, current_point[j], pt2[j]):
                    current_point[j] = value
                    path.append(tuple(current_point))
            
            # Remove the first and last points to avoid duplication
            path = path[:-1]
            # Create a new delta_z array for the path
            new_delta_zs = np.ones(len(path) + 1) * (delta_z / (len(path) + 1))
            
            if len(path) > 0:
                # Insert the new points into the original list
                pts[i + added_points + 1:i + added_points + 1] = path
                # Insert the new delta_zs into the original list
                delta_zs[i + added_points:i + added_points + 1] = list(new_delta_zs)
                # Update the added_points counter
                added_points += len(path)
                # Only a genuinely multi-parameter transition needs the
                # mutual-adjacent-point correction below; a single parameter
                # walked across several grid steps does not.
                if self.check_if_multi_adj_pt(pt1, pt2):
                    multi_adj_index.append(i)

                for k in range(len(path)):
                    index_mapping[i + added_points - len(path) + k] = i
                
            index_mapping[i + added_points] = i
        index_mapping[i + added_points+1] = i+1 # the last index (the len(delta_zs) = len(pts)-1)

        delta_zs = np.array(delta_zs)

        return pts, delta_zs, multi_adj_index, index_mapping
    

    def _get_additional_overlap_dict(self, simul_params, multi_adj_index):
        """
        Returns:
            additional_overlap_dict: dictionary, two keys: "ab" and "ba"
                additional_overlap_dict["ab"]: dict where multi_adj index in simul_params is key and list of overlaps is value
                additional_overlap_dict["ba"]: dict where multi_adj index in simul_params is key and list of overlaps is value
        """
        additional_overlap_dict = {"ab": dict(), "ba": dict()}
        for i in multi_adj_index:
            mutual_adj_pts = self.find_mutual_adj_pt(simul_params[i], simul_params[i+1])
            if not mutual_adj_pts:
                # No point is adjacent to both ends of this transition, so
                # there is no correction to make. Leaving the key out keeps the
                # phase-equalisation pass from meeting an empty array.
                continue
            additional_overlap_dict["ab"][i] = []
            additional_overlap_dict["ba"][i] = [] 
            for pt in mutual_adj_pts:
                overlap_ab, overlap_ba = self.data.get_overlap(simul_params[i], pt)
                additional_overlap_dict["ab"][i].append(overlap_ab)
                additional_overlap_dict["ba"][i].append(overlap_ba)
        
        return additional_overlap_dict
    
    def _generate_new_tracking_mode_names(self, overlap, tracking_mode_names, index):
        """
        Generate new tracking mode names for the mutual_adj_pt overlap
        Parameters:
            - overlap: The overlap matrix from additional_overlap_dict["ab"][original_index]
            - tracking_mode_names: The original tracking mode names for the EME path
            - index: The index of pt1 in the EME path
        Returns:
            - new_tracking_mode_names: New tracking mode names for the mutual_adj_pt overlap
        """
        num_modes = tracking_mode_names.shape[1]
        new_tracking_mode_names = np.zeros(num_modes, dtype=int)

        overlap = deepcopy(overlap)
        overlap = overlap[:num_modes, :num_modes]
        # Generate mode links for the sections between pt1 and mutual_pts    
        for i in range(num_modes):
            max_index = np.argmax(np.abs(overlap[:, i]))
            if np.abs(overlap[max_index, i]) > 0.5:
                new_tracking_mode_names[i] = tracking_mode_names[index, max_index]
            else:
                new_tracking_mode_names[i] = -1 # Indicating no valid link
        
        return new_tracking_mode_names
    
    def match_mutual_adj_overlap(self, additional_overlap_dict, EME_path, tracking_mode_names, index_mapping):
        matched_original_index = []
        for i, pt1 in enumerate(EME_path[:-1]):
            original_index = index_mapping[i]
            if original_index in additional_overlap_dict["ab"] and (not original_index in matched_original_index):
                matched_original_index.append(original_index)
                mutual_pt_overlpas = additional_overlap_dict["ab"][original_index]
                for idx, mutual_pt_overlap in enumerate(mutual_pt_overlpas):
                    overlap_ab = additional_overlap_dict["ab"][original_index][idx]
                    overlap_ba = additional_overlap_dict["ba"][original_index][idx]
                    
                    # Generate new tracking_mode_names for the mutual_adj_pt overlap
                    new_tracking_mode_names = self._generate_new_tracking_mode_names(overlap_ab, tracking_mode_names, i)
                    
                    sorted_overlap_ab, sorted_overlap_ba = self._reorder_additional_overlap(overlap_ab, overlap_ba, tracking_mode_names, new_tracking_mode_names, i)
                    additional_overlap_dict["ab"][original_index][idx] = sorted_overlap_ab
                    additional_overlap_dict["ba"][original_index][idx] = sorted_overlap_ba

        return additional_overlap_dict

    def calc_data_point(self, parameter_point, neighbours=None):
        return self.data.calc_data_point_modified(parameter_point, neighbours)

    def check_data_point_in_dataset(self, param_point):
        return self.data.check_data_point_in_dataset(param_point)
    #endregion update dataset
    
    #region get data from dataset
    def _get_overlaps(self, simul_param):
        """
        Parameters:
            -simul_params: list of tuple where each tuple is parameter point
        Returns:
            overlap_ab: <E_a, H_b>
            overlap_ba: <E_b, H_a> where a is section nearer to the input port and b is section nearer to output port
        """
        return self.data.get_overlaps(simul_param)
    
    def _get_neffs(self, simul_param):
        return self.data.get_neffs(simul_param)
    
    def _get_TE_pols(self, simul_param):
        return self.data.get_TE_pols(simul_param)
    
    def _get_overlaps_modified(self, simul_param, additional_param_dict):
        return self.data.get_overlaps_modified(simul_param, additional_param_dict)
    
    def _get_neffs_modified(self, simul_param):
        return self.data.get_neffs_modified(simul_param)
    
    def _get_TE_pols_modified(self, simul_param):
        return self.data.get_TE_pols_modified(simul_param)
    
    def _get_wavelength(self):
        return self.data.get_wavelength()
    
    def _get_cladding_index(self):
        return self.data.get_cladding_index()
    
    def find_mode_fields(self, param_point):
        return self.data.find_mode_fields(param_point)
    
    #endregion get data from dataset

    #region reorder data, overlaps & equalize phase
    def _generate_mode_links(self, overlap_matrices):
        """Link each section's modes to the previous section's, by overlap.

        Delegates to :func:`~dbeme.geometry.mode_tracking.hungarian_mode_links`.
        See that module for why the assignment has to be a permutation
        (docs/validation_backlog.md §5.13) and what the earlier per-column ``argmax`` got wrong.

        Returns:
            - mode_link: ndarray of shape (num_section-1, mode_number) 

                mode in (i+1)th section, j-th mode is linked to i-th section (mode_link[i,j])-th mode
        """
        mode_links, quality = hungarian_mode_links(overlap_matrices)
        self._mode_link_quality = quality
        return mode_links

    def _generate_tracking_mode_names(self, mode_links):
        """
        Generate mode names for tracking the physical mode along the propagation
        Parameters:
            - mode_links: ndarray of shape (num_section-1, mode_number)
        
        Returns:
            - tracking_mode_names: ndarray of shape (num_section, mode_number)\n
                new_mode_number is total number of modes appeared in the structure
        """
        num_sections, num_modes = mode_links.shape
        num_sections += 1

        tracking_mode_names = np.ones(shape=(num_sections, num_modes), dtype=int) * (-1)

        # initialize the first section name
        for i in range(num_modes):
            tracking_mode_names[0][i] = i
        
        # update remaining section names
        for i in range(num_sections-1):
            for j in range(num_modes):
                linked_index = mode_links[i,j]
                if linked_index < 0:
                    # new mode
                    tracking_mode_names[i+1,j] = tracking_mode_names.max() + 1
                else:
                    # tracked mode
                    tracking_mode_names[i+1,j] = tracking_mode_names[i, linked_index]
        
        return tracking_mode_names
    
    def _reorder_data(self, tracking_mode_names, data):
        """
        Parmeters:
            -tracking_mode_names: ndarray of shape (num_section, mode_number) with int type
            -data: ndarray shape of (section_count, 2*mode_numbers, ... ) data orderd by effective indices.
        Returns:
            - reorderd data: data orderd by polarization modes. ndarray of shape (section_count, max_mode_num , ...)
        """
        num_modes = int(data.shape[1]/2)
        forward_data = deepcopy(data[:,:num_modes])
        backward_data = deepcopy(data[:,num_modes:])

        new_half_shape = data.shape[:1] + (int(tracking_mode_names.max()+1), ) + data.shape[2:]
        new_total_shape = data.shape[:1] + (2*int(tracking_mode_names.max()+1), ) + data.shape[2:]

        reordered_data = np.zeros(shape = new_total_shape, dtype = data.dtype)
        reordered_forward_data = np.zeros(shape = new_half_shape, dtype = data.dtype)
        reordered_backward_data = np.zeros(shape = new_half_shape, dtype = data.dtype)

        section_count = tracking_mode_names.shape[0]
        mode_numbers = tracking_mode_names.shape[1]
        for i in range(section_count):
            for j in range(mode_numbers):
                index_num = tracking_mode_names[i,j]
                reordered_forward_data[i, index_num] = forward_data[i,j]
                reordered_backward_data[i, index_num] = backward_data[i,j]
        
        reordered_data[:,:int(tracking_mode_names.max()+1)] = reordered_forward_data
        reordered_data[:,int(tracking_mode_names.max()+1):] = reordered_backward_data

        return reordered_data
    
    def _reorder_overlap(self, tracking_mode_names, overlap_ab, overlap_ba):
        """
        Parmeters:
            -tracking_mode_names: ndarray of shape (num_section, mode_number) with int type
            -overlap_ab: ndarray shape of (section_count-1, 2*mode_numbers, 2*mode_numbers ) data orderd by effective indices.\n
                overlap_ab[i,j,k] is ovverlap(E_j(i-th section), H_k(i+1th secion))
            -overlap_bb: ndarray shape of (section_count-1, 2*mode_numbers, 2*mode_numbers ) data orderd by effective indices.\n
                overlap_ba[i,j,k] is ovverlap(E_j(i+1th section), H_k(i-th secion))
        Returns:
            - reorderd data: data orderd by polarization modes. ndarray of shape (section_count-1, 2*max_mode_num, 2*max_mode_num )
        """
        section_count, mode_num, _ = overlap_ab.shape
        mode_num = int(mode_num/2)
        new_mode_num = tracking_mode_names.max()+1

        # Initialize separated overlap matrices
        # expand overlap if reordered matrix is larger than original matrix
        overlap_ab_11 = np.zeros(shape=(section_count, new_mode_num, new_mode_num), dtype = np.complex64)
        overlap_ab_12 = np.zeros(shape=(section_count, new_mode_num, new_mode_num), dtype = np.complex64)
        overlap_ab_21 = np.zeros(shape=(section_count, new_mode_num, new_mode_num), dtype = np.complex64)
        overlap_ab_22 = np.zeros(shape=(section_count, new_mode_num, new_mode_num), dtype = np.complex64)

        overlap_ba_11 = np.zeros(shape=(section_count, new_mode_num, new_mode_num), dtype = np.complex64)
        overlap_ba_12 = np.zeros(shape=(section_count, new_mode_num, new_mode_num), dtype = np.complex64)
        overlap_ba_21 = np.zeros(shape=(section_count, new_mode_num, new_mode_num), dtype = np.complex64)
        overlap_ba_22 = np.zeros(shape=(section_count, new_mode_num, new_mode_num), dtype = np.complex64)
        
        # separate overlap matrices
        overlap_ab_11[:,:mode_num,:mode_num] = overlap_ab[:,:mode_num,:mode_num]
        overlap_ab_12[:,:mode_num,:mode_num] = overlap_ab[:,:mode_num,mode_num:]
        overlap_ab_21[:,:mode_num,:mode_num] = overlap_ab[:,mode_num:,:mode_num]
        overlap_ab_22[:,:mode_num,:mode_num] = overlap_ab[:,mode_num:,mode_num:]

        overlap_ba_11[:,:mode_num,:mode_num] = overlap_ba[:,:mode_num,:mode_num]
        overlap_ba_12[:,:mode_num,:mode_num] = overlap_ba[:,:mode_num,mode_num:]
        overlap_ba_21[:,:mode_num,:mode_num] = overlap_ba[:,mode_num:,:mode_num]
        overlap_ba_22[:,:mode_num,:mode_num] = overlap_ba[:,mode_num:,mode_num:]

        
        # initialize rows reordered overlap_ab and cols reordered overlap_ba
        overlap_ab_11_row_reordered = np.zeros(shape=(section_count, new_mode_num, new_mode_num), dtype = np.complex64)
        overlap_ab_12_row_reordered = np.zeros(shape=(section_count, new_mode_num, new_mode_num), dtype = np.complex64)
        overlap_ab_21_row_reordered = np.zeros(shape=(section_count, new_mode_num, new_mode_num), dtype = np.complex64)
        overlap_ab_22_row_reordered = np.zeros(shape=(section_count, new_mode_num, new_mode_num), dtype = np.complex64)

        overlap_ba_11_col_reordered = np.zeros(shape=(section_count, new_mode_num, new_mode_num), dtype = np.complex64)
        overlap_ba_12_col_reordered = np.zeros(shape=(section_count, new_mode_num, new_mode_num), dtype = np.complex64)
        overlap_ba_21_col_reordered = np.zeros(shape=(section_count, new_mode_num, new_mode_num), dtype = np.complex64)
        overlap_ba_22_col_reordered = np.zeros(shape=(section_count, new_mode_num, new_mode_num), dtype = np.complex64)

        # reorder row of overlap_ab and col of overlap_ba
        for i in range(section_count):
            for j in range(mode_num):
                index_num = tracking_mode_names[i,j]
                overlap_ab_11_row_reordered[i, index_num] = overlap_ab_11[i,j]
                overlap_ab_12_row_reordered[i, index_num] = overlap_ab_12[i,j]
                overlap_ab_21_row_reordered[i, index_num] = overlap_ab_21[i,j]
                overlap_ab_22_row_reordered[i, index_num] = overlap_ab_22[i,j]

                overlap_ba_11_col_reordered[i, :, index_num] = overlap_ba_11[i,:,j]
                overlap_ba_12_col_reordered[i, :, index_num] = overlap_ba_12[i,:,j]
                overlap_ba_21_col_reordered[i, :, index_num] = overlap_ba_21[i,:,j]
                overlap_ba_22_col_reordered[i, :, index_num] = overlap_ba_22[i,:,j]

        # initialize rows and column reordered overlap_ab and overlap_ba  
        overlap_ab_11_tot_reordered = np.zeros(shape=(section_count, new_mode_num, new_mode_num), dtype = np.complex64)
        overlap_ab_12_tot_reordered = np.zeros(shape=(section_count, new_mode_num, new_mode_num), dtype = np.complex64)
        overlap_ab_21_tot_reordered = np.zeros(shape=(section_count, new_mode_num, new_mode_num), dtype = np.complex64)
        overlap_ab_22_tot_reordered = np.zeros(shape=(section_count, new_mode_num, new_mode_num), dtype = np.complex64)

        overlap_ba_11_tot_reordered = np.zeros(shape=(section_count, new_mode_num, new_mode_num), dtype = np.complex64)
        overlap_ba_12_tot_reordered = np.zeros(shape=(section_count, new_mode_num, new_mode_num), dtype = np.complex64)
        overlap_ba_21_tot_reordered = np.zeros(shape=(section_count, new_mode_num, new_mode_num), dtype = np.complex64)
        overlap_ba_22_tot_reordered = np.zeros(shape=(section_count, new_mode_num, new_mode_num), dtype = np.complex64)

        # reorder col of overlap_ab and row of overlap_ba
        for i in range(section_count):
            for j in range(mode_num):
                index_num = tracking_mode_names[i+1,j]
                overlap_ba_11_tot_reordered[i, index_num] = overlap_ba_11_col_reordered[i,j]
                overlap_ba_12_tot_reordered[i, index_num] = overlap_ba_12_col_reordered[i,j]
                overlap_ba_21_tot_reordered[i, index_num] = overlap_ba_21_col_reordered[i,j]
                overlap_ba_22_tot_reordered[i, index_num] = overlap_ba_22_col_reordered[i,j]

                overlap_ab_11_tot_reordered[i, :, index_num] = overlap_ab_11_row_reordered[i,:,j]
                overlap_ab_12_tot_reordered[i, :, index_num] = overlap_ab_12_row_reordered[i,:,j]
                overlap_ab_21_tot_reordered[i, :, index_num] = overlap_ab_21_row_reordered[i,:,j]
                overlap_ab_22_tot_reordered[i, :, index_num] = overlap_ab_22_row_reordered[i,:,j]
        
        # merge reorderd overlap
        reordered_overlap_ab = np.zeros(shape=(section_count, 2*new_mode_num, 2*new_mode_num), dtype = np.complex64)
        reordered_overlap_ba = np.zeros(shape=(section_count, 2*new_mode_num, 2*new_mode_num), dtype = np.complex64)

        reordered_overlap_ab[:,:new_mode_num, :new_mode_num] = overlap_ab_11_tot_reordered
        reordered_overlap_ab[:,:new_mode_num, new_mode_num:] = overlap_ab_12_tot_reordered
        reordered_overlap_ab[:,new_mode_num:, :new_mode_num] = overlap_ab_21_tot_reordered
        reordered_overlap_ab[:,new_mode_num:, new_mode_num:] = overlap_ab_22_tot_reordered

        reordered_overlap_ba[:,:new_mode_num, :new_mode_num] = overlap_ba_11_tot_reordered
        reordered_overlap_ba[:,:new_mode_num, new_mode_num:] = overlap_ba_12_tot_reordered
        reordered_overlap_ba[:,new_mode_num:, :new_mode_num] = overlap_ba_21_tot_reordered
        reordered_overlap_ba[:,new_mode_num:, new_mode_num:] = overlap_ba_22_tot_reordered

        return reordered_overlap_ab, reordered_overlap_ba
    
    def _reorder_additional_overlap(self, overlap_ab, overlap_ba, tracking_mode_names, new_tracking_mode_names, index):
        """
        Reorder and equalize the phase of the overlap matrices.
        Parameters:
            - overlap_ab: ndarray of shape (2*mode_numbers, 2*mode_numbers) overlap(E_a, H_b)
            - overlap_ba: ndarray of shape (2*mode_numbers, 2*mode_numbers) overlap(E_b, H_a)
            - new_tracking_mode_names: ndarray of shape (mode_number,) with int type
                (mode_names of "b" in "overlap_a'b'")
            - index: The index of "a" of "overlap_'a'b" in the EME_path
        Returns:
            - equalized_ab: Reordered and phase-equalized overlap_ab
            - equalized_ba: Reordered and phase-equalized overlap_ba
        """
        original_mode_num = overlap_ab.shape[0] // 2
        new_mode_num = tracking_mode_names.max() + 1

        # Initialize separated overlap matrices
        # expand overlap if reordered matrix is larger than original matrix
        overlap_ab_11 = np.zeros(shape=(new_mode_num, new_mode_num), dtype = np.complex64)
        overlap_ab_12 = np.zeros(shape=(new_mode_num, new_mode_num), dtype = np.complex64)
        overlap_ab_21 = np.zeros(shape=(new_mode_num, new_mode_num), dtype = np.complex64)
        overlap_ab_22 = np.zeros(shape=(new_mode_num, new_mode_num), dtype = np.complex64)

        overlap_ba_11 = np.zeros(shape=(new_mode_num, new_mode_num), dtype = np.complex64)
        overlap_ba_12 = np.zeros(shape=(new_mode_num, new_mode_num), dtype = np.complex64)
        overlap_ba_21 = np.zeros(shape=(new_mode_num, new_mode_num), dtype = np.complex64)
        overlap_ba_22 = np.zeros(shape=(new_mode_num, new_mode_num), dtype = np.complex64)

        # Separate overlap matrices
        overlap_ab_11[:original_mode_num, :original_mode_num] = overlap_ab[:original_mode_num, :original_mode_num]
        overlap_ab_12[:original_mode_num, :original_mode_num] = overlap_ab[:original_mode_num, original_mode_num:]
        overlap_ab_21[:original_mode_num, :original_mode_num] = overlap_ab[original_mode_num:, :original_mode_num]
        overlap_ab_22[:original_mode_num, :original_mode_num] = overlap_ab[original_mode_num:, original_mode_num:]

        overlap_ba_11[:original_mode_num, :original_mode_num] = overlap_ba[:original_mode_num, :original_mode_num]
        overlap_ba_12[:original_mode_num, :original_mode_num] = overlap_ba[:original_mode_num, original_mode_num:]
        overlap_ba_21[:original_mode_num, :original_mode_num] = overlap_ba[original_mode_num:, :original_mode_num]
        overlap_ba_22[:original_mode_num, :original_mode_num] = overlap_ba[original_mode_num:, original_mode_num:]

        # Initialize reordered overlap matrices with the new size
        reordered_ab_11 = np.zeros((new_mode_num, new_mode_num), dtype=overlap_ab_11.dtype)
        reordered_ab_12 = np.zeros((new_mode_num, new_mode_num), dtype=overlap_ab_12.dtype)
        reordered_ab_21 = np.zeros((new_mode_num, new_mode_num), dtype=overlap_ab_21.dtype)
        reordered_ab_22 = np.zeros((new_mode_num, new_mode_num), dtype=overlap_ab_22.dtype)

        reordered_ba_11 = np.zeros((new_mode_num, new_mode_num), dtype=overlap_ba_11.dtype)
        reordered_ba_12 = np.zeros((new_mode_num, new_mode_num), dtype=overlap_ba_12.dtype)
        reordered_ba_21 = np.zeros((new_mode_num, new_mode_num), dtype=overlap_ba_21.dtype)
        reordered_ba_22 = np.zeros((new_mode_num, new_mode_num), dtype=overlap_ba_22.dtype)

        # Reorder rows of overlap_ab and columns of overlap_ba
        for i in range(original_mode_num):
            index_num = tracking_mode_names[index, i]
            reordered_ab_11[index_num] = overlap_ab_11[i]
            reordered_ab_12[index_num] = overlap_ab_12[i]
            reordered_ab_21[index_num] = overlap_ab_21[i]
            reordered_ab_22[index_num] = overlap_ab_22[i]

            reordered_ba_11[:, index_num] = overlap_ba_11[:, i]
            reordered_ba_12[:, index_num] = overlap_ba_12[:, i]
            reordered_ba_21[:, index_num] = overlap_ba_21[:, i]
            reordered_ba_22[:, index_num] = overlap_ba_22[:, i]

        # Reorder columns of overlap_ab and rows of overlap_ba
        final_ab_11 = np.zeros_like(reordered_ab_11)
        final_ab_12 = np.zeros_like(reordered_ab_12)
        final_ab_21 = np.zeros_like(reordered_ab_21)
        final_ab_22 = np.zeros_like(reordered_ab_22)

        final_ba_11 = np.zeros_like(reordered_ba_11)
        final_ba_12 = np.zeros_like(reordered_ba_12)
        final_ba_21 = np.zeros_like(reordered_ba_21)
        final_ba_22 = np.zeros_like(reordered_ba_22)

        for i in range(original_mode_num):
            index_num = new_tracking_mode_names[i]
            if index_num == -1: continue    # maybe this code cause the bug
            final_ab_11[:, index_num] = reordered_ab_11[:, i]
            final_ab_12[:, index_num] = reordered_ab_12[:, i]
            final_ab_21[:, index_num] = reordered_ab_21[:, i]
            final_ab_22[:, index_num] = reordered_ab_22[:, i]

            final_ba_11[index_num, :] = reordered_ba_11[i, :]
            final_ba_12[index_num, :] = reordered_ba_12[i, :]
            final_ba_21[index_num, :] = reordered_ba_21[i, :]
            final_ba_22[index_num, :] = reordered_ba_22[i, :]

        # Merge reordered matrices
        reordered_overlap_ab = np.zeros((2 * new_mode_num, 2 * new_mode_num), dtype=overlap_ab.dtype)
        reordered_overlap_ba = np.zeros((2 * new_mode_num, 2 * new_mode_num), dtype=overlap_ba.dtype)

        reordered_overlap_ab[:new_mode_num, :new_mode_num] = final_ab_11
        reordered_overlap_ab[:new_mode_num, new_mode_num:] = final_ab_12
        reordered_overlap_ab[new_mode_num:, :new_mode_num] = final_ab_21
        reordered_overlap_ab[new_mode_num:, new_mode_num:] = final_ab_22

        reordered_overlap_ba[:new_mode_num, :new_mode_num] = final_ba_11
        reordered_overlap_ba[:new_mode_num, new_mode_num:] = final_ba_12
        reordered_overlap_ba[new_mode_num:, :new_mode_num] = final_ba_21
        reordered_overlap_ba[new_mode_num:, new_mode_num:] = final_ba_22

        return reordered_overlap_ab, reordered_overlap_ba

        
    def _equalize_overlap_phase(self, overlap_ab, overlap_ba, overlap_dict = 0, index_mapping = 0):
        """
        Give every mode one sign per section, the same in both overlap sets.

        The mask of section i+1 is chosen so that ``Re diag(overlap_ab[i])``
        comes out positive for the forward and negative for the backward
        modes, and the same mask multiplies that section in ``overlap_ba``. A
        sign belongs to the mode's field, and the interface formulas combine
        ``O_ab`` with ``O_ba``, so both sets must carry the same one. Upstream
        took a second, independent set of masks from ``diag(overlap_ba)``: in a
        lossless basis ``O_ba ~ O_ab^T`` and the two agree, but on a PML basis a
        radiating branch can have diagonals of opposite sign in the two sets,
        and from there on it carried opposite signs in them (README, "What
        changed relative to upstream").

        Parameters:
            - overlap_ab : ndarray, ``overlap_ab[i] = <E(i), H(i+1)>`` - section i
                in its rows, section i+1 in its columns
            - overlap_ba : ndarray, ``overlap_ba[i] = <E(i+1), H(i)>`` - section
                i+1 in its rows, section i in its columns
            - overlap_dict: dictionary, two keys: "ab" and "ba"
                overlap_dict["ab"]: dict where multi_adj index in simul_params is key and list of overlaps is value
                overlap_dict["ba"]: dict where multi_adj index in simul_params is key and list of overlaps is value
            - index_mapping : dictionary mapping from EME_path index to simul_params_index

        The ``overlap_dict`` entries take the same masks as upstream gave them,
        now from ``overlap_ab`` for both sets; with a many-to-one
        ``index_mapping`` those are cumulative rather than the path section's
        own.  No propagator reads the dict.
        """
        if overlap_dict == 0:
            overlap_dict = {"ab": dict(), "ba": dict()}   # dummy overlap_dict
            index_mapping = {i: i for i in range(len(overlap_ab))}  # dummmy index mapping

        section_num, mode_count, _ = overlap_ab.shape
        mode_count = int(mode_count/2)
        backward_mode_mask = np.ones(shape=(mode_count,))
        backward_mode_mask = np.concatenate((backward_mode_mask, -backward_mode_mask))

        previous = np.ones(2 * mode_count)

        def section_mask(overlap):
            # an exactly zero diagonal says nothing about the sign: keep the previous
            # section's (np.sign(0) as a mask would delete the mode for the rest of the path)
            nonlocal previous
            mask = np.sign(np.diagonal(overlap).real) * backward_mode_mask
            previous = np.where(mask == 0, previous, mask)
            return previous

        for i in range(len(overlap_ab) - 1):
            # mask of section i+1, from the diagonal of the i-th ab overlap
            mask = section_mask(overlap_ab[i])

            # mask needs to be reshaped to broadcast correctly along rows and columns
            mask_row = mask[:, np.newaxis]  # Shape (m, 1) for broadcasting across columns
            mask_col = mask[np.newaxis, :]   # Shape (1, m) is fine for broadcasting across rows

            # section i+1 is the columns of ab[i] and the rows of ab[i+1] ...
            overlap_ab[i] *= mask_col
            overlap_ab[i + 1] *= mask_row
            if index_mapping[i] in overlap_dict["ab"]:
                overlap_dict["ab"][index_mapping[i]] *= mask_col
            if index_mapping[i+1] in overlap_dict["ab"]:
                overlap_dict["ab"][index_mapping[i+1]] *= mask_row

            # ... and the rows of ba[i] and the columns of ba[i+1]
            overlap_ba[i] *= mask_row
            overlap_ba[i + 1] *= mask_col
            if index_mapping[i] in overlap_dict["ba"]:
                overlap_dict["ba"][index_mapping[i]] *= mask_row
            if index_mapping[i+1] in overlap_dict["ba"]:
                overlap_dict["ba"][index_mapping[i+1]] *= mask_col

        if len(overlap_ab) > 0:
            # the last section: columns of the last ab overlap, rows of the last ba overlap
            last = len(overlap_ab) - 1
            mask = section_mask(overlap_ab[last])
            overlap_ab[last] *= mask[np.newaxis, :]
            overlap_ba[last] *= mask[:, np.newaxis]

        return overlap_ab, overlap_ba, overlap_dict

    #endregion reorder data, overlaps & equalize phase
    
    #region functions for some minor tasks in calc_output_data (_check_data_validity, _reduce_mode_number)

    def _check_data_validity(self):
        """
        check whether the data and geometry type is matched
        """
        pass

    def _reduce_mode_number(self, neff, TE_pol, overlap_ab, overlap_ba):
        num_sections, mode_numbers = neff.shape
        mode_numbers = int(mode_numbers/2)

        overlap_ab_new = np.zeros(shape = (num_sections-1, 2*self._limit_mode_number, 2*self._limit_mode_number), dtype = np.complex64)
        overlap_ba_new = np.zeros(shape = (num_sections-1, 2*self._limit_mode_number, 2*self._limit_mode_number), dtype = np.complex64)
        neff_new = np.zeros(shape = (num_sections, 2*self._limit_mode_number), dtype=np.complex64)
        TE_pol_new = np.zeros(shape=(num_sections, 2*self._limit_mode_number), dtype=np.float16)

        overlap_ab_new[:, :self._limit_mode_number, :self._limit_mode_number] = overlap_ab[:,:self._limit_mode_number,:self._limit_mode_number]
        overlap_ab_new[:, :self._limit_mode_number, self._limit_mode_number:] = \
            overlap_ab[:,:self._limit_mode_number,mode_numbers:mode_numbers+self._limit_mode_number]
        overlap_ab_new[:, self._limit_mode_number:, :self._limit_mode_number] = \
            overlap_ab[:,mode_numbers:mode_numbers+self._limit_mode_number,:self._limit_mode_number]
        overlap_ab_new[:, self._limit_mode_number:, self._limit_mode_number:] = \
            overlap_ab[:,mode_numbers:mode_numbers+self._limit_mode_number, mode_numbers:mode_numbers+self._limit_mode_number]
        
        overlap_ba_new[:, :self._limit_mode_number, :self._limit_mode_number] = overlap_ba[:,:self._limit_mode_number,:self._limit_mode_number]
        overlap_ba_new[:, :self._limit_mode_number, self._limit_mode_number:] = \
            overlap_ba[:,:self._limit_mode_number,mode_numbers:mode_numbers+self._limit_mode_number]
        overlap_ba_new[:, self._limit_mode_number:, :self._limit_mode_number] = \
            overlap_ba[:,mode_numbers:mode_numbers+self._limit_mode_number,:self._limit_mode_number]
        overlap_ba_new[:, self._limit_mode_number:, self._limit_mode_number:] = \
            overlap_ba[:,mode_numbers:mode_numbers+self._limit_mode_number, mode_numbers:mode_numbers+self._limit_mode_number]
        neff_new[:,:self._limit_mode_number] = neff[:,:self._limit_mode_number]
        neff_new[:,self._limit_mode_number:] = neff[:, mode_numbers:mode_numbers+self._limit_mode_number]
        TE_pol_new[:,:self._limit_mode_number] = TE_pol[:,:self._limit_mode_number]
        TE_pol_new[:,self._limit_mode_number:] = TE_pol[:, mode_numbers:mode_numbers+self._limit_mode_number]
        return neff_new, TE_pol_new, overlap_ab_new, overlap_ba_new
    

    def _reduce_mode_number_overlap_dict(self, overlap_dict):
        if len(overlap_dict["ab"]) == 0:
            return overlap_dict
        
        limit = self._limit_mode_number

        for key, overlap_ab in overlap_dict["ab"].items():
            # `_get_additional_overlap_dict` builds these as *lists* of
            # (2N, 2N) matrices, one per mutual adjacent point.
            overlap_ab = np.asarray(overlap_ab)
            num_adj_pt, mode_num, _ = overlap_ab.shape
            mode_num = mode_num//2
            overlap_ab_new = np.zeros((num_adj_pt, 2 * limit, 2 * limit), dtype=np.complex64)

            overlap_ab_new[:, :limit, :limit] = overlap_ab[:, :limit, :limit]
            overlap_ab_new[:, :limit, limit:] = overlap_ab[:, :limit, mode_num:mode_num + limit]
            overlap_ab_new[:, limit:, :limit] = overlap_ab[:, mode_num:mode_num + limit, :limit]
            overlap_ab_new[:, limit:, limit:] = overlap_ab[:, mode_num:mode_num + limit, mode_num:mode_num + limit]

            # Stored back as a list, not an array: `match_mutual_adj_overlap`
            # later replaces individual entries with re-ordered matrices whose
            # size is set by the number of *tracked* modes, which need not
            # match the limit.  A list tolerates that; an ndarray does not.
            overlap_dict["ab"][key] = list(overlap_ab_new)

        for key, overlap_ba in overlap_dict["ba"].items():
            overlap_ba = np.asarray(overlap_ba)
            num_adj_pt, mode_num, _ = overlap_ba.shape
            mode_num = mode_num//2
            overlap_ba_new = np.zeros((num_adj_pt, 2 * limit, 2 * limit), dtype=np.complex64)

            overlap_ba_new[:, :limit, :limit] = overlap_ba[:, :limit, :limit]
            overlap_ba_new[:, :limit, limit:] = overlap_ba[:, :limit, mode_num:mode_num + limit]
            overlap_ba_new[:, limit:, :limit] = overlap_ba[:, mode_num:mode_num + limit, :limit]
            overlap_ba_new[:, limit:, limit:] = overlap_ba[:, mode_num:mode_num + limit, mode_num:mode_num + limit]

            overlap_dict["ba"][key] = list(overlap_ba_new)

        return overlap_dict


    
    #endregion functions for some minor tasks in calc_output_data (_check_data_validity, _reduce_mode_number)

    #region functions for child class usage
    def scan_and_store(self, z_values, function, reference_values):
        interpolated_values = function(z_values)
        # Find the closest reference_values after continuous scanning
        values = np.array([reference_values[np.argmin(np.abs(reference_values - val))] for val in interpolated_values])
        
        # Remove adjacent duplicates
        result_propagation = [z_values[0]]
        result_values = [values[0]]
        for i in range(1, len(values)):
            if values[i] != values[i-1]:
                result_propagation.append(z_values[i])
                result_values.append(values[i])
            # if values[i] != values[i-1]:
            #     result_propagation.append(z_values[i])
            #     result_values.append(values[i])
        return np.array(result_propagation), np.array(result_values)
    
    def interpolate_values(self, propagation_tot, propagation, values):
        values_tot = []
        for z in propagation_tot:
            closest_index = np.searchsorted(propagation, z, side="right") - 1
            if closest_index >= 0 and closest_index < len(values):
                values_tot.append(values[closest_index])
            else:
                values_tot.append(values[-1] if len(values) > 0 else 0)
        return np.array(values_tot)
    
    def find_nearest_value(self, value, reference_values):
        """
        Parameter:
            - value:
            - reference_values
        Return
            - nearest value : value in the reference_values that is nearest to the input value
        """
        nearest_value = reference_values[np.argmin(np.abs(reference_values - value))]
        return nearest_value
    
    def remove_duplicate_points(self, prop_lengths, *values):
        """
        Remove duplicate values where value[i] == value[i+1] where value is a list in values
        It should be len(prop_lengths) == len(values[0]) == ... == len(values[-1])
        """
        num_values = len(values)
        new_prop_lengths = []
        new_values = [[] for i in range(num_values)]

        new_prop_lengths.append(prop_lengths[0])
        for j in range(num_values):
            new_values[j].append(values[j][0])

        for i in range(len(prop_lengths)-1):
            # check values are duplicated at i-th element and (i+1)th element
            is_duplicated = 1
            for j in range(num_values):
                is_duplicated *= (values[j][i] == values[j][i+1])
            
            # Put value if it is not duplicated
            if is_duplicated: continue
            new_prop_lengths.append(prop_lengths[i+1])
            for j in range(num_values):
                new_values[j].append(values[j][i+1])

        new_prop_lengths = np.array(new_prop_lengths)
        new_values = np.array(new_values)
        
        return new_prop_lengths, new_values            
        


    #endregion functions for child class usage

    #region code test
    def plot_overlap_along_prop(self, mode_num):
        num_sections, mode_count = self.output_data["neff"].shape
        mode_count = int(mode_count/2)
        if mode_num > 0 and mode_num < mode_count + 1: 
            mode_num_title = mode_num-1
            mode_num -= 1
        elif mode_num < 0 and mode_num > -(mode_count + 1):
            mode_num_title = -mode_num + 1
            mode_num = mode_count - mode_num -1
        
        plt.plot(self.output_data["overlap_ab"][:,mode_num,mode_num], label= "overlap_ab")
        plt.plot(self.output_data["overlap_ba"][:,mode_num,mode_num], label= "overlap_ba")
        plt.title("Overlap along propagation of same mode " + "(mode" + str(mode_num_title) + ")")
        plt.legend()
        plt.show()

        pass
    #endregion code test

    
