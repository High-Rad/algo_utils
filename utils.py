from typing import Tuple, List, Collection

import numpy as np
from tqdm.contrib.concurrent import process_map

# todo ???? >>
from algo_typing import IndexExpression3D
from file_operations import load_nifti_data
from segmentation_processing import expand_labels, expand_per_label
# todo ???? <<

# todo - matching
def find_pairs(baseline_moved_labeled, followup_labeled, reverse=False, voxelspacing=None, max_dilate_param=5,
               return_iteration_and_reverse_indicator=False):
    working_baseline_moved_labeled = baseline_moved_labeled
    working_followup_labeled = followup_labeled

    distance_cache_bl, distance_cache_fu = None, None

    bl_matched_tumors, fu_matched_tumors = [], []

    list_of_pairs = []
    # Hyper-parameter for sensitivity of the matching (5 means that 10 pixels between the scans will be same)
    for dilate in range(max_dilate_param):

        # find intersection areas between baseline and followup
        matched = np.logical_and((working_baseline_moved_labeled > 0), working_followup_labeled > 0)

        # iterate over tumors in baseline that intersect tumors in followup
        for i in np.unique((matched * working_baseline_moved_labeled).flatten()):
            if i == 0:
                continue

            # find intersection between current BL tumor and FU tumors
            overlap = ((working_baseline_moved_labeled == i) * working_followup_labeled)

            # get the labels of the FU tumors that intersect the current BL tumor
            follow_up_num = np.unique(overlap.flatten())

            # in case there is more than 1 FU tumor that intersect the current BL tumor
            if follow_up_num.shape[0] > 2:
                sum = 0
                # iterate over the FU tumors that intersect the current BL tumor
                for j in follow_up_num[1:]:
                    # in case the current FU tumor has the biggest found overlap with current BL tumor,
                    # till this iteration
                    if sum < (overlap == j).sum():
                        sum = (overlap == j).sum()
                        biggest = j

                    # in case the overlap of the current FU tumor with the current BL tumor
                    # is grader than 10% of the BL or FU tumor
                    elif ((overlap == j).sum() / (working_followup_labeled == j).sum()) > 0.1 or (
                            (overlap == j).sum() / (working_baseline_moved_labeled == i).sum()) > 0.1:
                        # a match was found
                        if reverse:
                            if return_iteration_and_reverse_indicator:
                                list_of_pairs.append((int(reverse), dilate + 1, (j, i)))
                            else:
                                list_of_pairs.append((j, i))
                        else:
                            if return_iteration_and_reverse_indicator:
                                list_of_pairs.append((int(reverse), dilate + 1, (i, j)))
                            else:
                                list_of_pairs.append((i, j))
                        # zero the current FU tumor and the current BL tumor
                        bl_matched_tumors.append(i)
                        fu_matched_tumors.append(j)
                        working_baseline_moved_labeled[working_baseline_moved_labeled == i] = 0
                        working_followup_labeled[working_followup_labeled == j] = 0
                # marking the FU tumor with the biggest overlap with the current BL tumor as a found match
                if reverse:
                    if return_iteration_and_reverse_indicator:
                        list_of_pairs.append((int(reverse), dilate + 1, (biggest, i)))
                    else:
                        list_of_pairs.append((biggest, i))
                else:
                    if return_iteration_and_reverse_indicator:
                        list_of_pairs.append((int(reverse), dilate + 1, (i, biggest)))
                    else:
                        list_of_pairs.append((i, biggest))

                # zero the current BL tumor and the FU tumor that has jost been
                # marked as a match with the current BL tumor
                bl_matched_tumors.append(i)
                fu_matched_tumors.append(biggest)
                working_baseline_moved_labeled[working_baseline_moved_labeled == i] = 0
                working_followup_labeled[working_followup_labeled == biggest] = 0

            # in case there is only 1 FU tumor that intersect the current BL tumor
            elif follow_up_num.shape[0] > 1:
                # a match was found
                if reverse:
                    if return_iteration_and_reverse_indicator:
                        list_of_pairs.append((int(reverse), dilate + 1, (follow_up_num[-1], i)))
                    else:
                        list_of_pairs.append((follow_up_num[-1], i))
                else:
                    if return_iteration_and_reverse_indicator:
                        list_of_pairs.append((int(reverse), dilate + 1, (i, follow_up_num[-1])))
                    else:
                        list_of_pairs.append((i, follow_up_num[-1]))

                # zero the current BL tumor and the FU tumor that intersects it
                bl_matched_tumors.append(i)
                fu_matched_tumors.append(follow_up_num[-1])
                working_baseline_moved_labeled[working_baseline_moved_labeled == i] = 0
                working_followup_labeled[working_followup_labeled == follow_up_num[-1]] = 0

        if dilate == (max_dilate_param - 1) or np.all(working_baseline_moved_labeled == 0) or np.all(
                working_followup_labeled == 0):
            break

        # dilation without overlap and considering resolution
        working_baseline_moved_labeled, distance_cache_bl = expand_labels(baseline_moved_labeled, distance=dilate + 1,
                                                                          voxelspacing=voxelspacing,
                                                                          distance_cache=distance_cache_bl,
                                                                          return_distance_cache=True)
        working_baseline_moved_labeled[np.isin(working_baseline_moved_labeled, bl_matched_tumors)] = 0
        working_followup_labeled, distance_cache_fu = expand_labels(followup_labeled, distance=dilate + 1,
                                                                    voxelspacing=voxelspacing,
                                                                    distance_cache=distance_cache_fu,
                                                                    return_distance_cache=True)
        working_followup_labeled[np.isin(working_followup_labeled, fu_matched_tumors)] = 0

    return (list_of_pairs)

# todo - matching
def match_2_cases(BL_tumors_labels, FU_tumors_labels, voxelspacing=None, max_dilate_param=5,
                  return_iteration_and_reverse_indicator=False):
    """
    • This version removes the tumors immediately after their match, not at the end of the iteration (as a result,
        the number of the tumors may affect the final results. Additionally, it requires one check as (bl=BL, fu=FU) and
        one check as (bl=FU, fu=BL)).
    • This version works with python 'for' iterations and not with numpy optimizations.
    """
    first = BL_tumors_labels.copy()
    second = FU_tumors_labels.copy()

    list_of_pairs = find_pairs(first, second, voxelspacing=voxelspacing, max_dilate_param=max_dilate_param,
                               return_iteration_and_reverse_indicator=return_iteration_and_reverse_indicator)

    first = BL_tumors_labels.copy()
    second = FU_tumors_labels.copy()
    list_of_pairs2 = find_pairs(second, first, reverse=True, voxelspacing=voxelspacing,
                                max_dilate_param=max_dilate_param,
                                return_iteration_and_reverse_indicator=return_iteration_and_reverse_indicator)

    resulting_list = list(list_of_pairs)
    if return_iteration_and_reverse_indicator:
        if len(resulting_list) > 0:
            _, _, resulting_matches = zip(*resulting_list)
            resulting_list.extend(x for x in list_of_pairs2 if x[2] not in resulting_matches)
        else:
            resulting_list = list(list_of_pairs2)
    else:
        resulting_list.extend(x for x in list_of_pairs2 if x not in resulting_list)

    return resulting_list

# todo - matching
def match_2_cases_v2(baseline_moved_labeled, followup_labeled, voxelspacing=None, max_dilate_param=5,
                     return_iteration_indicator=False):
    """
    • This version removes the tumors only at the end of the iterations.
    • This version works with minimum python 'for' iterations and mostly with numpy optimizations.
    """
    working_baseline_moved_labeled = baseline_moved_labeled
    working_followup_labeled = followup_labeled

    distance_cache_bl, distance_cache_fu = None, None

    if return_iteration_indicator:
        pairs = np.array([]).reshape([0, 3])
    else:
        pairs = np.array([]).reshape([0, 2])
    # Hyper-parameter for sensitivity of the matching (5 means that 10 pixels between the scans will be same)
    for dilate in range(max_dilate_param):

        # find pairs of intersection of tumors
        pairs_of_intersection = np.stack([working_baseline_moved_labeled, working_followup_labeled]).astype(np.int16)
        pairs_of_intersection, overlap_vol = np.unique(
            pairs_of_intersection[:, np.all(pairs_of_intersection != 0, axis=0)].T, axis=0, return_counts=True)

        if pairs_of_intersection.size > 0:

            relevant_bl_tumors, relevant_bl_tumors_vol = np.unique(
                working_baseline_moved_labeled[np.isin(working_baseline_moved_labeled, pairs_of_intersection[:, 0])],
                return_counts=True)
            relevant_fu_tumors, relevant_fu_tumors_vol = np.unique(
                working_followup_labeled[np.isin(working_followup_labeled, pairs_of_intersection[:, 1])],
                return_counts=True)

            # intersection_matrix_overlap_vol[i,j] = #voxels_in_the_overlap_of_relevant_bl_tumors[i]_and_relevant_fu_tumors[j]
            intersection_matrix_overlap_vol = np.zeros((relevant_bl_tumors.size, relevant_fu_tumors.size))
            intersection_matrix_overlap_vol[np.searchsorted(relevant_bl_tumors, pairs_of_intersection[:, 0]),
                                            np.searchsorted(relevant_fu_tumors,
                                                            pairs_of_intersection[:, 1])] = overlap_vol

            # intersection_matrix_overlap_with_bl_ratio[i,j] = #voxels_in_the_overlap_of_relevant_bl_tumors[i]_and_relevant_fu_tumors[j] / #voxels_in_relevant_bl_tumors[i]
            intersection_matrix_overlap_with_bl_ratio = intersection_matrix_overlap_vol / relevant_bl_tumors_vol.reshape(
                [-1, 1])

            # intersection_matrix_overlap_with_fu_ratio[i,j] = #voxels_in_the_overlap_of_relevant_bl_tumors[i]_and_relevant_fu_tumors[j] / #voxels_in_relevant_fu_tumors[j]
            intersection_matrix_overlap_with_fu_ratio = intersection_matrix_overlap_vol / relevant_fu_tumors_vol.reshape(
                [1, -1])

            # take as a match any maximum intersection with bl tumors
            current_pairs_inds = np.stack(
                [np.arange(relevant_bl_tumors.size), np.argmax(intersection_matrix_overlap_vol, axis=1)]).T

            # take as a match any maximum intersection with fu tumors
            current_pairs_inds = np.unique(np.concatenate([current_pairs_inds, np.stack(
                [np.argmax(intersection_matrix_overlap_vol, axis=0), np.arange(relevant_fu_tumors.size)]).T]), axis=0)

            # take as a match any intersection with a overlap ratio with either bl of fu above 10 percent
            current_pairs_inds = np.unique(np.concatenate(
                [current_pairs_inds, np.stack(np.where(intersection_matrix_overlap_with_bl_ratio > 0.1)).T]), axis=0)
            current_pairs_inds = np.unique(np.concatenate(
                [current_pairs_inds, np.stack(np.where(intersection_matrix_overlap_with_fu_ratio > 0.1)).T]), axis=0)

            if return_iteration_indicator:
                current_pairs = np.stack(
                    [relevant_bl_tumors[current_pairs_inds[:, 0]], relevant_fu_tumors[current_pairs_inds[:, 1]],
                     np.repeat(dilate + 1, current_pairs_inds.shape[0])]).T
            else:
                current_pairs = np.stack(
                    [relevant_bl_tumors[current_pairs_inds[:, 0]], relevant_fu_tumors[current_pairs_inds[:, 1]]]).T

            pairs = np.concatenate([pairs, current_pairs])

        if dilate == (max_dilate_param - 1):
            break

        # dilation without overlap, and considering resolution
        working_baseline_moved_labeled, distance_cache_bl = expand_labels(baseline_moved_labeled, distance=dilate + 1,
                                                                          voxelspacing=voxelspacing,
                                                                          distance_cache=distance_cache_bl,
                                                                          return_distance_cache=True)
        working_followup_labeled, distance_cache_fu = expand_labels(followup_labeled, distance=dilate + 1,
                                                                    voxelspacing=voxelspacing,
                                                                    distance_cache=distance_cache_fu,
                                                                    return_distance_cache=True)

        # zero the BL tumor and the FU tumor in the matches
        working_baseline_moved_labeled[np.isin(working_baseline_moved_labeled, pairs[:, 0])] = 0
        working_followup_labeled[np.isin(working_followup_labeled, pairs[:, 1])] = 0

        if np.all(working_baseline_moved_labeled == 0) or np.all(working_followup_labeled == 0):
            break

    if return_iteration_indicator:
        return [(p[2], (p[0], p[1])) for p in pairs]
    return [tuple(p) for p in pairs]

# todo - matching
def match_2_cases_v3(baseline_moved_labeled, followup_labeled, voxelspacing=None, max_dilate_param=5,
                     return_iteration_indicator=False):
    """
    • This version removes the tumors only at the end of the iterations.
    • This version works with minimum python 'for' iterations and mostly with numpy optimizations.
    • This version's match criteria is only the ratio of intersection between tumors, and it doesn't take a maximum
        intersection as a match.
    """
    working_baseline_moved_labeled = baseline_moved_labeled
    working_followup_labeled = followup_labeled

    distance_cache_bl, distance_cache_fu = None, None

    pairs = np.array([]).reshape([0, 3 if return_iteration_indicator else 2])
    # Hyper-parameter for sensitivity of the matching (5 means that 10 pixels between the scans will be same)
    for dilate in range(max_dilate_param):

        # find pairs of intersection of tumors
        pairs_of_intersection = np.stack([working_baseline_moved_labeled, working_followup_labeled]).astype(np.int16)
        pairs_of_intersection, overlap_vol = np.unique(
            pairs_of_intersection[:, np.all(pairs_of_intersection != 0, axis=0)].T, axis=0, return_counts=True)

        if pairs_of_intersection.size > 0:

            relevant_bl_tumors, relevant_bl_tumors_vol = np.unique(
                working_baseline_moved_labeled[np.isin(working_baseline_moved_labeled, pairs_of_intersection[:, 0])],
                return_counts=True)
            relevant_fu_tumors, relevant_fu_tumors_vol = np.unique(
                working_followup_labeled[np.isin(working_followup_labeled, pairs_of_intersection[:, 1])],
                return_counts=True)

            # intersection_matrix_overlap_vol[i,j] = #voxels_in_the_overlap_of_relevant_bl_tumors[i]_and_relevant_fu_tumors[j]
            intersection_matrix_overlap_vol = np.zeros((relevant_bl_tumors.size, relevant_fu_tumors.size))
            intersection_matrix_overlap_vol[np.searchsorted(relevant_bl_tumors, pairs_of_intersection[:, 0]),
                                            np.searchsorted(relevant_fu_tumors,
                                                            pairs_of_intersection[:, 1])] = overlap_vol

            # intersection_matrix_overlap_with_bl_ratio[i,j] = #voxels_in_the_overlap_of_relevant_bl_tumors[i]_and_relevant_fu_tumors[j] / #voxels_in_relevant_bl_tumors[i]
            intersection_matrix_overlap_with_bl_ratio = intersection_matrix_overlap_vol / relevant_bl_tumors_vol.reshape(
                [-1, 1])

            # intersection_matrix_overlap_with_fu_ratio[i,j] = #voxels_in_the_overlap_of_relevant_bl_tumors[i]_and_relevant_fu_tumors[j] / #voxels_in_relevant_fu_tumors[j]
            intersection_matrix_overlap_with_fu_ratio = intersection_matrix_overlap_vol / relevant_fu_tumors_vol.reshape(
                [1, -1])

            # take as a match any maximum intersection with bl tumors
            # current_pairs_inds = np.stack([np.arange(relevant_bl_tumors.size), np.argmax(intersection_matrix_overlap_vol, axis=1)]).T

            # take as a match any maximum intersection with fu tumors
            # current_pairs_inds = np.unique(np.concatenate([current_pairs_inds, np.stack([np.argmax(intersection_matrix_overlap_vol, axis=0), np.arange(relevant_fu_tumors.size)]).T]), axis=0)

            current_pairs_inds = np.array([], dtype=np.int16).reshape([0, 2])
            # take as a match any intersection with a overlap ratio with either bl of fu above 10 percent
            current_pairs_inds = np.unique(np.concatenate(
                [current_pairs_inds, np.stack(np.where(intersection_matrix_overlap_with_bl_ratio > 0.1)).T]), axis=0)
            current_pairs_inds = np.unique(np.concatenate(
                [current_pairs_inds, np.stack(np.where(intersection_matrix_overlap_with_fu_ratio > 0.1)).T]), axis=0)

            if return_iteration_indicator:
                current_pairs = np.stack(
                    [relevant_bl_tumors[current_pairs_inds[:, 0]], relevant_fu_tumors[current_pairs_inds[:, 1]],
                     np.repeat(dilate + 1, current_pairs_inds.shape[0])]).T
            else:
                current_pairs = np.stack(
                    [relevant_bl_tumors[current_pairs_inds[:, 0]], relevant_fu_tumors[current_pairs_inds[:, 1]]]).T

            pairs = np.concatenate([pairs, current_pairs])

        if dilate == (max_dilate_param - 1):
            break

        # dilation without overlap, and considering resolution
        working_baseline_moved_labeled, distance_cache_bl = expand_labels(baseline_moved_labeled, distance=dilate + 1,
                                                                          voxelspacing=voxelspacing,
                                                                          distance_cache=distance_cache_bl,
                                                                          return_distance_cache=True)
        working_followup_labeled, distance_cache_fu = expand_labels(followup_labeled, distance=dilate + 1,
                                                                    voxelspacing=voxelspacing,
                                                                    distance_cache=distance_cache_fu,
                                                                    return_distance_cache=True)

        # zero the BL tumor and the FU tumor in the matches
        working_baseline_moved_labeled[np.isin(working_baseline_moved_labeled, pairs[:, 0])] = 0
        working_followup_labeled[np.isin(working_followup_labeled, pairs[:, 1])] = 0

        if np.all(working_baseline_moved_labeled == 0) or np.all(working_followup_labeled == 0):
            break

    if return_iteration_indicator:
        return [(p[2], (p[0], p[1])) for p in pairs]
    return [tuple(p) for p in pairs]

# todo - matching
def match_2_cases_v4(baseline_moved_labeled, followup_labeled, voxelspacing=None, max_dilate_param=5,
                     return_iteration_indicator=False):
    """
    • This version removes the tumors only at the end of the iterations.
    • This version works with minimum python 'for' iterations and mostly with numpy optimizations.
    • This version's match criteria is only the ratio of intersection between tumors, and it doesn't take a maximum
        intersection as a match.
    • This version expands the tumors after each iteration relative to the tumors size.
    """

    def dists_to_expand(diameters: np.ndarray, i_1: float = 6, i_10: float = 3) -> np.ndarray:
        """
        The distance to expand is a function of the diameter.

        i_1 is the number of mm to expand for a tumor with a diameter of 1
        i_10 is the number of mm to expand for a tumor with a diameter of 10
        See here the visual graph: https://www.desmos.com/calculator/lznykxikim
        """

        b = (10 * i_10 - i_1) / 9
        c = ((i_1 ** 10) / i_10) ** (1 / 9)

        if isinstance(diameters, (int, float)):
            diameters = np.array([diameters])

        diameters = np.clip(diameters, 1, 20)
        dists = np.empty(diameters.shape, np.float32)

        # for tumors with diameter less or equal to 10, the distance to expand
        # as a function of the diameter is (i_1 - b)/x + b, where x is the diameter, and b and i_1 are defined above.
        dists[diameters <= 10] = ((i_1 - b) / diameters[diameters <= 10]) + b

        # for tumors with diameter greater than 10, the distance to expand
        # as a function of the diameter is c * (i_1/c)^x, where x is the diameter, and c and i_1 are defined above.
        dists[diameters > 10] = c * ((i_1 / c) ** diameters[diameters > 10])

        return dists

    def dists_to_expand_v2(diameters: np.ndarray, i: float = 5, j: float = 3, k: float = 0.05) -> np.ndarray:
        """
        The distance (number of mm) to expand is a function of the diameter. The following functions are in use:

        func_1: a/x + b, where x is the diameter.
        func_2: c/e^(dx), where x is the diameter.

        For a diameter between 1 and 10, func_1 is used, and for a diameter between 10 and 20, func_2 is used.
        All the diameters is clipped to range [1,20]

        The parameters, {a,b} for func_1 and {c,d} for func_2 is decided due to the given arguments.

        i is the number of mm to expand for a tumor with a diameter of 1
        j is the number of mm to expand for a tumor with a diameter of 10
        k is the number of mm to expand for a tumor with a diameter of 20
        See here the visual graph: https://www.desmos.com/calculator/dvokawlytl
        """

        if isinstance(diameters, (int, float)):
            diameters = np.array([diameters])

        diameters = np.clip(diameters, 1, 20)
        dists = np.empty(diameters.shape, np.float32)

        # for tumors with diameter less or equal to 10, the distance to expand
        # as a function of the diameter is 10(i-j)/9x + (10j-i)/9, where x is the diameter, and i and j are defined above.
        dists[diameters <= 10] = 10 * (i - j) / (9 * diameters[diameters <= 10]) + (10 * j - i) / 9

        # for tumors with diameter greater than 10, the distance to expand
        # as a function of the diameter is j * (j/k)^((10-x)/10), where x is the diameter, and j and k are defined above.
        dists[diameters > 10] = j * ((j / k) ** ((10 - diameters[diameters > 10]) / 10))

        return dists

    def extract_diameters(tumors_labeled_case: np.ndarray) -> np.ndarray:
        tumors_labels, tumors_vol = np.unique(tumors_labeled_case, return_counts=True)
        tumors_vol = tumors_vol[tumors_labels > 0] * np.asarray(voxelspacing).prod()
        return (6 * tumors_vol / np.pi) ** (1 / 3)

    bl_dists_to_expand = dists_to_expand_v2(extract_diameters(baseline_moved_labeled))
    bl_max_relevant_distances = max_dilate_param * bl_dists_to_expand

    fu_dists_to_expand = dists_to_expand_v2(extract_diameters(followup_labeled))
    fu_max_relevant_distances = max_dilate_param * fu_dists_to_expand

    # working_baseline_moved_labeled = baseline_moved_labeled
    # working_followup_labeled = followup_labeled

    distance_cache_bl, distance_cache_fu = None, None

    pairs = np.array([]).reshape([0, 3 if return_iteration_indicator else 2])
    # Hyper-parameter for sensitivity of the matching (5 means that 10 pixels between the scans will be same)
    for dilate in range(max_dilate_param):

        # dilation without overlap, and considering resolution
        working_baseline_moved_labeled, distance_cache_bl = expand_per_label(baseline_moved_labeled,
                                                                             dists_to_expand=(
                                                                                                         dilate + 1) * bl_dists_to_expand,
                                                                             max_relevant_distances=bl_max_relevant_distances,
                                                                             voxelspacing=voxelspacing,
                                                                             distance_cache=distance_cache_bl,
                                                                             return_distance_cache=True)
        working_followup_labeled, distance_cache_fu = expand_per_label(followup_labeled,
                                                                       dists_to_expand=(
                                                                                                   dilate + 1) * fu_dists_to_expand,
                                                                       max_relevant_distances=fu_max_relevant_distances,
                                                                       voxelspacing=voxelspacing,
                                                                       distance_cache=distance_cache_fu,
                                                                       return_distance_cache=True)

        # zero the BL tumor and the FU tumor in the matches
        working_baseline_moved_labeled[np.isin(working_baseline_moved_labeled, pairs[:, 0])] = 0
        working_followup_labeled[np.isin(working_followup_labeled, pairs[:, 1])] = 0

        # find pairs of intersection of tumors
        pairs_of_intersection = np.stack([working_baseline_moved_labeled, working_followup_labeled]).astype(np.int16)
        pairs_of_intersection, overlap_vol = np.unique(
            pairs_of_intersection[:, np.all(pairs_of_intersection != 0, axis=0)].T, axis=0, return_counts=True)

        if pairs_of_intersection.size > 0:

            relevant_bl_tumors, relevant_bl_tumors_vol = np.unique(
                working_baseline_moved_labeled[np.isin(working_baseline_moved_labeled, pairs_of_intersection[:, 0])],
                return_counts=True)
            relevant_fu_tumors, relevant_fu_tumors_vol = np.unique(
                working_followup_labeled[np.isin(working_followup_labeled, pairs_of_intersection[:, 1])],
                return_counts=True)

            # intersection_matrix_overlap_vol[i,j] = #voxels_in_the_overlap_of_relevant_bl_tumors[i]_and_relevant_fu_tumors[j]
            intersection_matrix_overlap_vol = np.zeros((relevant_bl_tumors.size, relevant_fu_tumors.size))
            intersection_matrix_overlap_vol[np.searchsorted(relevant_bl_tumors, pairs_of_intersection[:, 0]),
                                            np.searchsorted(relevant_fu_tumors,
                                                            pairs_of_intersection[:, 1])] = overlap_vol

            # intersection_matrix_overlap_with_bl_ratio[i,j] = #voxels_in_the_overlap_of_relevant_bl_tumors[i]_and_relevant_fu_tumors[j] / #voxels_in_relevant_bl_tumors[i]
            intersection_matrix_overlap_with_bl_ratio = intersection_matrix_overlap_vol / relevant_bl_tumors_vol.reshape(
                [-1, 1])

            # intersection_matrix_overlap_with_fu_ratio[i,j] = #voxels_in_the_overlap_of_relevant_bl_tumors[i]_and_relevant_fu_tumors[j] / #voxels_in_relevant_fu_tumors[j]
            intersection_matrix_overlap_with_fu_ratio = intersection_matrix_overlap_vol / relevant_fu_tumors_vol.reshape(
                [1, -1])

            # take as a match any maximum intersection with bl tumors
            # current_pairs_inds = np.stack([np.arange(relevant_bl_tumors.size), np.argmax(intersection_matrix_overlap_vol, axis=1)]).T

            # take as a match any maximum intersection with fu tumors
            # current_pairs_inds = np.unique(np.concatenate([current_pairs_inds, np.stack([np.argmax(intersection_matrix_overlap_vol, axis=0), np.arange(relevant_fu_tumors.size)]).T]), axis=0)

            current_pairs_inds = np.array([], dtype=np.int16).reshape([0, 2])
            # take as a match any intersection with a overlap ratio with either bl of fu above 10 percent
            current_pairs_inds = np.unique(np.concatenate(
                [current_pairs_inds, np.stack(np.where(intersection_matrix_overlap_with_bl_ratio > 0.1)).T]), axis=0)
            current_pairs_inds = np.unique(np.concatenate(
                [current_pairs_inds, np.stack(np.where(intersection_matrix_overlap_with_fu_ratio > 0.1)).T]), axis=0)

            if return_iteration_indicator:
                current_pairs = np.stack(
                    [relevant_bl_tumors[current_pairs_inds[:, 0]], relevant_fu_tumors[current_pairs_inds[:, 1]],
                     np.repeat(dilate + 1, current_pairs_inds.shape[0])]).T
            else:
                current_pairs = np.stack(
                    [relevant_bl_tumors[current_pairs_inds[:, 0]], relevant_fu_tumors[current_pairs_inds[:, 1]]]).T

            pairs = np.concatenate([pairs, current_pairs])

        if np.all(working_baseline_moved_labeled == 0) or np.all(working_followup_labeled == 0):
            break

    if return_iteration_indicator:
        return [(p[2], (p[0], p[1])) for p in pairs]
    return [tuple(p) for p in pairs]

# todo - matching
def match_2_cases_v5(baseline_moved_labeled, followup_labeled, voxelspacing=None, max_dilate_param=5,
                     return_iteration_indicator=False):
    """
    • This version removes the tumors only at the end of the iterations.
    • This version works with minimum python 'for' iterations and mostly with numpy optimizations.
    • This version's match criteria is only the ratio of intersection between tumors, and it doesn't take a maximum
        intersection as a match.
    • This version dilates the images ones in the beginning.
    """

    if np.all(baseline_moved_labeled == 0) or np.all(followup_labeled == 0):
        return []

    distance_cache_bl, distance_cache_fu = None, None

    pairs = np.array([]).reshape([0, 3 if return_iteration_indicator else 2])
    # Hyper-parameter for sensitivity of the matching (5 means that 10 pixels between the scans will be same)
    for dilate in range(max_dilate_param):

        # dilation without overlap, and considering resolution
        working_baseline_moved_labeled, distance_cache_bl = expand_labels(baseline_moved_labeled, distance=dilate + 1,
                                                                          voxelspacing=voxelspacing,
                                                                          distance_cache=distance_cache_bl,
                                                                          return_distance_cache=True)
        working_followup_labeled, distance_cache_fu = expand_labels(followup_labeled, distance=dilate + 1,
                                                                    voxelspacing=voxelspacing,
                                                                    distance_cache=distance_cache_fu,
                                                                    return_distance_cache=True)

        if dilate > 0:
            # zero the BL tumor and the FU tumor in the matches
            working_baseline_moved_labeled[np.isin(working_baseline_moved_labeled, pairs[:, 0])] = 0
            working_followup_labeled[np.isin(working_followup_labeled, pairs[:, 1])] = 0

            if np.all(working_baseline_moved_labeled == 0) or np.all(working_followup_labeled == 0):
                break

        # find pairs of intersection of tumors
        pairs_of_intersection = np.stack([working_baseline_moved_labeled, working_followup_labeled]).astype(np.int16)
        pairs_of_intersection, overlap_vol = np.unique(
            pairs_of_intersection[:, np.all(pairs_of_intersection != 0, axis=0)].T, axis=0, return_counts=True)

        if pairs_of_intersection.size > 0:

            relevant_bl_tumors, relevant_bl_tumors_vol = np.unique(
                working_baseline_moved_labeled[np.isin(working_baseline_moved_labeled, pairs_of_intersection[:, 0])],
                return_counts=True)
            relevant_fu_tumors, relevant_fu_tumors_vol = np.unique(
                working_followup_labeled[np.isin(working_followup_labeled, pairs_of_intersection[:, 1])],
                return_counts=True)

            # intersection_matrix_overlap_vol[i,j] = #voxels_in_the_overlap_of_relevant_bl_tumors[i]_and_relevant_fu_tumors[j]
            intersection_matrix_overlap_vol = np.zeros((relevant_bl_tumors.size, relevant_fu_tumors.size))
            intersection_matrix_overlap_vol[np.searchsorted(relevant_bl_tumors, pairs_of_intersection[:, 0]),
                                            np.searchsorted(relevant_fu_tumors,
                                                            pairs_of_intersection[:, 1])] = overlap_vol

            # intersection_matrix_overlap_with_bl_ratio[i,j] = #voxels_in_the_overlap_of_relevant_bl_tumors[i]_and_relevant_fu_tumors[j] / #voxels_in_relevant_bl_tumors[i]
            intersection_matrix_overlap_with_bl_ratio = intersection_matrix_overlap_vol / relevant_bl_tumors_vol.reshape(
                [-1, 1])

            # intersection_matrix_overlap_with_fu_ratio[i,j] = #voxels_in_the_overlap_of_relevant_bl_tumors[i]_and_relevant_fu_tumors[j] / #voxels_in_relevant_fu_tumors[j]
            intersection_matrix_overlap_with_fu_ratio = intersection_matrix_overlap_vol / relevant_fu_tumors_vol.reshape(
                [1, -1])

            current_pairs_inds = np.array([], dtype=np.int16).reshape([0, 2])
            # take as a match any intersection with a overlap ratio with either bl of fu above 10 percent
            current_pairs_inds = np.unique(np.concatenate(
                [current_pairs_inds, np.stack(np.where(intersection_matrix_overlap_with_bl_ratio > 0.1)).T]), axis=0)
            current_pairs_inds = np.unique(np.concatenate(
                [current_pairs_inds, np.stack(np.where(intersection_matrix_overlap_with_fu_ratio > 0.1)).T]), axis=0)

            if return_iteration_indicator:
                current_pairs = np.stack(
                    [relevant_bl_tumors[current_pairs_inds[:, 0]], relevant_fu_tumors[current_pairs_inds[:, 1]],
                     np.repeat(dilate + 1, current_pairs_inds.shape[0])]).T
            else:
                current_pairs = np.stack(
                    [relevant_bl_tumors[current_pairs_inds[:, 0]], relevant_fu_tumors[current_pairs_inds[:, 1]]]).T

            pairs = np.concatenate([pairs, current_pairs])

    if return_iteration_indicator:
        return [(p[2], (p[0], p[1])) for p in pairs]
    return [tuple(p) for p in pairs]

# todo image-processing
def is_a_scan(case: np.ndarray) -> bool:
    if np.unique(case).size <= 2:
        return False
    return True

# todo image-processing
def is_a_mask(case: np.ndarray) -> bool:
    if np.any((case != 0) & (case != 1)):
        return False
    return True

# todo image-processing
def is_a_labeled_mask(case: np.ndarray, relevant_labels: Collection) -> bool:
    if np.all(np.isin(case, relevant_labels)):
        return True
    return False

# todo image-processing
def __get_mean_and_std(scan: str):
    try:
        s, _ = load_nifti_data(scan)
        return s.mean(), s.std()
    except Exception:
        return np.nan, np.nan

# todo image-processing
def get_duplicate_scans(scans: List[str], multiprocess: bool = False):
    if multiprocess:
        params = np.round(process_map(__get_mean_and_std, scans), 4)
    else:
        params = np.round([__get_mean_and_std(s) for s in scans], 4)
    params = params[~np.isnan(params).any(axis=1)]
    _, inds, counts = np.unique(params, return_inverse=True, return_counts=True, axis=0)

    duplicates = []
    for i, c in enumerate(counts):
        if c > 1:
            duplicates.append(list(np.asarray(scans)[np.where(inds == i)]))
    return duplicates


# todo image-processing
def find_joint_z_slices(im1: np.ndarray, im2: np.ndarray) -> Tuple[IndexExpression3D, IndexExpression3D]:
    """
    Find overlap joint z-slices between two images.

    Notes
    _____
    Assumes that im1 and im2 are 3D images similar in x and y shape.

    Parameters
    ----------
    im1: numpy.ndarray
        1st 3D image.
    im2: numpy.ndarray
        2nd 3D image.

    Returns
    -------
    s1, s2 : IndexExpression3D
        The 3D index expressions (Tuple[slice, slice, slice]) of the joint slices in im1 and im2, respectively. They are
        None if there is no overlap between the images.
    """

    # ensure images have the same shape over x and y axes
    assert np.all(np.asarray(im1.shape[0:2]) == np.asarray(im2.shape[0:2]))

    # create the 2D window bounding box
    x_ind, y_ind = np.asarray(im1.shape[0:2]) // 2
    window_size = 20
    candidate_2d_window = np.s_[x_ind: x_ind + window_size, y_ind: y_ind + window_size]

    # there is an overlap between im1 and im2 over the z axis, iff one of the following holds:
    # 1) the top z-slice of im1 is contained in im2 (condition-1)
    # 2) the top z-slice of im2 is contained in im1

    s1, s2 = None, None

    # checking if condition-1 holds
    top_z_candidate_1 = im1[..., 0][candidate_2d_window]
    for z2 in range(im2.shape[-1]):  # iterate over whole im2's z axis and look for a similar candidate
        if np.allclose(top_z_candidate_1, im2[..., z2][candidate_2d_window], rtol=0, atol=1e-1):
            joint_z_window_size = min(im1.shape[-1], im2.shape[-1] - z2)
            s1_candidate = np.s_[:, :, :joint_z_window_size]
            s2_candidate = np.s_[:, :, z2: z2 + joint_z_window_size]
            if np.allclose(im1[s1_candidate], im2[s2_candidate], rtol=0, atol=1e-1):
                s1 = s1_candidate
                s2 = s2_candidate
                break

    # in case condition-1 doesn't hold
    if s1 is None:

        # checking if condition-2 holds
        top_z_candidate_2 = im2[..., 0][candidate_2d_window]
        for z1 in range(im1.shape[-1]):  # iterate over whole im1's z axis and look for a similar candidate
            if np.allclose(top_z_candidate_2, im1[..., z1][candidate_2d_window], rtol=0, atol=1e-1):
                joint_z_window_size = min(im2.shape[-1], im1.shape[-1] - z1)
                s2_candidate = np.s_[:, :, :joint_z_window_size]
                s1_candidate = np.s_[:, :, z1: z1 + joint_z_window_size]
                if np.allclose(im1[s1_candidate], im2[s2_candidate], rtol=0, atol=1e-1):
                    s1 = s1_candidate
                    s2 = s2_candidate
                    break

    return s1, s2