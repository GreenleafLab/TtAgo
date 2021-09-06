#!/usr/bin/env python
""" 
Fit binding curves across a CPseries file
"""

import matplotlib as mpl
mpl.use('Agg') # Don't display plots
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy as sp
import argparse
import sys
import os
import random
from lmfit import Parameters, minimize, Model, report_fit
from joblib import Parallel, delayed
from itertools import compress
import time


### MAIN ###

def main():
    ################ Parse input parameters ################

    #set up command line argument parser
    parser = argparse.ArgumentParser(description='Script for fitting binding curves for a CPseries file')
    group = parser.add_argument_group('required arguments:')
    group.add_argument('-cs', '--cpseries', required=True,
                        help='CPseries.pkl file')
    group.add_argument('-ca', '--cpannot', required=True,
                        help='CPannot.pkl file')
    group.add_argument('-c', '--concentrations', required=True,
                        help='flat file of concentrations in matched order to fluorescence data in CPseries')
    group.add_argument('-vt', '--variant_table', required=True,
                        help='experiment variant table')

    group = parser.add_argument_group('optional arguments for processing data')
    group.add_argument('-n','--num_cores', type=int, default=10,
                        help='number of cores to use')
    group.add_argument('-mg','--multiguide_group', default=None,
                        help='Which multiguide group to restrict to.')
    group.add_argument('-pl','--plots', action="store_true",
                        help='Use flag to generate plots of fits.')
    group.add_argument('-pd','--plot_dir', default="curve_plots/",
                        help='Use flag to generate plots of fits.')


    if not len(sys.argv) > 1:
        parser.print_help()
        sys.exit()


    #parse command line arguments
    args = parser.parse_args()

    cpseries_file = args.cpseries
    cpannot_file = args.cpannot
    concentration_file = args.concentrations
    variant_file = args.variant_table
    num_cores = args.num_cores

    # Make the plot directory if plotting
    if args.plots:
        if not os.path.isdir(args.plot_dir):
            print "Plot directory {} does not exist. Creating...".format(args.plot_dir)
            os.mkdir(args.plot_dir)
        plot_dir = args.plot_dir
    else:
        plot_dir = None

    ##############################
    # Read in and preprocess data
    ##############################

    print "Reading in data..."
    binding_df = pd.read_pickle(cpseries_file)
    # drop any rows with all nans
    binding_df.dropna(axis=0, how='all', inplace=True)

    concentrations = np.recfromtxt(concentration_file)

    annot_df = pd.read_pickle(cpannot_file)
    variant_df = pd.read_table(variant_file, header=None)

    # Construct dict of labels for plots
    label_dict = {}
    if len(variant_df.columns) == 6:
        variant_df.columns = ['Sequence', 'variant_ID', 'group_name', 'mut_annotation', 'wt_seq', 'guide_number']
        for idx, row in variant_df.iterrows():
            ID = str(row.variant_ID)
            group = str(row.group_name)
            annot = str(row.mut_annotation)
            guide = "p".join(str(row.guide_number).split('.'))
            label_dict[ID] = "_".join([guide, group, annot])

    else:
        variant_df.columns = ['Sequence', 'variant_ID', 'group_name', 'mut_annotation']
        for idx, row in variant_df.iterrows():
            ID = str(row.variant_ID)
            group = str(row.group_name)
            annot = str(row.mut_annotation)
            label_dict[ID] = "_".join([group, annot])

    if args.multiguide_group:
        print "Only fitting variants in multiguide group {}".format(args.multiguide_group)
        group_ids = list(variant_df[variant_df.guide_number.apply(guide_group) == args.multiguide_group].variant_ID.values)
        group_ids.append('11111111')
        groups = annot_df.groupby('variant_ID').groups.keys()
        # Need to do this elaborate filtering procedure to keep variants that have multiple variant IDs
        valid = [contains_variant_ID(g, group_ids) for g in groups]
        valid_groups = list(compress(groups, valid))
        annot_df = annot_df[annot_df.variant_ID.isin(valid_groups)].copy()

    # Merge annot and cpseries
    merged_df = annot_df.merge(binding_df, left_index=True, right_index=True, how='inner')

    # Normalize by fiducial signal
    print "Normalizing to fiducial..."
    fiducial_meds = merged_df.groupby('variant_ID').get_group('11111111').iloc[:,1:].median().values
    merged_df.iloc[:,1:] = merged_df.iloc[:,1:] / fiducial_meds

    # Force fmin to simply be the median signal across all variants
    fmin_val = float(merged_df.iloc[:,1].median())

    ##############################################
    # MAD filter for outlier fluorescence clusters
    ##############################################

    mad_cut = 3.0
    print "Performing mad filtering with cutoff of {}...".format(mad_cut)
    initial_length = len(merged_df.index)
    mad_filt_df = MAD_filter_clusters(merged_df, mad_cutoff=mad_cut)
    filt_length = len(mad_filt_df.index)
    print "Removed {} outlier clusters ({}%)".format(initial_length - filt_length, round(100*(initial_length - filt_length)/float(initial_length), 3))

    ###########################################
    # Perform initial median fluorescence fits
    ###########################################

    # groupby variant_ID
    grouped = mad_filt_df.groupby('variant_ID')

    print "Splitting data into {} chunks and fitting...".format(num_cores)
    # Now perform the actual fits
    variant_IDs = grouped.groups.keys()

    # shuffle list and then make (num_cores) chunks
    random.shuffle(variant_IDs)
    chunk_list = list(make_chunks(variant_IDs, num_cores))

    # Create a new list of dataframes containing chunked variant_IDs
    grouped_list = [mad_filt_df[mad_filt_df.variant_ID.isin(chunk)].copy().groupby('variant_ID') for chunk in chunk_list]

    # Perform initial median fits in parallel
    if num_cores > 1:
        fit_df_list = (Parallel(n_jobs=num_cores, verbose=10)\
            (delayed(median_fits)(
                sub_grouped, concentrations, fmin_val) for sub_grouped in grouped_list))
    else:
        fit_df_list = [median_fits(
            sub_grouped, concentrations, fmin_val) for sub_grouped in grouped_list]

    med_fit_df = pd.concat(fit_df_list)
    med_fit_df.to_csv('median_variant_fits.txt', sep='\t')

    #####################################################
    # Get empirical fmax distribution of 'tight binders'
    #####################################################

    # Define 'tight binders' as those that are 97.5% saturated at last point
    tight_Kd_cutoff = concentrations[-1]/0.975 - concentrations[-1]
    print "Identifying 'tight binders' with Kd < {}".format(tight_Kd_cutoff)
    tight_binders = med_fit_df[(med_fit_df.Kd < tight_Kd_cutoff) & (med_fit_df.rsq > 0.9)].copy()
    tight_binders.to_csv('tight_binder_fits.txt', sep='\t')
    tight_fmaxes = tight_binders.fmax.values
    tight_variants = set(tight_binders.index.values)

    #####################################################
    # Bootstrap fits, enforcing fmax when necessary
    #####################################################

    nboot = 1000
    print "Bootstrapping medians to get confidence intervals on fit parameters..."
    start = time.time()
    if num_cores > 1:
        fit_df_list = (Parallel(n_jobs=num_cores, verbose=10)\
            (delayed(bootstrap_fits)(
                sub_grouped, concentrations, tight_fmaxes, tight_variants, label_dict, fmin_val=fmin_val, nboot=nboot, plot_dir=plot_dir) for sub_grouped in grouped_list))
    else:
        fit_df_list = [bootstrap_fits(
            sub_grouped, concentrations, tight_fmaxes, tight_variants, label_dict, fmin_val=fmin_val, nboot=nboot, plot_dir=plot_dir) for sub_grouped in grouped_list]

    print "Fitting finished, {} minutes.".format(round((time.time() - start)/60.0, 3))
    full_fit_df = pd.concat(fit_df_list)
    full_fit_df.to_csv('hill_equation_fits.txt', sep='\t')


# MAD filtering

def mad(arr):
    """
    Median Absolute Deviation
    """
    med = np.nanmedian(arr)
    return np.nanmedian(np.abs(arr - med))


def MAD_filter_clusters(df, mad_cutoff=3.0, n_outliers_allowed=2):
    """
    Remove clusters if they contain any outlier fluorescence values
    """
    nvals = df.iloc[:,1:].shape[1]
    grouped = df.groupby('variant_ID')
    # mad = grouped.apply(lambda x: x.mad(axis=0))
    groupIDs = grouped.groups.keys()
    filtered_groups = []
    for gID in groupIDs:
        data = grouped.get_group(gID).iloc[:,1:].values
        med_values = np.nanmedian(data, axis=0)
        mad_values = np.apply_along_axis(mad, axis=0, arr=data)
        mad_pass = (nvals - np.sum(abs(data - med_values) < mad_cutoff*mad_values, axis=1)) <= n_outliers_allowed 
        filtered_groups.append(grouped.get_group(gID).loc[mad_pass,:].copy())
    new_df = pd.concat(filtered_groups)
    return new_df


# Fitting functions:

def hill_equation(x, fmin, fmax, Kd, n=1):
    """
    Hill-langmuir equation for equilibrium binding
    x = concentrations (in nM)
    fmin = minimum signal (at 0 concentration)
    fmax = maximum concentration (at infinite concentration)
    return = signal value
    """
    return fmin + fmax*(x**n /(Kd + x**n))


def hill_equation_params(fmax=None, fmin=None, Kd=None, n=None):
    # Define parameters object
    params = Parameters()
    default_params = {
        "fmax":{"value": 1.0, "vary": True, "min": 0.0, "max": np.inf},
        "fmin":{"value": 0.0, "vary": True, "min":0.0, "max":np.inf},
        "Kd":{"value": 1.0, "vary": True, "min": 0.0, "max": np.inf},
        "n":{"value": 1.0, "vary": False, "min": -np.inf, "max": np.inf}

    }
    if fmax:
        for opt, val in fmax.items():
            default_params["fmax"][opt] = val
    if fmin:
        for opt, val in fmin.items():
            default_params["fmin"][opt] = val
    if Kd:
        for opt, val in Kd.items():
            default_params["Kd"][opt] = val
    if n:
        for opt, val in n.items():
            default_params["n"][opt] = val

    for p, dct in default_params.items():
        params.add(p, **dct)


    return params


def median_fits(grouped, x, fmin_val=0.0):
    """
    Fit the median fluorescence for each group of variants.
    Save results of fit for use in constraining lower affinity fits.
    Input:
        grouped = pandas groupby object
        x = concentrations
    """
    results_dict = {}
    group_IDs = grouped.groups.keys()
    for vID in group_IDs:
        data = grouped.get_group(vID).iloc[:,1:].values
        nclust = data.shape[0]
        median_fluorescence = np.nanmedian(data, axis=0)

        # Initialize parameters as such:
        # fmax = max median fluorescence observed (minimum is fmin)
        # Kd = max protein concentration used
        # fmin = minimum median fluorescence observed (don't let it float)
        params = hill_equation_params(
            fmax={"value":max(median_fluorescence), "min":fmin_val},
            Kd={"value":max(x)},
            fmin={"value":fmin_val, "vary":False})

        fit_model = Model(hill_equation)
        try:
            fit = fit_model.fit(median_fluorescence, params, x=x)
        except:
            #print "Error while fitting {}".format(vID)
            continue
        
        ## Things we want to report
        # quality of fit:
        ss_error = np.sum((fit.residual)**2)
        ss_total = np.sum((median_fluorescence - np.nanmean(median_fluorescence))**2)
        rsq = 1 - ss_error/ss_total
        rmse = np.sqrt(np.nanmean((fit.residual)**2))

        results_dict[vID] = {
            'Kd': fit.params['Kd'].value, 
            'fmax': fit.params['fmax'].value, 
            'fmin': fit.params['fmin'].value, 
            'rsq': rsq, 
            'rmse': rmse,
            'ier': fit.ier,
            'Kd_stderr': fit.params['Kd'].stderr,
            'fmax_stderr': fit.params['fmax'].stderr,
            'fmin_stderr': fit.params['fmin'].stderr,
            'nclust': nclust,
        }
        
    return pd.DataFrame(results_dict).T


def bootstrap_fits(grouped, x, tight_fmaxes, tight_variants, label_dict, fmin_val=0.0, nboot=1000, ci=[2.5,97.5], plot_dir=None):
    """
    Fit every group in grouped using the indicated params.
    The median fit is the reported fit. Estimate error by 
    resampling clusters and refitting medians, reporting 
    the 95CI of the bootstrapped parameters.
    """
    results_dict = {}
    group_IDs = grouped.groups.keys()
    tight_fmax_2p5, tight_fmax_med, tight_fmax_97p5 = np.nanpercentile(tight_fmaxes, q=[2.5, 50.0, 97.5])
    fit_model = Model(hill_equation)

    for vID in group_IDs:
        data = grouped.get_group(vID).iloc[:,1:].values
        nclust = data.shape[0]
        median_fluorescence = np.nanmedian(data, axis=0)

        # # If no median fluorescence value surpasses 2.5th percentile of fmax dist, 
        # # and if variant was not in previously selected tight binder group, 
        # # then enforce fmax distribution
        # fmax_force = 0.0
        # if all(median_fluorescence < tight_fmax_2p5) and (vID not in tight_variants):
        #     fmax_force = 1.0

        # 20200228:
        # New fmax forcing criteria:
        # fit medians first. If fit fmax is > tight_fmax_med*fold or < tight_fmax_med*fold, then enforce fmax distribution
        fold = 3.0
        params = hill_equation_params(
                    fmax={"value":max(median_fluorescence), "min":fmin_val},
                    Kd={"value":max(x)},
                    fmin={"value":fmin_val, "vary":False})
        med_fit = fit_model.fit(median_fluorescence, params, x=x)
        med_fmax = med_fit.params['fmax'].value
        fmax_force = 0.0
        if (med_fmax > tight_fmax_med*fold) or (med_fmax < tight_fmax_med/fold):
            fmax_force = 1.0
        # 20200228: End new fmax forcing criteria
        
        # Now bootstrap parameters
        med_array = np.empty((nboot, len(x)))
        boot_fits = {}

        for b in range(nboot):
            # Sample medians with replacement
            meds = np.nanmedian(data[np.random.choice(nclust, size=nclust, replace=True)], axis=0)
            # Enforce fmax distribution, if necessary
            if fmax_force == 1.0:
                sampled_fmax = np.random.choice(tight_fmaxes, size=1)[0]
                params = hill_equation_params(
                    fmax={"value": sampled_fmax, "vary": False}, 
                    Kd={"value":max(x)},
                    fmin={"value":fmin_val, "vary":False})
            else:
                params = hill_equation_params(
                    fmax={"value":max(meds), "min":fmin_val},
                    Kd={"value":max(x)},
                    fmin={"value":fmin_val, "vary":False})
            
            try:
                fit = fit_model.fit(meds, params, x=x)
            except:
                print "Error while bootstrap fitting {}".format(vID)
                continue

            med_array[b] = meds
            boot_fits[b] = {
                'Kd': fit.params['Kd'].value, 
                'fmax': fit.params['fmax'].value, 
                'fmin': fit.params['fmin'].value, 
                'Kd_stderr': fit.params['Kd'].stderr,
                'fmax_stderr': fit.params['fmax'].stderr,
                'fmin_stderr': fit.params['fmin'].stderr,
            }

        # Concatenate bootstrapped results
        all_fits = pd.DataFrame(boot_fits).transpose()
        param_names = all_fits.columns.tolist()
        fit_data = np.hstack([np.percentile(all_fits.loc[:, param], [50, ci[0], ci[1]])
                       for param in param_names])
        fit_idx = np.hstack([['{}{}'.format(param_name, s) for s in ['', '_lb', '_ub']]
                       for param_name in param_names])
        fit_results = pd.Series(index=fit_idx, data=fit_data)


        # Get rsq and rmse
        resid = hill_equation(x, fit_results['fmin'], fit_results['fmax'], fit_results['Kd']) - median_fluorescence
        ss_error = np.sum(resid**2)
        ss_total = np.sum((median_fluorescence - np.nanmean(median_fluorescence))**2)
        rsq = 1 - ss_error/ss_total
        rmse = np.sqrt(np.nanmean(resid**2))

        # Save final results
        results_dict[vID] = {
            'Kd': fit_results['Kd'], 
            'Kd_2p5': fit_results['Kd_lb'],
            'Kd_97p5': fit_results['Kd_ub'],
            'fmax': fit_results['fmax'], 
            'fmax_2p5': fit_results['fmax_lb'], 
            'fmax_97p5': fit_results['fmax_ub'], 
            'fmin': fit_results['fmin'], 
            'fmin_2p5': fit_results['fmin_lb'], 
            'fmin_97p5': fit_results['fmin_ub'], 
            'Kd_stderr': fit_results['Kd_stderr'],
            'fmax_stderr': fit_results['fmax_stderr'],
            'fmin_stderr': fit_results['fmin_stderr'],
            'rsq': rsq, 
            'rmse': rmse,
            'ier': fit.ier,
            'nclust': nclust,
            'forced_fmax': fmax_force
        }
        
        
        # Get median confidence intervals for plotting
        med_ci = np.nanpercentile(med_array, q=ci, axis=0)
        yerr = abs(median_fluorescence - med_ci)
        
        # Plot fit
        if plot_dir:
            fig, ax = plt.subplots()
            ax = plot_bootstrapped_Kd_fit(ax, x, median_fluorescence, yerr, 
                                             results_dict[vID]['Kd'], 
                                             [results_dict[vID]['Kd_2p5'], results_dict[vID]['Kd_97p5']], 
                                             results_dict[vID]['fmin'], 
                                             [results_dict[vID]['fmin_2p5'], results_dict[vID]['fmin_97p5']], 
                                             results_dict[vID]['fmax'], 
                                             [results_dict[vID]['fmax_2p5'], results_dict[vID]['fmax_97p5']], 
                                             nclust, tight_fmax_97p5, fmax_force)
            plt.tight_layout()
            for v in vID.split(';'):
                file_name = "/{}_{}.pdf".format(v,label_dict[v])
                plt.savefig(plot_dir+file_name, dpi=300)
            plt.close()
        
    return pd.DataFrame(results_dict).T


def plot_bootstrapped_Kd_fit(ax, x, y, y_ci, 
                                     Kd, Kd_ci,
                                     fmin, fmin_ci, 
                                     fmax, fmax_ci, nclust, max_fmax, force_fmax,
                                     showParams=True, showR2=True):
    """
    Plot the bootstrapped binding curve fit
    """
    x = np.array(x)
    y = np.array(y)
    if any([c == 0 for c in x]):
        zero_pos = np.where(x == 0)[0][0]
        plot_x = np.delete(x, zero_pos)
        plot_y = np.delete(y, zero_pos)
    ax.errorbar(x=x,y=y, yerr=y_ci, fmt='o', c='black', ecolor='black', elinewidth=0.8, ms=4)

    extend_min = min(plot_x) / 2.0
    extend_max = max(plot_x) * 2.0
    ax.set_xlim(extend_min,extend_max)
    fit_x = np.logspace(np.log10(extend_min), np.log10(extend_max), num=100)
    fit_y = hill_equation(fit_x, fmin, fmax, Kd)
    ax.plot(fit_x, fit_y, c="black", linestyle="--")
    ax.set_xscale('log')

    ymin, ymax = ax.get_ylim()
    if ymax > max_fmax:
        ax.set_ylim(ymin, max_fmax)
    
    ss_error = np.sum((hill_equation(x, fmin, fmax, Kd) - y)**2)
    ss_total = np.sum((y - np.nanmean(y))**2)
    rsq = 1 - ss_error/ss_total
    
    y_lower = hill_equation(fit_x, fmin_ci[0], fmax_ci[0], Kd_ci[0])
    y_upper = hill_equation(fit_x, fmin_ci[1], fmax_ci[1], Kd_ci[1])
    ax.fill_between(fit_x, y_lower, y_upper, alpha=0.2)
    ax.set_xlabel("Concentration (nM)")
    ax.set_ylabel("Normalized Fluorescence (a.u.)")
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(0.0, ymax)

    # Make plot square:
    ax.set_aspect('equal', 'datalim')
    
    if showParams and showR2:
        if Kd > 0.001:
            label_txt = "$K_{{D}} = {:0.3f} ({:0.3f},{:0.3f})$ $nM$\n$R^2 = {:0.3f}$\nclusters = {}\nforced fmax = {}".format(
                Kd, Kd_ci[0], Kd_ci[1], rsq, int(nclust), force_fmax)
        else:
            label_txt = "$K_{{D}} = {:0.3e} ({:0.3e},{:0.3e})$ $nM$\n$R^2 = {:0.3f}$\nclusters = {}\nforced fmax = {}".format(
                Kd, Kd_ci[0], Kd_ci[1], rsq, int(nclust), force_fmax)
        if Kd > np.median(plot_x):
            ax.text(0.05, 0.95, label_txt, transform=ax.transAxes, 
                    verticalalignment='top', horizontalalignment='left', fontsize=12, 
                    bbox={'facecolor': ax.get_facecolor(), 'alpha': 1.0, 'pad': 10, 'edgecolor':'none'})
        else:
            ax.text(0.95, 0.05, label_txt, transform=ax.transAxes, 
                    verticalalignment='bottom', horizontalalignment='right', fontsize=12, 
                    bbox={'facecolor': ax.get_facecolor(), 'alpha': 1.0, 'pad': 10, 'edgecolor':'none'})
    return ax


def make_chunks(l, n):
    # Make n chunks of list l
    ll = len(l)
    chunk_size = int(np.ceil(ll / float(n)))
    for i in range(0, ll, chunk_size):
        yield l[i:i+chunk_size]


def guide_group(guide_label):
    if isinstance(guide_label, basestring):
        group, num = guide_label.split('.')
        return group
    else:
        return "no group"


def contains_variant_ID(varID, valid_IDs):
    ids = varID.split(';')
    if any([v in valid_IDs for v in ids]):
        return True
    return False



if __name__ == '__main__':
    main()
