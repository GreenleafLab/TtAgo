#!/usr/bin/env python
""" 
Perform single singel exponential decay fits across a CPseries file

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
    parser = argparse.ArgumentParser(description='Script for fitting single exponential decay for a CPseries file')
    group = parser.add_argument_group('required arguments:')
    group.add_argument('-cs', '--cpseries', required=True,
                        help='CPseries.pkl file')
    group.add_argument('-ca', '--cpannot', required=True,
                        help='CPannot.pkl file')
    group.add_argument('-td', '--time_dict', required=True,
                        help='pickled dictionary of image times')
    group.add_argument('-vt', '--variant_table', required=True,
                        help='experiment variant table')

    group = parser.add_argument_group('optional arguments for processing data')
    group.add_argument('-nt', '--norm_table',
                        help='normalization variant table')
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
    diss_time_file = args.time_dict
    variant_file = args.variant_table
    norm_file = args.norm_table
    num_cores = args.num_cores

    # Make the plot directory if plotting
    if args.plots:
        if not os.path.isdir(args.plot_dir):
            print "Plot directory {} does not exist. Creating...".format(args.plot_dir)
            os.mkdir(args.plot_dir)
        plot_dir = args.plot_dir
    else:
        plot_dir = None

    # Read in data
    print "Reading in data..."
    diss_df = pd.read_pickle(cpseries_file)
    # drop any rows with all nans
    diss_df.dropna(axis=0, how='all', inplace=True)

    diss_times = pd.read_pickle(diss_time_file)
    # Just use the 9th image times
    diss_times = diss_times['009']

    # If series length is one more than diss_times, assume first point is the baseline
    series_points = len(diss_df.columns)
    if series_points > len(diss_times):
        if series_points == len(diss_times) + 1:
            print "Series length and times don't match. Setting first point in series to 0.0 seconds..."
            diss_times = [0.0] + diss_times
        else:
            print "Series length and times differ by more than 1. Are these the right files?"
            sys.exit()

    annot_df = pd.read_pickle(cpannot_file)
    variant_df = pd.read_table(variant_file, header=None)
    
    # Construct dict of labels for plots
    label_dict = {}
    if len(variant_df.columns) == 6:
        variant_df.columns = ['Sequence', 'variant_ID', 'group_name', 'mut_annotation', 'wt_seq', 'guide_number']
        #norm_df.columns = ['Sequence', 'variant_ID', 'group_name', 'mut_annotation', 'wt_seq', 'guide_number']
        for idx, row in variant_df.iterrows():
            ID = str(row.variant_ID)
            group = str(row.group_name)
            annot = str(row.mut_annotation)
            guide = "p".join(str(row.guide_number).split('.'))
            label_dict[ID] = "_".join([guide, group, annot])

    else:
        variant_df.columns = ['Sequence', 'variant_ID', 'group_name', 'mut_annotation']
        #norm_df.columns = ['Sequence', 'variant_ID', 'group_name', 'mut_annotation']
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
    merged_df = annot_df.merge(diss_df, left_index=True, right_index=True, how='inner')

    # Normalize everything by pre-defined normalization variants if provided
    if norm_file:
        print "Normalizing to variants in norm table..."
        norm_df = pd.read_table(norm_file, header=None)
        if len(variant_df.columns) == 6:
            norm_df.columns = ['Sequence', 'variant_ID', 'group_name', 'mut_annotation', 'wt_seq', 'guide_number']
        else:
            norm_df.columns = ['Sequence', 'variant_ID', 'group_name', 'mut_annotation']
        norm_var_IDs = [str(x) for x in norm_df.variant_ID.values]
        norm_series = merged_df[merged_df.variant_ID.isin(norm_var_IDs)]
        norm_meds = norm_series.iloc[:,1:].median().values

        merged_df.iloc[:,1:] = merged_df.iloc[:,1:] / norm_meds

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
    # Perform bootstrapped single exp decay fits
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

    # Now fit variants in parallel
    nboot = 1000
    start = time.time()
    if num_cores > 1:
        fit_df_list = (Parallel(n_jobs=num_cores, verbose=10)\
            (delayed(bootstrap_fits)(
                sub_grouped, diss_times, label_dict, nboot=nboot, plot_dir=plot_dir) for sub_grouped in grouped_list))
    else:
        fit_df_list = [bootstrap_fits(
            sub_grouped, diss_times, label_dict, nboot=nboot, plot_dir=plot_dir) for sub_grouped in grouped_list]

    print "Fitting finished, {} minutes.".format(round((time.time() - start)/60.0, 3))
    full_fit_df = pd.concat(fit_df_list)
    full_fit_df.to_csv('single_exponential_decay_fits.txt', sep='\t')


# MAD filtering

def MAD_filter_clusters(df, mad_cutoff=3.0):
    """
    Remove clusters if they contain any outlier fluorescence values
    """
    grouped = df.groupby('variant_ID')
    mad = grouped.apply(lambda x: x.mad(axis=0))
    cutoff_df = grouped.apply(lambda x: abs(x.iloc[:,1:] - mad.loc[x.name,:]) < mad_cutoff*mad.loc[x.name,:])
    all_pass = cutoff_df.apply(lambda x: all(x.values), axis=1).values
    return df.loc[all_pass,:].copy()


# Fitting functions:

def single_exp_decay(x, fmin, fmax, kc):
    """
    Single exponential decay function. 
    Functions provided to lmfit Model must have dependent variable as first argument.
    x = time points 
    fmin = decay floor
    fmax = max signal
    kc = decay rate
    """
    x = np.array(x)
    return fmin + (fmax - fmin)*np.exp(-kc*x)


def single_exp_decay_params(fmax=None, span=None, kc=None):
    # Define parameters object
    params = Parameters()
    default_params = {
        "fmax":{"value": 1.0, "vary": True, "min": -np.inf, "max": np.inf},
        "span":{"value": 0.1, "vary": True, "min":0.01, "max":np.inf},
        "kc":{"value": 0.0001, "vary": True, "min": -np.inf, "max": np.inf}

    }
    if fmax:
        for opt, val in fmax.items():
            default_params["fmax"][opt] = val
    if kc:
        for opt, val in kc.items():
            default_params["kc"][opt] = val
    if span:
        for opt, val in span.items():
            default_params["span"][opt] = val

    for p, dct in default_params.items():
        params.add(p, **dct)

    # Enforce that fmax > fmin and that fmin <= 0.3*fmax
    params.add("fmin", value=0.1, expr='max([0.3*fmax - span, 0.01])')
    return params


def bootstrap_fits(grouped, x, label_dict, nboot=1000, ci=[2.5,97.5], plot_dir=None):
    """
    Fit every group in grouped using the indicated params.
    The median fit is the actually reported fit. Estimate error by 
    resampling clusters and refitting the medians, reporting 
    the 95CI of the bootstrapped parameters.
    """
    results_dict = {}
    group_IDs = grouped.groups.keys()
    fit_model = Model(single_exp_decay)

    for vID in group_IDs:
        data = grouped.get_group(vID).iloc[:,1:].values
        nclust = data.shape[0]
        median_fluorescence = np.nanmedian(data, axis=0)

        # If fluorescence does not decay by >= 20%, then call it a 'non cleaver'
        initial = np.median(median_fluorescence[:3])
        final = np.median(median_fluorescence[-3:])
        min_kc = min_kc = -np.log(0.8)/max(x)
        if (initial - final)/initial < 0.3:
            # Save non-fit results
            results_dict[vID] = {
                'kc': min_kc,
                'kc_2p5': min_kc,
                'kc_97p5': min_kc,
                'fmax': initial,
                'fmax_2p5': initial,
                'fmax_97p5': initial,
                'fmin': final, 
                'fmin_2p5': final,
                'fmin_97p5': final,
                'kc_stderr': 0.0,
                'fmax_stderr': 0.0,
                'fmin_stderr': 0.0,
                'rsq': 1.0, 
                'rmse': 0.0,
                'ier': 1.0,
                'nclust': nclust,
            }
        else:
            # Now bootstrap parameters
            med_array = np.empty((nboot, len(x)))
            boot_fits = {}

            for b in range(nboot):
                # Sample medians with replacement
                meds = np.nanmedian(data[np.random.choice(nclust, size=nclust, replace=True)], axis=0)
                # Force fmax to be +/- 10% of the maximum fluorescence value
                maxf = max(meds)
                params = single_exp_decay_params(fmax={"value": maxf, "min": maxf - 0.1*maxf, "max": maxf + 0.1*maxf})

                try:
                    fit = fit_model.fit(meds, params, x=x)
                except:
                    print "Error while bootstrap fitting {}".format(vID)
                    continue
                med_array[b] = meds
                boot_fits[b] = {
                    'kc': fit.params['kc'].value,
                    'fmax': fit.params['fmax'].value,
                    'fmin': fit.params['fmin'].value,
                    'kc_stderr': fit.params['kc'].stderr,
                    'fmax_stderr': fit.params['fmax'].stderr,
                    'fmin_stderr': fit.params['fmin'].stderr
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
            resid = single_exp_decay(x, fit_results['fmin'], fit_results['fmax'], fit_results['kc']) - median_fluorescence
            ss_error = np.sum(resid**2)
            ss_total = np.sum((median_fluorescence - np.nanmean(median_fluorescence))**2)
            rsq = 1 - ss_error/ss_total
            rmse = np.sqrt(np.nanmean(resid**2))
            
            # Save final results
            results_dict[vID] = {
                'kc': fit_results['kc'],
                'kc_2p5': fit_results['kc_lb'],
                'kc_97p5': fit_results['kc_ub'],
                'fmax': fit_results['fmax'],
                'fmax_2p5': fit_results['fmax_lb'],
                'fmax_97p5': fit_results['fmax_ub'],
                'fmin': fit_results['fmin'], 
                'fmin_2p5': fit_results['fmin_lb'], 
                'fmin_97p5': fit_results['fmin_ub'], 
                'kc_stderr': fit_results['kc_stderr'],
                'fmax_stderr': fit_results['fmax_stderr'],
                'fmin_stderr': fit_results['fmin_stderr'],
                'rsq': rsq, 
                'rmse': rmse,
                'ier': fit.ier,
                'nclust': nclust,
            }
            
            # Get median confidence intervals for plotting
            med_ci = np.nanpercentile(med_array, q=ci, axis=0)
            yerr = abs(median_fluorescence - med_ci)
            
            # Plot fit
            if plot_dir:
                fig, ax = plt.subplots()
                ax = plot_bootstrapped_single_exp_fit(ax, x, median_fluorescence, yerr, 
                                                 results_dict[vID]['kc'], 
                                                 [results_dict[vID]['kc_2p5'], results_dict[vID]['kc_97p5']], 
                                                 results_dict[vID]['fmin'], 
                                                 [results_dict[vID]['fmin_2p5'], results_dict[vID]['fmin_97p5']], 
                                                 results_dict[vID]['fmax'], 
                                                 [results_dict[vID]['fmax_2p5'], results_dict[vID]['fmax_97p5']], nclust)
                for v in vID.split(';'):
                    file_name = "/{}_{}.pdf".format(v,label_dict[v])
                    plt.savefig(plot_dir+file_name, dpi=300)
                plt.close()
        
    return pd.DataFrame(results_dict).T


def plot_bootstrapped_single_exp_fit(ax, x, y, y_ci, 
                                     kc, kc_ci,
                                     fmin, fmin_ci, 
                                     fmax, fmax_ci, nclust,
                                     showParams=True, showR2=True):
    """
    Plot the bootstrapped single exponential decay
    """
    x = np.array(x)
    y = np.array(y)
    ax.errorbar(x=x,y=y, yerr=y_ci, fmt='o', c='black', ecolor='black', elinewidth=0.8, ms=4)
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(-10.0,xmax)
    fit_x = np.linspace(0.0, xmax, 1000)
    fit_y = single_exp_decay(fit_x, fmin, fmax, kc)
    ax.plot(fit_x, fit_y, c="black", linestyle="--")
    
    rsq = 1 - np.var(single_exp_decay(x, fmin, fmax, kc) - y) / np.var(y)
    
    y_lower = single_exp_decay(fit_x, fmin_ci[0], fmax_ci[0], kc_ci[0])
    y_upper = single_exp_decay(fit_x, fmin_ci[1], fmax_ci[1], kc_ci[1])
    ax.fill_between(fit_x, y_lower, y_upper, alpha=0.2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized Fluorescence (a.u.)")
    ymin, ymax = ax.get_ylim()
    # Don't let ymax get out of control
    ymax = max(0.25, min(max(y)*1.5, ymax))
    ax.set_ylim(0.0, ymax)
    half_life = np.log(2.0)/kc
    
    if showParams and showR2:
        label_txt = "$k_{{c}} = {:0.3e}$ $s^{{-1}}$\n$t_{{1/2}} = {:0.3f}$ $min$\n$R^2 = {:0.3f}$\nnclust = {}".format(
                kc, half_life/60.0, rsq, nclust)
        ax.text(0.95, 0.95, label_txt, transform=ax.transAxes, 
                verticalalignment='top', horizontalalignment='right', fontsize=12, 
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