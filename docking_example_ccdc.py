# This script can be used for any purpose without limitation subject to the
# conditions at http://www.ccdc.cam.ac.uk/Community/Pages/Licences/v2.aspx
#
# This permission notice and the following statement of attribution must be
# included in all copies or substantial portions of this script.
#
# 2017-02-10: created by the Cambridge Crystallographic Data Centre
#
#

'''
This simple example script uses docking to evaluate different scoring functions

Some of the analysis code depends on the numpy package and the pandas package
being installed.
Presentation of the results depends on the package matplotlib.

.. seealso:: <http://www.numpy.org/> for more information on numpy.
.. seealso:: <http://pandas.pydata.org/> for more information on pandas.
.. seealso:: <https://matplotlib.org/> for more information on matplotlib.
'''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
import os
import multiprocessing
from multiprocessing import Pool
import re
import numpy as np
import pandas as pd
from ccdc.docking import Docker
from ccdc.io import MoleculeReader, EntryReader
from ccdc.protein import Protein


def run_gold(params):
    (protein, ligand, reference, ndocks,
     fitness_function, rescore_function, no_run, base_out_dir, max_results) = params

    protein_file = os.path.abspath(protein)
    ligand_file = os.path.abspath(ligand)
    native_ligand_file = os.path.abspath(reference)

    docker = Docker()
    settings = docker.settings
    protein = Protein.from_file(protein_file)

    ligand = MoleculeReader(ligand_file)[0]

    native_ligand = MoleculeReader(native_ligand_file)[0]
    settings.add_protein_file(protein_file)
    settings.add_ligand_file(ligand_file, ndocks)
    settings.reference_ligand_file = reference
    settings.binding_site = settings.BindingSiteFromLigand(protein, native_ligand, 6.0)

    settings.diverse_solutions = (True, 3, 1.0)
    # The max_results parameter can be passed in to speed up runs for testing purposes.
    # However this WILL reduce the usefulness of the results as the set of structures to compare
    # is reduced to the given number.
    if max_results is not None:
        settings.early_termination = (True, max_results, 1.0)
    else:
        settings.early_termination = False

    settings.fitness_function = fitness_function
    rescore_label = ''
    if rescore_function != 'None':
        settings.rescore_function = rescore_function
        rescore_label = "score_" + str(rescore_function)
    else:
        rescore_function = fitness_function

    output_dir = os.path.abspath(get_docking_directory(base_out_dir,
                                                       fitness_function, rescore_function, no_run))
    settings.output_directory = output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    settings.output_file = os.path.join(
        output_dir, get_docking_file(fitness_function, rescore_function, no_run))
    docker.dock(file_name=os.path.join(output_dir,
                                       get_conf_file(fitness_function, rescore_function, no_run)))


def rescore_gold(params):
    (reference, fitness_function, rescore_function, no_run, base_out_dir) = params
    in_path = get_docking_directory(base_out_dir, fitness_function, fitness_function, no_run)

    protein_file = os.path.join(in_path, 'gold_protein.mol2')
    ligand_file = os.path.join(in_path,
                               get_docking_file(fitness_function, fitness_function, no_run))

    # protein settings for docking
    docker = Docker()
    settings = docker.settings
    protein = Protein.from_file(protein_file)
    settings.add_protein_file(os.path.join(protein_file))
    settings.add_ligand_file(ligand_file, 1)

    # define the binding site using the reference ligand
    native_ligand_file = os.path.abspath(reference)
    native_ligand = MoleculeReader(native_ligand_file)[0]
    settings.binding_site = settings.BindingSiteFromLigand(protein, native_ligand, 6.)


    settings.reference_ligand_file = native_ligand_file


    settings.fitness_function = None
    settings.rescore_function = rescore_function

    output_dir = get_docking_directory(base_out_dir, fitness_function, rescore_function, no_run)
    settings.output_directory = output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    settings.output_file = os.path.join(output_dir, get_docking_file(
        fitness_function, rescore_function, no_run))
    docker.dock(file_name=os.path.join(get_docking_directory(base_out_dir, fitness_function,
                                                             rescore_function, no_run),
                                       get_conf_file(fitness_function, rescore_function, no_run)))


def get_conf_file(fitness_function, scoring_function, no_run):
    return '%s_score_%s_run_%s.conf' % (fitness_function, scoring_function, no_run)


def get_docking_directory(base_dir, fitness_function, scoring_function, no_run):
    return os.path.abspath(os.path.join(base_dir,
                                        'fitness_%s' % fitness_function,
                                        'score_%s' % scoring_function,
                                        'run_%s' % no_run))


def get_docking_file(fitness_function, scoring_function, no_run):
    return 'poses_%s_score_%s_run_%s.mol2' % (fitness_function, scoring_function, no_run)


def get_path(base_dir, fitness_function, scoring_function, no_run):
    return os.path.join(
        get_docking_directory(base_dir, fitness_function, scoring_function, no_run),
        get_docking_file(fitness_function, scoring_function, no_run))


def get_scores(ndocks, nruns, output_directory):
    SF = ('goldscore', 'plp', 'chemscore', 'asp')
    re_RMSD = re.compile(r'\w*\.Reference\.RMSD')
    re_FITNESS = re.compile(r'.+\..+\.Fitness')
    df = {}
    pose_metrics = {}
    for run in range(1, int(ndocks) + 1):
        for dockfunc in SF:
            # dockmols = {}
            dock_entries = {}
            poselist = {}
            for sf in SF:
                this_path = get_path(output_directory, dockfunc, sf, run)
                # dockmols[sf] = MoleculeReader(this_path)
                dock_entries[sf] = EntryReader(this_path)
            for i in range(0, int(nruns)):
                table = {}
                for sf in SF:
                    table['Docking_function'] = sf
                    d = dock_entries[sf][i]
                    # m = dockmols[sf][i]
                    for key, value in d.attributes.items():
                        test = re_RMSD.match(key)
                        if test:
                            table[key] = value
                        test_fitness = re_FITNESS.match(key)
                        if test_fitness:
                            table[key] = value
                series = pd.Series(table)
                poselist[i] = series
            df = pd.DataFrame(poselist)
            pose_metrics[dockfunc, run] = df.T

    # Next, let's work out if any correct poses were found (sampling success)
    # and if so, if the scoring or rescoring functions were able to detect them
    # as the top solution
    # We create a dictionary for each docking/rescoring/run combination,
    # add them to a list and convert them to a dataframe
    RMSD_threshold = 2.0
    data = []
    for dockfunc in ['asp', 'chemscore', 'goldscore', 'plp']:
        for run in range(1, int(ndocks) + 1):
            df = pose_metrics[dockfunc, run]
            for scorefunc in ['Gold.ASP.Fitness', 'Gold.Chemscore.Fitness',
                              'Gold.Goldscore.Fitness', 'Gold.PLP.Fitness', 'Gold.Reference.RMSD']:
                dockings = {}
                dockings['dockfunc'] = dockfunc
                dockings['run'] = run
                dockings['scorefunc'] = scorefunc
                dockings['worked'] = 0
                flag = False
                if scorefunc == 'Gold.Reference.RMSD':
                    flag = True
                if (float(df.sort_values(by=scorefunc,
                                         ascending=flag).iloc[0]['Gold.Reference.RMSD']) <
                        float(RMSD_threshold)):
                    dockings['worked'] = 1
                data.append(dockings)
    success_list = pd.DataFrame(data)

    # Now we can average the success obtained by a combination of docking
    # and scoring function using a pivot table
    mean_success = pd.pivot_table(success_list, index=['dockfunc', 'scorefunc'], values=[
                                  'worked'], aggfunc=np.mean)
    mean_success.to_csv(os.path.join(output_directory, 'mean_sucess.csv'))

    # Last thing to do is to illustrate our data in a visual diagram
    plot_data = mean_success.unstack()
    dockfunc = plot_data.index.get_level_values(0)
    scorefunc = [str(a).split('.')[1].lower() for a in plot_data.columns.get_level_values(1)]
    scorefunc = [s.replace('reference', 'rmsd') for s in scorefunc]
    plt.yticks(np.arange(0.5, len(dockfunc), 1), dockfunc)
    plt.xticks(np.arange(0.5, len(scorefunc), 1), scorefunc)
    plt.pcolor(plot_data, cmap=matplotlib.cm.Blues)
    plt.title('GOLD docking performance')
    plt.xlabel('scoring function')
    plt.ylabel('docking function')
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, 'statistics.png'))
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run dockings to evaluate different fitness functions.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--protein', required=True, dest='protein',
                        help='The prepared protein .mol2 file')
    parser.add_argument('-l', '--ligand', required=True, dest='ligand',
                        help='The prepared ligand .mol2 file')
    parser.add_argument('-r', '--reference', required=True, dest='reference',
                        help='The reference ligand to define the active site')
    parser.add_argument('-n', '--ndocks', type=int, default=10, dest='ndocks',
                        help='The number of dockings to attempt per ligand')
    parser.add_argument('-t', '--ntests', type=int, default=5, dest='ntests',
                        help='The number of parallel test runs to attempt')
    parser.add_argument('-d', '--output_directory', default=os.path.abspath('output'),
                        help='The directory to store output files in.')
    parser.add_argument('-c', '--num_cores', type=int, default=multiprocessing.cpu_count(),
                        help='How many docking processes to run in parallel.')
    parser.add_argument('-m', '--max_results', type=int, default=None,
                        help='Number of results after which to end docking for testing purposes.')
    return parser.parse_args()


if __name__ == '__main__':
    import ccdc.utilities
    ccdc.utilities._fix_multiprocessing_on_macos()
    args = parse_args()

    SF = ('goldscore', 'plp', 'chemscore', 'asp')

    with Pool(args.num_cores) as p:
        # generate parameters for docking runs and distribute them among CPUs
        params = [(args.protein, args.ligand, args.reference, int(args.ndocks),
                scoring_function, 'None', run, os.path.abspath(args.output_directory),
                args.max_results)
                for scoring_function in SF
                for run in range(1, int(args.ntests) + 1)]
        p.map(run_gold, params)

        # generate parameters for rescoring runs and distribute them among CPUs
        rescore_params = [(args.reference, scoring_function, rescoring_function,
                        run, os.path.abspath(args.output_directory))
                        for scoring_function in SF
                        for rescoring_function in SF
                        for run in range(1, int(args.ntests) + 1)
                        if scoring_function != rescoring_function]
        p.map(rescore_gold, rescore_params)

    # Actually run the dockings
    get_scores(args.ntests, args.ndocks, os.path.abspath(args.output_directory))