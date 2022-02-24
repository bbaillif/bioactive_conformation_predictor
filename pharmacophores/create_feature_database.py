#!/usr/bin/env python
#
# This script can be used for any purpose without limitation subject to the
# conditions at http://www.ccdc.cam.ac.uk/Community/Pages/Licences/v2.aspx
#
# This permission notice and the following statement of attribution must be
# included in all copies or substantial portions of this script.
#
# 2017-04-05: created by the Cambridge Crystallographic Data Centre
#
"""
create_feature_database.py - Example script to create a Pharmacophore Feature Database.
"""

import argparse
import os
import csv
import tempfile
import shutil

from ccdc.io import EntryReader
from ccdc.pharmacophore import Pharmacophore
from ccdc.utilities import Colour

####################################################################################################################

# If you are creating a Structure database from the CSD set DatabaseInfo.file_name to a list of CSD
# database file names.  If you want to include any available CSD updates include them in the list.
# Alternatively, set DatabaseInfo.file_name to the string 'csd' rather than a list and all CSD updates
# will be automatically included.
csd_path = EntryReader('CSD').file_name
if not isinstance(csd_path, (list, tuple)):
    csd_path = [csd_path]
if len(csd_path) > 1:
    csd_path = [csd_path[0]] # Do not use any CSD updates if available

def parse_args():
    '''Define and parse the arguments to the script.'''
    parser = argparse.ArgumentParser(
        description='Create a new CSD-CrossMiner Pharmacophore Feature Database.'
    )

    parser.add_argument(
        '-i',
        '--input_structures',
        help='Input protein structural files (a directory of mol2 files [])',
        default=None
    )
    parser.add_argument(
        '--input_dna_structures',
        help='Input nucleic acid structural files (a directory of mol2 files [])',
        default=None
    )

    parser.add_argument(
        '-a',
        '--annotations',
        help='Path to the annotations CSV file for the set of input structures',
        default=None
    )
    parser.add_argument(
        '--dna_annotations',
        help='Path to the annotations CSV file for the set of nucleic acid input structures',
        default=None
    )
    parser.add_argument(
        '-o',
        '--output_feature_database',
        help='Output filename for the generated pharmacophore feature database [out.feat]',
        default='out.feat'
    )
    parser.add_argument(
        '-f',
        '--feature_definitions',
        help='Directory containing the set of feature definition files',
        default=None
    )
    parser.add_argument(
        '-m',
        '--protein_database',
        help='Filename of intermediate protein structure database [out.csdsqlx]',
        default='out.csdsqlx'
    )
    parser.add_argument(
        '--dna_database',
        help='Filename of intermediate nucleic acid structure database [out_dna.csdsqlx]',
        default='out_dna.csdsqlx'
    )
    parser.add_argument(
        '-c',
        '--use_csd',
        help='Whether to include all entries from the current CSD [False]',
        choices=['True', 'False'],
        default='False'
    )
    parser.add_argument(
        '-C',
        '--maximum_csd_structures',
        help='Number of entries to include from the current CSD',
        type=int,
        default=0
    )
    parser.add_argument(
        '-A',
        '--csd_annotations',
        help='Path to the CSD annotations CSV file for CSD structures',
        default=None
    )
    parser.add_argument(
        '-t',
        '--number_of_threads',
        help='Number of threads to use for feature database creation',
        type=int,
        default=None
    )

    return parser.parse_args()


def load_annotations(feature_database_path, annotation_csv, do_write=True):
    """This uses a CSV file to annotate the feature database.

    The csv file should list the annotation names in the first row, and the identifiers
    should be given in the first column. *e.g.*,

        id, first, second
        AABHTZ, one, two
        AACANI10, three, four
        AACANI11, five, once I caught a fish alive,
        AACFAZ, seven, eight
    """
    feature_database = Pharmacophore.FeatureDatabase.from_file(feature_database_path)
    with open(annotation_csv, 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            identifier = row['identifier'].strip()
            del row['identifier']
            d = {
                k: v.strip() for k, v in row.items()
            }
            feature_database.annotate(identifier, **d)

    if do_write:
        tempdir = tempfile.mkdtemp()
        print("Writing feature database size: %d to file: %s" %
              (len(feature_database), feature_database_path))
        feature_database.write(os.path.join(tempdir, os.path.basename(feature_database_path)))
        shutil.copyfile(os.path.join(tempdir, os.path.basename(feature_database_path)),
                        feature_database_path)


def create_databases(
    input_file_path, feature_definitions_path, protein_database_path, feature_db_path,
    include_csd=False, maximum_csd_structures=None, number_of_threads=None,
    input_dna_file_path=None, dna_database_path=None
):
    settings = Pharmacophore.FeatureDatabase.Creator.Settings(
        feature_definition_directory=feature_definitions_path, n_threads=number_of_threads
    )
    Pharmacophore.read_feature_definitions(feature_definitions_path)
    feature_defs = list(Pharmacophore.feature_definitions.values())
    creator = Pharmacophore.FeatureDatabase.Creator(settings)

    databases = []

    # If you already have a structure database, you can set database_path to point to it
    # and omit the input_file_path argument

    def add_mol2_input_database(mol2_path, database_path, colour, databases):
        if os.path.exists(database_path):
            print('Using existing structure database %s' % database_path)
            if mol2_path is not None:
                print('--- WARNING: loose mol2 files in %s will be ignored in favour of this!' %
                      mol2_path)
        elif mol2_path is not None:
            print('Using loose mol2 files in %s' % mol2_path)

        if mol2_path is not None or os.path.exists(database_path):
            db_info = Pharmacophore.FeatureDatabase.DatabaseInfo(
                mol2_path,
                0,
                colour
            )
            structure_database = creator.StructureDatabase(
                db_info,
                use_crystal_symmetry=False,
                structure_database_path=database_path,
            )
            databases.append(structure_database)

    add_mol2_input_database(input_file_path, protein_database_path,
                            Colour(255, 170, 0, 255), databases)
    add_mol2_input_database(input_dna_file_path, dna_database_path,
                            Colour(170, 255, 0, 255), databases)

    #  CSD feature database
    if include_csd:
        db_info = Pharmacophore.FeatureDatabase.DatabaseInfo(
            csd_path, maximum_csd_structures, (255, 255, 0, 255))
        csd_database = creator.StructureDatabase(
            db_info,
            use_crystal_symmetry=True,
            structure_filters=creator.StructureDatabase.default_csd_filters()
        )
        databases.append(csd_database)

    print('Writing feature database to: ' + feature_db_path)
    feature_database = creator.create(databases, feature_defs)
    feature_database.write(feature_db_path)

    return creator


def main():
    args = parse_args()
    if args.input_structures is None and args.protein_database is None \
        and args.input_dna_structures is None and args.dna_database is None \
        and args.use_csd == 'False':
        raise RuntimeError(
            'At least one of input_structures, protein_database, input_dna_structures, dna_database and use_csd must be specified')

    creator = create_databases(
        args.input_structures,
        args.feature_definitions,
        args.protein_database,
        args.output_feature_database,
        include_csd=args.use_csd == 'True',
        maximum_csd_structures=args.maximum_csd_structures,
        number_of_threads=args.number_of_threads,
        input_dna_file_path=args.input_dna_structures,
        dna_database_path=args.dna_database
    )

    if args.annotations:
        load_annotations(args.output_feature_database, args.annotations)

    if args.dna_annotations:
        load_annotations(args.output_feature_database, args.dna_annotations)

    if args.csd_annotations:
        load_annotations(args.output_feature_database, args.csd_annotations)

    print('CSD-CrossMiner Feature Database generation is complete!')


if __name__ == "__main__":
    main()