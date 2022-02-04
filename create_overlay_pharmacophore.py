#!/usr/bin/env python
#
# This script can be used for any purpose without limitation subject to the
# conditions at http://www.ccdc.cam.ac.uk/Community/Pages/Licences/v2.aspx
#
# This permission notice and the following statement of attribution must be
# included in all copies or substantial portions of this script.
#
# 2018-09-25: created by the Cambridge Crystallographic Data Centre
#
'''
    create_overlay_pharmacophore.py -   create a pharmacophore from an overlay of molecules.

    A molecule file of the overlaid molecules is required. Supported file format: mol2, pdb, sdf.

    Example command:
        python create_overlay_pharmacophore.py -o output_directory overlay_file.mol2
'''
##############################################################################################################

import os
import argparse
import collections

from ccdc import io, utilities
from ccdc.descriptors import GeometricDescriptors

from ccdc.pharmacophore import Pharmacophore

##############################################################################################################

class Runner(argparse.ArgumentParser):
    def __init__(self):
        super(self.__class__, self).__init__(description=__doc__,
                                             formatter_class=argparse.RawTextHelpFormatter)
        self.add_argument(
            'overlay_file',
            help='Molecule file of overlaid molecules'
        )
        self.add_argument(
            '-o', '--output-directory', default='.',
            help='Where output will be stored'
        )
        self.add_argument(
            '-f', '--feature_definitions', '--features', nargs='*',
            help='Feature definitions to be used.  Default is the standard set of feature definitions.'
        )
        self.add_argument(
            '-t', '--threshold', type=float, default=0.0,
            help='Threshold at which features will be defined'
        )
        self.args = self.parse_args()

    def run(self):
        if not os.path.exists(self.args.output_directory):
            os.makedirs(self.args.output_directory)

        Pharmacophore.read_feature_definitions()
        self.crystals = list(io.CrystalReader(self.args.overlay_file))
        if self.args.threshold <= 0.0:
            self.args.threshold = (len(self.crystals))/2.0
        if self.args.feature_definitions:
            self.feature_definitions = [v for k, v in Pharmacophore.feature_definitions.items() if k in self.args.feature_definitions]
        else:
            self.feature_definitions = [
                fd for fd in Pharmacophore.feature_definitions.values()
                if fd.identifier != 'exit_vector' and
                fd.identifier != 'heavy_atom' and
                fd.identifier != 'hydrophobe'
            ]

        complete_set_of_features = []
        for fd in self.feature_definitions:
            detected = [fd.detect_features(c) for c in self.crystals]
            all_feats = [f for l in detected for f in l]
            if not all_feats:
                    continue
            minx = min(f.spheres[0].centre.x() for f in all_feats)
            miny = min(f.spheres[0].centre.y() for f in all_feats)
            minz = min(f.spheres[0].centre.z() for f in all_feats)
            maxx = max(f.spheres[0].centre.x() for f in all_feats)
            maxy = max(f.spheres[0].centre.y() for f in all_feats)
            maxz = max(f.spheres[0].centre.z() for f in all_feats)
            g = utilities.Grid((minx-1., miny-1., minz-1.), (maxx+1, maxy+1, maxz+1), 0.2)

            spheres = []
            for f in all_feats:
                if f.spheres[0] in spheres:
                    g.set_sphere(f.spheres[0].centre, f.spheres[0].radius, 0)
                else:
                    spheres.append(f.spheres[0])
                    g.set_sphere(f.spheres[0].centre, f.spheres[0].radius, 1)

            islands = g.islands(self.args.threshold)
            print('Feature: %s, max value %.2f, n_features %d' %
                (fd.identifier, g.extrema[1], len(islands))
            )
            for island in islands:
                # how do I make a feature from an island?  Location of highest value
                indices = island.indices_at_value(island.extrema[1])
                centre = indices[0]
                org = island.bounding_box[0]
                centre = tuple(org[i] + island.spacing*centre[i] for i in range(3))
                radius = 1.0
                # Any other spheres?
                if len(all_feats[0].spheres) > 1:
                    # Pick all features which contain centre
                    feat_dists ={}
                    for f in all_feats:
                        dist, feat = (GeometricDescriptors.point_distance(f.spheres[0].centre, centre), f)
                        if dist in feat_dists:
                            feat_dists[dist].append(feat)
                        else:
                            feat_dists[dist] = [feat]

                        feat_dists = collections.OrderedDict(sorted(feat_dists.items()))
                        shortest_distance = list(feat_dists.keys())[0]

                    if len(feat_dists[shortest_distance]) > 1:
                        new_feat = [
                            Pharmacophore.Feature(fd, GeometricDescriptors.Sphere(centre, radius),
                                                  feat_dists[shortest_distance][i].spheres[1])
                            for i in range(len(feat_dists[shortest_distance]))]
                    else:
                        new_feat = [
                            Pharmacophore.Feature(fd, GeometricDescriptors.Sphere(centre, radius),
                                                  feat_dists[shortest_distance][0].spheres[1])]
                else:
                    new_feat = [
                        Pharmacophore.Feature(fd, GeometricDescriptors.Sphere(centre, radius))]

                complete_set_of_features.extend(new_feat)
            model = Pharmacophore.Query(complete_set_of_features)

            model.write(os.path.join(self.args.output_directory, 'model.cm'))

##############################################################################################################

if __name__ == '__main__':
    r = Runner()
    r.run()