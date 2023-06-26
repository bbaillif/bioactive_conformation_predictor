import torch

from torch_geometric.nn import radius_graph
from torch_geometric.nn.models import SchNet
from torch_scatter import scatter
from .atomistic_nn import AtomisticNN

class AtomicSchNet(AtomisticNN, SchNet) :
    """
    Modification of the PyG SchNet implementation to recover the atomic
    contributions to the prediction
    
    :param readout: Readout function to perform on the list of individual
        atomic values
    :type readout: str
    :param num_interactions: Number of interaction blocks (see SchNet publication)
    :type num_interactions: int
    :param cutoff: Cutoff for neighbourhood graph 
    :type cutoff: float
    """
    
    def __init__(self, 
                 readout: str = 'add',
                 num_interactions: int = 6,
                 cutoff: float = 10):
        AtomisticNN.__init__(self,
                             readout=readout)
        SchNet.__init__(self,
                        num_interactions=num_interactions, 
                        cutoff=cutoff)
        
    
    def forward(self, 
                z: torch.Tensor, 
                pos: torch.Tensor, 
                batch=None):
        """
        Compute values for each atom in the input. Truncated version of
        forward function from SchNet PyG implementation
        
        :param z: Sequence of atomic numbers 
        :type z: torch.Tensor (n_atoms)
        :param pos: Sequence of atomic cartesian positions
        :type pos: torch.Tensor (n_atoms, 3)
        :param batch: Sequence of batch identifier: which molecule the atom 
            corresponds to e.g. [0,0,0,1,1,1,1] means that the 3 first atoms
            belong to molecule 0, then the 4 next atoms belongs to molecule 1.
            If batch is None (default value), all atoms will be considered coming
            from the same molecule
        :type batch: torch.Tensor (n_atoms)
        """
        assert z.dim() == 1 and z.dtype == torch.long
        
        batch = torch.zeros_like(z) if batch is None else batch

        h = self.embedding(z)

        edge_index = radius_graph(pos, r=self.cutoff, batch=batch,
                                  max_num_neighbors=self.max_num_neighbors)
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        edge_attr = self.distance_expansion(edge_weight)

        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)

        if self.dipole:
            # Get center of mass.
            mass = self.atomic_mass[z].view(-1, 1)
            c = scatter(mass * pos, batch, dim=0) / scatter(mass, batch, dim=0)
            h = h * (pos - c.index_select(0, batch))

        if not self.dipole and self.mean is not None and self.std is not None:
            h = h * self.std + self.mean

        if not self.dipole and self.atomref is not None:
            h = h + self.atomref(z)

        return h