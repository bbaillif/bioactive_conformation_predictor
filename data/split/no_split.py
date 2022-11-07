from .data_split import DataSplit

class NoSplit(DataSplit) :
    
    def __init__(self) -> None:
        self.split_type = 'no_split'
        self.split_i = 0
        super().__init__(split_type=self.split_type, 
                         split_i=self.split_i)
        
    def get_smiles(self,
                   subset_name=None) :
        return self.cel_df['smiles'].unique()