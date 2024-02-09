import os
import pandas as pd
import requests

from tqdm import tqdm
from bioconfpred.params import (ENZYME_URL,
                    ENZYME_DAT_FILEPATH)

class ENZYME():
    """Parse data on enzyme classifications obtained at
    https://ftp.expasy.org/databases/enzyme/
    
    :param root: Path of the ENZYME directory
    :type root: str
    """
    
    def __init__(self,
                 enzyme_filepath: str = ENZYME_DAT_FILEPATH) -> None:
        self.enzyme_filepath = enzyme_filepath
        if not os.path.exists(self.enzyme_filepath):
            self.download_enzyme()
        
    
    def download_enzyme(self) -> None:
        """Download the enzyme.dat file
        """
        r = requests.get(ENZYME_URL)
        with open(ENZYME_DAT_FILEPATH, 'wb') as f:
            f.write(r.content)
        
        
    def get_table(self) -> pd.DataFrame:
        """Get all ENZYME data in a single table

        :return: Dataframe with ENZYME data
        :rtype: pd.DataFrame
        """
        
        table = self.parse_data()
        for i in range(4):
            table[f'level_{i+1}_combined'] = table['enzyme_id'].apply(lambda s: '.'.join(s.split('.')[:i+1]))
            table[f'level_{i+1}_only'] = table['enzyme_id'].apply(lambda s: s.split('.')[i])
        return table
    
    def parse_data(self) -> pd.DataFrame:
        """Parse the enzyme.dat file

        :return: Raw table of enzyme data
        :rtype: pd.DataFrame
        """
        
        with open(self.enzyme_filepath, 'r') as f:
            lines = f.readlines()
            
        enzyme_id = None
        
        rows = []
        print('Loading enzyme classes data')
        for line in tqdm(lines):
            prefix = line[:2]
            data = line[5:]
            if prefix == 'ID':
                enzyme_id = data.strip()
                current_uniprot_ids = []
                current_enzyme_names = []
            elif prefix == 'DE':
                class_name = data.strip()
            elif prefix == 'DR':
                tuples = data.split(';')
                for t in tuples:
                    t = t.strip()
                    if len(t) > 0:
                        s = t.split(',')
                        try:
                            uniprot_id = s[0].strip()
                            enzyme_name = s[1].strip()
                            current_uniprot_ids.append(uniprot_id)
                            current_enzyme_names.append(enzyme_name)
                        except:
                            import pdb;pdb.set_trace()
            elif prefix == '//':
                if enzyme_id is not None: # "//" is present before the first class
                    
                    for i, uniprot_id in enumerate(current_uniprot_ids):
                        enzyme_name = current_enzyme_names[i]
                        
                        row = {}
                        row['enzyme_id'] = enzyme_id
                        row['class_name'] = class_name
                        row['uniprot_id'] = uniprot_id
                        row['enzyme_name'] = enzyme_name
                        
                        rows.append(row)
                        
        df = pd.DataFrame(rows)
                    
        return df