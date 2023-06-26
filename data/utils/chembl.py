import sqlite3
from sqlite3 import Connection
import pandas as pd
import os
from params import (CHEMBL_TARGZ_FILEPATH,
                    CHEMBL_URL,
                    CHEMBL_SQLITE_PATH,
                    CHEMBL_DIRPATH)

class ChEMBL():
    """
    Used to query the ChEMBL SQLite file
    
    :param root: Path where ChEMBL files are located
    :type root: str
    
    """
    
    def __init__(self,
                 root: str = CHEMBL_DIRPATH) -> None:
        self.root = root
        self.chembl_sqlite_path = CHEMBL_SQLITE_PATH
        if not os.path.exists(self.chembl_sqlite_path):
            self.download_chembl()
        
    
    def download_chembl(self):
        """Download the ChEMBL sqlite file
        """
        if not os.path.exists(CHEMBL_TARGZ_FILEPATH):
            os.system(f'wget {CHEMBL_URL} -O {CHEMBL_TARGZ_FILEPATH}')
        # r = requests.get(CHEMBL_URL)
        # with open(CHEMBL_TARGZ_FILEPATH, 'wb') as f:
        #     f.write(r.content)
        os.system(f'tar -xf {CHEMBL_TARGZ_FILEPATH} -C {CHEMBL_DIRPATH}')
        
        
    def get_connection(self) -> Connection:
        """
        Get the sqlite3 connection object
        
        :return: Connection to ChEMBL SQLite 
        :rtype: Connection
        """
        return sqlite3.connect(self.chembl_sqlite_path)
    
    def get_target_table(self, 
                         level: int = 2) -> pd.DataFrame:
        """
        Get the table containing information about each gene target classification 
        in ChEMBL
        
        :param level: To which level obtain the gene classification
        :type level: int
        :return: Gene target classification in DataFrame
        :rtype: pd.DataFrame
        """
        
        query = """SELECT accession, component_synonym, protein_class_desc 
        FROM component_sequences c
        JOIN component_class d ON c.component_id = d.component_id
        JOIN protein_classification e ON d.protein_class_id = e.protein_class_id
        JOIN component_synonyms f ON c.component_id = f.component_id
        WHERE f.syn_type = 'GENE_SYMBOL'"""
        
        with self.get_connection() as connection :
            df = pd.read_sql_query(query, con=connection)
        df[f'level{level}'] = df['protein_class_desc'].apply(self.get_target_level, args=[level])
        df['gene_symbol_lowercase'] = df['component_synonym'].str.lower()
        
        return df
    
    def get_target_level(self, 
                         s: str, 
                         level: int = 2) :
        """
        Convert a string with classification level separated with a double
        whitespace into a similar string up to given level
        e.g. get_target_level('level1  level2  level3  level4')
         = 'level1  level2'
        :param s: input string
        :type s: str
        :param level: maximum level to keep
        :type level: int
        """
        
        split_s = s.split('  ')
        return '  '.join(split_s[:level])