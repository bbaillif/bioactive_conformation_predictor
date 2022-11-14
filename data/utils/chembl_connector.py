import sqlite3
from sqlite3 import Connection
import pandas as pd
 
class ChEMBLConnector():
    """
    Used to query the ChEMBL SQLite file
    
    :param chembl_sqlite_path: Path where the SQLite file is located
    :type chembl_sqlite_path: str
    
    """
    
    def __init__(self,
                 chembl_sqlite_path: str='/home/bb596/hdd/ChEMBL/chembl_29_sqlite/chembl_29.db') -> None:
        self.chembl_sqlite_path = chembl_sqlite_path
        
    def get_connection(self) -> Connection:
        """
        Get the sqlite3 connection object
        
        :return: Connection to ChEMBL SQLite 
        :rtype: Connection
        """
        return sqlite3.connect(self.chembl_sqlite_path)
    
    def get_target_table(self, level=2) -> pd.DataFrame:
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
        Convert a string with classification level separated with "  " into
        to a similar string up to given level
        e.g. get_target_level('level1  level2  level3  level4')
         = 'level1  level2'
        :param s: input string
        :type s: str
        :param level: maximum level to keep
        :type level: int
        """
        split_s = s.split('  ')
        return '  '.join(split_s[:level])