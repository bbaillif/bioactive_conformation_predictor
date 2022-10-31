import sqlite3
import pandas as pd
 
class ChEMBLConnector():
    
    def __init__(self,
                 chembl_sqlite_path: str='/home/bb596/hdd/ChEMBL/chembl_29_sqlite/chembl_29.db') -> None:
        self.chembl_sqlite_path = chembl_sqlite_path
        
    def get_connector(self) :
        return sqlite3.connect(self.chembl_sqlite_path)
    
    def get_target_table(self, level=2) :
        query = """SELECT accession, component_synonym, protein_class_desc 
FROM component_sequences c
JOIN component_class d ON c.component_id = d.component_id
JOIN protein_classification e ON d.protein_class_id = e.protein_class_id
JOIN component_synonyms f ON c.component_id = f.component_id
WHERE f.syn_type = 'GENE_SYMBOL'"""
        with self.get_connector() as connector :
            df = pd.read_sql_query(query, con=connector)
        df[f'level{level}'] = df['protein_class_desc'].apply(self.get_target_level, args=[level])
        df['gene_symbol_lowercase'] = df['component_synonym'].str.lower()
        return df
    
    def get_target_level(self, s, level=2) :
        split_s = s.split('  ')
        return '  '.join(split_s[:level])