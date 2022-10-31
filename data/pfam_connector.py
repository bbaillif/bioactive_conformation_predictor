import os
import json
import ssl
import sys

from urllib import request
from urllib.error import HTTPError
from time import sleep

class PFamConnector():
    
    def __init__(self) -> None:
        self.base_url = 'https://www.ebi.ac.uk:443/interpro/api/entry/pfam/structure/PDB/'
        
    def from_pdb(self,
                 pdb_id: str):
        """Retrieve PFAM information from PDB id
        Inspired from https://www.ebi.ac.uk/interpro/result/download/#/entry/pfam/structure/PDB/2vtm/|json
        """
        
        pfam_ids = []
        
        context = ssl._create_unverified_context()

        next = os.path.join(self.base_url, pdb_id)
        
        attempts = 0
        while next:
            try:
                req = request.Request(next, headers={"Accept": "application/json"})
                res = request.urlopen(req, context=context)
                # If the API times out due a long running query
                if res.status == 408:
                    # wait just over a minute
                    sleep(61)
                    # then continue this loop with the same URL
                    continue
                elif res.status == 204:
                    #no data so leave loop
                    break
                payload = json.loads(res.read().decode())
                next = payload["next"]
                attempts = 0
            except HTTPError as e:
                if e.code == 408:
                    sleep(61)
                    continue
                else:
                    # If there is a different HTTP error, it wil re-try 3 times before failing
                    if attempts < 3:
                        attempts += 1
                        sleep(61)
                        continue
                    else:
                        raise e

            for item in payload["results"]:
                metadata = item['metadata']
                pfam_id = metadata['accession']
                pfam_ids.append(pfam_id)
            
            # Don't overload the server, give it time before asking for more
            if next:
                sleep(1)
            
        return pfam_ids