import requests
import pandas as pd
import csv
from phage_name_fetcher import phage_names


results = []

#print(phage_names)


for name in phage_names:
    
    url = f"https://phagesdb.org/api/genesbyphage/{name}/"

    response = requests.get(url)
    gene_Data = response.json()

    integrase_count = 0
    repressor_count = 0 
    holin_count = 0
        
    genes = gene_Data.get('results', gene_Data.get('genes'))
    #prints raw data like the cluster and start/stop of each gene—and each genes notes/name
    
    count = {"phage": name, "integrase": integrase_count, "repressor": repressor_count, "holin": holin_count}
    
    for product in genes:
            names = product.get('Notes')
            #notes = all orfs encoded by genes

            if "repressor" in names:
                count["repressor"] +=1
            
            elif "holin" in names:
                count["holin"] +=1

            elif "integrase" in names:
                count["integrase"] +=1

    results.append(count)

df = pd.DataFrame(results)
print(df)

df.to_csv("output2.csv", index=False)
                

        

    