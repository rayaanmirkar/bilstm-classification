import pandas as pd
from Bio import Entrez, SeqIO
import time

Entrez.email = "ilmrexplorer@gmail.com"
Entrez.tool = "(High_School_Independent_Researcher)_PhageBiLSTM_Project"

df = pd.read_csv('cleaned_processed_benchmark_set.csv')
protein_sentences = []


for acc in df['RefSeq accession number']:
   
    try:
        handle = Entrez.efetch(db="nucleotide", id = str(acc), rettype = 'gb', retmode = "text" )
        record = SeqIO.read(handle, "genbank")
        handle.close()


        proteins = " ".join([str(f.qualifiers['translation'][0]) for f in record.features if 'translation' in f.qualifiers])
        protein_sentences.append(proteins)

        print(f"got : {acc}")
        

    except Exception as e:
        protein_sentences.append("")
        print(f"Skipped {acc} due to error {e}...")

    time.sleep(1)

df['protein_sentence'] = protein_sentences 
df.to_csv('final_data_training.csv', index= False)