import requests
import pandas as pd

phage_names = []
seq_status = []





def get_all_mc2155_phages(): 
      url = "https://phagesdb.org/api/host_strains/2/phagelist/?page=7&page_size=150"
      response = requests.get(url)  

      if response.ok:
        print("Successful Request")

        data = response.json()

        phage_list_data = data.get('results')
        

        for record in phage_list_data:
            name = record.get('phage_name')
            
            is_sequenced = record.get("seq_finished")
            
        
            
            phage_names.append(name)
            seq_status.append(is_sequenced)


      else:
        print("No")
      
                              
                  
                




get_all_mc2155_phages()

df = pd.DataFrame(seq_status, columns=['Status'])
df.to_csv("status2.csv", index=False)

print(phage_names)
print(seq_status)

