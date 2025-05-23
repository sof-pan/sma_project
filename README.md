# Run Louvain and Leiden algorithms on selected datasets V2.00  

## Prerequisites 

### Install necessary libraries Using Conda

[Install Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install)

Run `conda env create -f environment.yml
conda activate envSMA
`
### install Neo4j 
[possible via docker](https://docs.docker.com/get-started/)

Run `docker run -d --name neo4j-test \
  -p7474:7474 -p7687:7687 \
  -e NEO4J_AUTH=neo4j/test1234 \
  neo4j:latest
`

## Email EU Core Network

Dataset [here](https://snap.stanford.edu/data/email-Eu-core.html).

### Use for leiden louvain from scratch

Quality checks, pre-processing, louvain and leiden algorithms can be run by running: \
`python email_eu_core_network.py`

The resulting graphs can be exported to a running neo4j database by running: \
`python email_eu_core_network.py --export_graphs`\
<b>The database will be emptied at the start of the run.</b>\
In order for this to work, a neo4j database needs to be running at `bolt://localhost:7687`. The default user name `neo4j` is set and the password is for the database is set to `test1234`.

## Brain fly drosophila medulla [use for leiden louvain from scratch]

Dataset [here](https://networkrepository.com/bn-fly-drosophila-medulla-1.php).

### Use for leiden louvain from scratch

Quality checks, pre-processing, louvain and leiden algorithms can be run by running: \
`python brain_network.py`

The resulting graphs can be exported to a running neo4j database by running: \
`python brain_network.py --export_graphs`\
<b>The database will be emptied at the start of the run.</b>\
In order for this to work, a neo4j database needs to be running at `bolt://localhost:7687`. The default user name `neo4j` is set and the password is for the database is set to `test1234`.


## Youtube social network 

### Operating diagram  
![image](https://github.com/user-attachments/assets/a1911ba2-dba9-4502-86bc-c5b861263217)

Dataset [here](https://snap.stanford.edu/data/com-Youtube.html) 

### Use for leiden louvain from scratch

Quality checks, pre-processing, louvain and leiden algorithms can be run by running: \
`python com-youtube.ungraph_network.py`

<b>The script create two JSON files</b>  
- youtube_network_results_louvain_scratch.json
- youtube_network_results_leiden_scratch.json

This file can be use by analysis_leiden_louvain.py 

### Use for leiden louvain Lib
Quality checks, pre-processing, louvain and leiden algorithms can be run by running: \
`python com-youtube.ungraph_network_lib.py`

<b>The script create two JSON files</b> 
- youtube_network_results_louvain_scratch.json
- youtube_network_results_leiden_scratch.json

This file can be use by analysis_leiden_louvain.py


## analysis_leiden_louvain_xx.py 
<b> create graphs to be mapped from the 4 json files created for each dataset. </b>
for exemple for Youtube social network 

### Youtube social network  
analysis_leiden_louvain_youtube_net.py 

youtube social network the 4 JSON file:
- youtube_network_results_louvain_scratch.json
- youtube_network_results_leiden_scratch.json
- youtube_network_results_louvain_lib.json
- youtube_network_results_leiden_lib.json

