# Run Louvain and Leiden algorithms on selected datasets

## Install necessary libraries

Run `pip install -r requirements.txt`

## Email EU Core Network

Dataset [here](https://snap.stanford.edu/data/email-Eu-core.html).

Quality checks, pre-processing, louvain and leiden algorithms can be run by running: \
`python email_eu_core_network.py`

The resulting graphs can be exported to a running neo4j database by running: \
`python email_eu_core_network.py --export_graphs`\
<b>The database will be emptied at the start of the run.</b>\
In order for this to work, a neo4j database needs to be running at `bolt://localhost:7687`. The default user name `neo4j` is set and the password is for the database is set to `test1234`.

## Brain fly drosophila medulla

Dataset [here](https://networkrepository.com/bn-fly-drosophila-medulla-1.php).

Quality checks, pre-processing, louvain and leiden algorithms can be run by running: \
`python brain_network.py`

The resulting graphs can be exported to a running neo4j database by running: \
`python brain_network.py --export_graphs`\
<b>The database will be emptied at the start of the run.</b>\
In order for this to work, a neo4j database needs to be running at `bolt://localhost:7687`. The default user name `neo4j` is set and the password is for the database is set to `test1234`.