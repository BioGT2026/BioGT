# 固定边关联关系
edge_types = {
    "UD": "Undirected",      # 代号 "UD" 对应 "Undirected"
    "SD": "SingleDirected",  # 代号 "SD" 对应 "SingleDirected"
    "DD": "DoubleDirected",  # 代号 "DD" 对应 "DoubleDirected"
    "EQ": "Equal"            # 代号 "EQ" 对应 "Equal"
}
fieldnames_edge = ["Node 1 ID", "Node 1 Data Type", "Node 1 Source", "Edge Type", "Edge Direction", "Edge Weight", "Node 2 ID", "Node 2 Data Type", "Node 2 Source"]
fieldnames_node = ["ID", "Type", "Name", "Sequence", "Comments"]
NCBI_GENE_PREFIX = "NCBIGene:"
UNIPROT_PROTEIN_PREFIX = "Uniprot:"
CLINVAR_VARIANT_PREFIX = "ClinVar:"
OMIM_PREFIX = "OMIM:"
PUBCHEM_PREFIX = "PubChem:"
REACTOME_PREFIX = "Reactome:" # New prefix for Reactome entries
# Ranges of edge weights for different edge types: -1 to 1. -1 means negative correlation, 0 means no correlation, 1 means positive correlation.

def safe_join(items, sep='|'):
    if not items:  # Handles None, [], "", etc.
        return ""
    try:
        return sep.join(map(str, items))  # Works for lists, tuples, sets
    except TypeError:  # Fallback for non-iterables (e.g., single string/number)
        return str(items)
    
def evidence_to_weight(evidence):
    """
    Convert evidence type to weight.
    
    Parameters:
    ----------
    evidence : str
        Evidence code (e.g. "IEA", "EXP", "IDA", etc.)

    Returns:
    -------
    float
        Weight value between 0.0 and 1.0 based on evidence reliability
    """
    edge_weight = 0.0
    if evidence == "PCS" or evidence == "IEA" or evidence == "IGC" or evidence == "ISA" or evidence == "ISM" or evidence == "ISO" or evidence == "ISS" or evidence == "NAS":
        edge_weight = 0.1
    elif evidence == "HDA" or evidence == "HEP" or evidence == "HGI" or evidence == "HMP" or evidence == "HTP" or evidence == "IBA" or evidence == "IC" or evidence == "IKR" or evidence == "RCA" or evidence == "TAS":
        edge_weight = 0.5
    elif evidence == "EXP" or evidence == "IDA" or evidence == "IMP" or evidence == "IGI" or evidence == "IEP" or evidence == "IPI":
        edge_weight = 1.0
    elif evidence == "ND":
        edge_weight = 0.0
    else:
        print(f"Unknown evidence type: {evidence}")
        # write to a log file
        with open("unknown_evidence.log", "a") as log_file:
            log_file.write(f"Unknown evidence type: {evidence}\n")
            log_file.close()
    
    return edge_weight
