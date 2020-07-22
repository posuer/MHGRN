import torch
import networkx as nx
import itertools
import json
from tqdm import tqdm
from .conceptnet import merged_relations
import numpy as np
import random
from scipy import sparse
import pickle
from scipy.sparse import csr_matrix, coo_matrix
from multiprocessing import Pool
from .maths import *
import os
from utils.utils import check_path

__all__ = ['generate_graph']

concept2id = None
id2concept = None
relation2id = None
id2relation = None

cpnet = None
cpnet_all = None
cpnet_simple = None


def load_resources(cpnet_vocab_path):
    global concept2id, id2concept, relation2id, id2relation

    with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
        id2concept = [w.strip() for w in fin]
    concept2id = {w: i for i, w in enumerate(id2concept)}

    id2relation = merged_relations
    relation2id = {r: i for i, r in enumerate(id2relation)}


def load_cpnet(cpnet_graph_path = None, cpnet_simple_graph_path = None):
    global cpnet, cpnet_simple
    cpnet = nx.read_gpickle(cpnet_graph_path)
    cpnet_simple = nx.Graph()
    for u, v, data in cpnet.edges(data=True): #remove relations from graph
        w = data['weight'] if 'weight' in data else 1.0
        if cpnet_simple.has_edge(u, v):
            cpnet_simple[u][v]['weight'] += w
        else:
            cpnet_simple.add_edge(u, v, weight=w)
    if cpnet_simple_graph_path:
        pickle.dump(cpnet_simple, open(cpnet_simple_graph_path,'wb'))



def relational_graph_generation(qcs, acs, paths, rels):
    raise NotImplementedError()  # TODO


# plain graph generation
def plain_graph_generation(qcs, acs, paths, rels):
    global cpnet, concept2id, relation2id, id2relation, id2concept, cpnet_simple

    graph = nx.Graph()
    
    for p in paths:
        for c_index in range(len(p) - 1):
            h = p[c_index]
            t = p[c_index + 1]
            # TODO: the weight can computed by concept embeddings and relation embeddings of TransE
            graph.add_edge(h, t, weight=1.0)

    for qc1, qc2 in list(itertools.combinations(qcs, 2)): # gengyu: all head linked together? then 
        if cpnet_simple.has_edge(qc1, qc2):
            graph.add_edge(qc1, qc2, weight=1.0)

    for ac1, ac2 in list(itertools.combinations(acs, 2)):
        if cpnet_simple.has_edge(ac1, ac2):
            graph.add_edge(ac1, ac2, weight=1.0)

    if len(qcs) == 0:
        qcs.append(-1)

    if len(acs) == 0:
        acs.append(-1)

    if len(paths) == 0:
        for qc in qcs:
            for ac in acs:
                graph.add_edge(qc, ac, rel=-1, weight=0.1)

    g = nx.convert_node_labels_to_integers(graph, label_attribute='cid')  # re-index
    return nx.node_link_data(g)


def generate_adj_matrix_per_inst(nxg_str):
    global id2relation
    n_rel = len(id2relation)

    nxg = nx.node_link_graph(json.loads(nxg_str))
    n_node = len(nxg.nodes)
    cids = np.zeros(n_node, dtype=np.int32)
    for node_id, node_attr in nxg.nodes(data=True):
        cids[node_id] = node_attr['cid']

    adj = np.zeros((n_rel, n_node, n_node), dtype=np.uint8)
    for s in range(n_node):
        for t in range(n_node):
            s_c, t_c = cids[s], cids[t]
            if cpnet_all.has_edge(s_c, t_c):
                for e_attr in cpnet_all[s_c][t_c].values():
                    if e_attr['rel'] >= 0 and e_attr['rel'] < n_rel:
                        adj[e_attr['rel']][s][t] = 1
    cids += 1
    adj = coo_matrix(adj.reshape(-1, n_node))
    return (adj, cids)


def concepts2adj(node_ids):
    global id2relation
    cids = np.array(node_ids, dtype=np.int32) #list to numpy
    n_rel = len(id2relation) # number of relations
    n_node = cids.shape[0] #number of nodes
    adj = np.zeros((n_rel, n_node, n_node), dtype=np.uint8) #adj with 3-d matrix: #rel, #node, #node
    for s in range(n_node):
        for t in range(n_node): #for every two node pair
            s_c, t_c = cids[s], cids[t] #get node id
            if cpnet.has_edge(s_c, t_c): 
                for e_attr in cpnet[s_c][t_c].values(): 
                    if e_attr['rel'] >= 0 and e_attr['rel'] < n_rel:
                        adj[e_attr['rel']][s][t] = 1
    # cids += 1  # note!!! index 0 is reserved for padding
    adj = coo_matrix(adj.reshape(-1, n_node))

    return adj, cids


def concepts_to_adj_matrices_1hop_neighbours(data):
    qc_ids, ac_ids = data
    qa_nodes = set(qc_ids) | set(ac_ids)
    extra_nodes = set()
    for u in set(qc_ids) | set(ac_ids):
        if u in cpnet.nodes:
            extra_nodes |= set(cpnet[u])
    extra_nodes = extra_nodes - qa_nodes
    schema_graph = sorted(qc_ids) + sorted(ac_ids) + sorted(extra_nodes)
    arange = np.arange(len(schema_graph))
    qmask = arange < len(qc_ids)
    amask = (arange >= len(qc_ids)) & (arange < (len(qc_ids) + len(ac_ids)))
    adj, concepts = concepts2adj(schema_graph)
    return adj, concepts, qmask, amask


def concepts_to_adj_matrices_1hop_neighbours_without_relatedto(data):
    qc_ids, ac_ids = data
    qa_nodes = set(qc_ids) | set(ac_ids)
    extra_nodes = set()
    for u in set(qc_ids) | set(ac_ids):
        if u in cpnet.nodes:
            for v in cpnet[u]:
                for data in cpnet[u][v].values():
                    if data['rel'] not in (15, 32):
                        extra_nodes.add(v)
    extra_nodes = extra_nodes - qa_nodes
    schema_graph = sorted(qc_ids) + sorted(ac_ids) + sorted(extra_nodes)
    arange = np.arange(len(schema_graph))
    qmask = arange < len(qc_ids)
    amask = (arange >= len(qc_ids)) & (arange < (len(qc_ids) + len(ac_ids)))
    adj, concepts = concepts2adj(schema_graph)
    return adj, concepts, qmask, amask


def concepts_to_adj_matrices_2hop_qa_pair(data):
    qc_ids, ac_ids = data
    qa_nodes = set(qc_ids) | set(ac_ids)
    extra_nodes = set()
    for qid in qc_ids:
        for aid in ac_ids:
            if qid != aid and qid in cpnet_simple.nodes and aid in cpnet_simple.nodes:
                extra_nodes |= set(cpnet_simple[qid]) & set(cpnet_simple[aid])
    extra_nodes = extra_nodes - qa_nodes
    schema_graph = sorted(qc_ids) + sorted(ac_ids) + sorted(extra_nodes)
    arange = np.arange(len(schema_graph))
    qmask = arange < len(qc_ids)
    amask = (arange >= len(qc_ids)) & (arange < (len(qc_ids) + len(ac_ids)))
    adj, concepts = concepts2adj(schema_graph)
    return adj, concepts, qmask, amask


def concepts_to_adj_matrices_2hop_all_pair(data):

    qc_ids, ac_ids = data

    qa_nodes = set(qc_ids) | set(ac_ids)
    extra_nodes = []
    for qid in qa_nodes:
        for aid in qa_nodes:
            if qid != aid and qid in cpnet_simple.nodes and aid in cpnet_simple.nodes:
                extra_nodes.extend(list(set(cpnet_simple[qid]) & set(cpnet_simple[aid]))) # find all 2 hop paths, extra_nodes are middile nodes 
    extra_nodes = set(extra_nodes)
    extra_nodes = extra_nodes - qa_nodes
    schema_graph = sorted(qc_ids) + sorted(ac_ids) + sorted(extra_nodes)
    arange = np.arange(len(schema_graph))
    qmask = arange < len(qc_ids)
    amask = (arange >= len(qc_ids)) & (arange < (len(qc_ids) + len(ac_ids)))
    adj, concepts = concepts2adj(schema_graph)
    return adj, concepts, qmask, amask

#Gengyu
def concepts_to_adj_matrices_pruned_path_ckb(data):

    qc_ids, ac_ids, extra_nodes = data

    qa_nodes = set(qc_ids) | set(ac_ids)
    extra_nodes = extra_nodes - qa_nodes
    schema_graph = sorted(qc_ids) + sorted(ac_ids) + sorted(extra_nodes)
    arange = np.arange(len(schema_graph))
    qmask = arange < len(qc_ids)
    amask = (arange >= len(qc_ids)) & (arange < (len(qc_ids) + len(ac_ids)))
    adj, concepts = concepts2adj(schema_graph)
    return adj, concepts, qmask, amask


def concepts_to_adj_matrices_2step_relax_all_pair(data):
    qc_ids, ac_ids = data
    qa_nodes = set(qc_ids) | set(ac_ids)
    extra_nodes = set()
    for qid in qc_ids:
        for aid in ac_ids:
            if qid != aid and qid in cpnet_simple.nodes and aid in cpnet_simple.nodes:
                extra_nodes |= set(cpnet_simple[qid]) & set(cpnet_simple[aid])
    intermediate_ids = extra_nodes - qa_nodes
    for qid in intermediate_ids:
        for aid in ac_ids:
            if qid != aid and qid in cpnet_simple.nodes and aid in cpnet_simple.nodes:
                extra_nodes |= set(cpnet_simple[qid]) & set(cpnet_simple[aid])
    for qid in qc_ids:
        for aid in intermediate_ids:
            if qid != aid and qid in cpnet_simple.nodes and aid in cpnet_simple.nodes:
                extra_nodes |= set(cpnet_simple[qid]) & set(cpnet_simple[aid])
    extra_nodes = extra_nodes - qa_nodes
    schema_graph = sorted(qc_ids) + sorted(ac_ids) + sorted(extra_nodes)
    arange = np.arange(len(schema_graph))
    qmask = arange < len(qc_ids)
    amask = (arange >= len(qc_ids)) & (arange < (len(qc_ids) + len(ac_ids)))
    adj, concepts = concepts2adj(schema_graph)
    return adj, concepts, qmask, amask


def concepts_to_adj_matrices_3hop_qa_pair(data):
    qc_ids, ac_ids = data
    qa_nodes = set(qc_ids) | set(ac_ids)
    extra_nodes = set()
    for qid in qc_ids:
        for aid in ac_ids:
            if qid != aid and qid in cpnet_simple.nodes and aid in cpnet_simple.nodes:
                for u in cpnet_simple[qid]:
                    for v in cpnet_simple[aid]:
                        if cpnet_simple.has_edge(u, v):  # ac is a 3-hop neighbour of qc
                            extra_nodes.add(u)
                            extra_nodes.add(v)
                        if u == v:  # ac is a 2-hop neighbour of qc
                            extra_nodes.add(u)
    extra_nodes = extra_nodes - qa_nodes
    schema_graph = sorted(qc_ids) + sorted(ac_ids) + sorted(extra_nodes)
    arange = np.arange(len(schema_graph))
    qmask = arange < len(qc_ids)
    amask = (arange >= len(qc_ids)) & (arange < (len(qc_ids) + len(ac_ids)))
    adj, concepts = concepts2adj(schema_graph)
    return adj, concepts, qmask, amask


#####################################################################################################
#                     functions below this line will be called by preprocess.py                     #
#####################################################################################################

#Gengyu
def generate_graph_ckb(keywords_path ,pruned_paths_path, kb, cpnet_graph_path, output_path):
    print(f'generating schema graphs for and {pruned_paths_path}...')
    if not os.path.isdir('/'.join(output_path.split('/')[:-1])):
        os.mkdir('/'.join(output_path.split('/')[:-1]))
    global concept2id, id2concept, relation2id, id2relation
    if any(x is None for x in [concept2id, id2concept, relation2id, id2relation]):
        #load_resources(cpnet_vocab_path)
        concept2id = kb.kb_vocab
        id2concept = kb.invert_kb_vocab
        relation2id = kb.relation2id
        id2relation = kb.id2relation

    global cpnet, cpnet_simple
    if cpnet is None or cpnet_simple is None:
        load_cpnet(cpnet_graph_path)

    keywords_dict = pickle.load(open(keywords_path, 'rb'))


    with open(pruned_paths_path, 'r') as fin_pf, \
            open(output_path, 'w') as fout:
        fin_pf_lines = fin_pf.readlines()
        idx = 0
        for line_pf in tqdm(fin_pf_lines, total=len(fin_pf_lines)):
            #mcp = json.loads(line_gr)
            qa_pairs = json.loads(line_pf)

            #format keywords_dict[idx] = {"cq_word":set(), "answerA":set(), "answerB":set(), "answerC":set() }
            mcp ={"qc":keywords_dict[int(idx / 3)]['cq_word'], "ac": list(keywords_dict[int(idx / 3)].values())[idx % 3 + 1]}
            idx += 1

            statement_paths = []
            statement_rel_list = []
            for qas in qa_pairs:
                if qas["pf_res"]:
                    cur_paths = [item["path"] for item in qas["pf_res"]]
                    cur_rels = [item["rel"] for item in qas["pf_res"]]
                else:
                    cur_paths = []
                    cur_rels = []
                    continue
                statement_paths.extend(cur_paths)
                statement_rel_list.extend(cur_rels)

                #mcp["qc"].add(('conceptnet',qas['qc']))
                #mcp["ac"].add(('conceptnet',qas['ac']))
                '''
                try:
                    concept2id[('conceptnet',qas['qc'])]
                except:
                    #print()
                    print(qas['qc'], qas['ac'])
                    #print(kb.invert_kb_vocab[cur_paths[0][0]])
                    for path in cur_paths:
                        print(kb.invert_kb_vocab[cur_paths[-1]])
                '''
            qcs = [concept2id[('conceptnet', kb.lemmatize_word(c))] for c in mcp["qc"] if kb.lemmatize_word(c)]
            acs = [concept2id[('conceptnet', kb.lemmatize_word(c))] for c in mcp["ac"] if kb.lemmatize_word(c)]

            gobj = plain_graph_generation(qcs=qcs, acs=acs,
                                          paths=statement_paths,
                                          rels=statement_rel_list)
            fout.write(json.dumps(gobj) + '\n')

    print(f'schema graphs saved to {output_path}')
    print()



def generate_graph(grounded_path, pruned_paths_path, cpnet_vocab_path, cpnet_graph_path, output_path):
    print(f'generating schema graphs for {grounded_path} and {pruned_paths_path}...')

    global concept2id, id2concept, relation2id, id2relation
    if any(x is None for x in [concept2id, id2concept, relation2id, id2relation]):
        load_resources(cpnet_vocab_path)

    global cpnet, cpnet_simple
    if cpnet is None or cpnet_simple is None:
        load_cpnet(cpnet_graph_path)

    nrow = sum(1 for _ in open(grounded_path, 'r'))
    with open(grounded_path, 'r') as fin_gr, \
            open(pruned_paths_path, 'r') as fin_pf, \
            open(output_path, 'w') as fout:
        for line_gr, line_pf in tqdm(zip(fin_gr, fin_pf), total=nrow):
            mcp = json.loads(line_gr)
            qa_pairs = json.loads(line_pf)

            statement_paths = []
            statement_rel_list = []
            for qas in qa_pairs:
                if qas["pf_res"] is None:
                    cur_paths = []
                    cur_rels = []
                else:
                    cur_paths = [item["path"] for item in qas["pf_res"]]
                    cur_rels = [item["rel"] for item in qas["pf_res"]]
                statement_paths.extend(cur_paths)
                statement_rel_list.extend(cur_rels)

            qcs = [concept2id[c] for c in mcp["qc"]]
            acs = [concept2id[c] for c in mcp["ac"]]

            gobj = plain_graph_generation(qcs=qcs, acs=acs,
                                          paths=statement_paths,
                                          rels=statement_rel_list)
            fout.write(json.dumps(gobj) + '\n')

    print(f'schema graphs saved to {output_path}')
    print()


def generate_adj_matrices(ori_schema_graph_path, cpnet_graph_path, cpnet_vocab_path, output_path, num_processes, num_rels=34, debug=False):
    print(f'generating adjacency matrices for {ori_schema_graph_path} and {cpnet_graph_path}...')

    global concept2id, id2concept, relation2id, id2relation
    if any(x is None for x in [concept2id, id2concept, relation2id, id2relation]):
        load_resources(cpnet_vocab_path)

    global cpnet_all
    if cpnet_all is None:
        cpnet_all = nx.read_gpickle(cpnet_graph_path)

    with open(ori_schema_graph_path, 'r') as fin:
        nxg_strs = [line for line in fin]

    if debug:
        nxgs = nxgs[:1]

    with Pool(num_processes) as p:
        res = list(tqdm(p.imap(generate_adj_matrix_per_inst, nxg_strs), total=len(nxg_strs)))

    with open(output_path, 'wb') as fout:
        pickle.dump(res, fout)

    print(f'adjacency matrices saved to {output_path}')
    print()

#Gengyu
def generate_adj_data_from_pruned_paths_ckb(cpnet_graph_path, pruned_paths_path, kb, keywords_path, output_path, num_processes, n_path=None, max_node_num=None, seed=0):
    """
    This function will save
        (1) adjacency matrics (each in the form of a (R*N, N) coo sparse matrix)
        (2) concepts ids
        (3) qmask that specifices whether a node is a question concept
        (4) amask that specifices whether a node is a answer concept
    to the output path in python pickle format

    grounded_path: str
    cpnet_graph_path: str
    cpnet_vocab_path: str
    output_path: str
    num_processes: int
    n_path: int
    """
    print(f'generating adj data for {pruned_paths_path} and keep top {n_path} for each keyword pair...')

    global concept2id, id2concept, relation2id, id2relation, cpnet_simple, cpnet
    if any(x is None for x in [concept2id, id2concept, relation2id, id2relation]):
        #load_resources(cpnet_vocab_path)
        concept2id = dict(kb.kb_vocab)
        id2concept = dict(kb.invert_kb_vocab)
        relation2id = dict(kb.relation2id)
        id2relation = dict(kb.id2relation)
        
    if cpnet is None or cpnet_simple is None:
        load_cpnet(cpnet_graph_path )

    keywords_dict = pickle.load(open(keywords_path, 'rb'))
    #format keywords_dict[idx] = {"cq_word":set(), "answerA":set(), "answerB":set(), "answerC":set() }
    qa_data = []
    for value in keywords_dict:
        for key, words in value.items():
            if key.startswith("answer"):
                mcp ={"qc":value['cq_word'], "ac": words}
                q_ids = set(concept2id[c] for c in mcp["qc"])
                a_ids = set(concept2id[c] for c in mcp["ac"])
                q_ids = q_ids - a_ids
                qa_data.append((q_ids, a_ids))
    
    pruned_path_file = open(pruned_paths_path,'r')
    pruned_path_lines = pruned_path_file.readlines()
    assert len(pruned_path_lines) == len(qa_data)

    qa_path_data = []
    for idx, line in tqdm(enumerate(pruned_path_lines), total=len(pruned_path_lines), desc='extract node id from pruned paths'): 
        line_json = json.loads(line) 
        this_qa_path_nodes = []
        for keyword_pair in line_json:
            for path in keyword_pair['pf_res'][:n_path]: #only use top n path
                this_qa_path_nodes.extend(list(path['path'][1:])) #union
        this_qa_path_nodes = set(this_qa_path_nodes)
        if max_node_num and len(this_qa_path_nodes)>max_node_num: #randomly choose maximum n node
            random.seed(seed)
            this_qa_path_nodes = list(this_qa_path_nodes)
            random.shuffle(this_qa_path_nodes)
            this_qa_path_nodes = set(this_qa_path_nodes[:max_node_num])
        qa_path_data.append((qa_data[idx][0], qa_data[idx][1], this_qa_path_nodes))

    with Pool(num_processes) as p:
        res = list(tqdm(p.imap(concepts_to_adj_matrices_pruned_path_ckb, qa_path_data), total=len(qa_data)))

    # res is a list of tuples, each tuple consists of four elements (adj, concepts, qmask, amask)
    if n_path: output_path=output_path+'.npath'+str(n_path)
    if max_node_num: output_path=output_path+'.mxnode'+str(max_node_num)
    with open(output_path, 'wb') as fout:
        pickle.dump(res, fout)

    print(f'adj data saved to {output_path}')
    print()



#Gengyu
def generate_adj_data_from_grounded_concepts_ckb(cpnet_graph_path, kb, keywords_path, output_path, num_processes):
    """
    This function will save
        (1) adjacency matrics (each in the form of a (R*N, N) coo sparse matrix)
        (2) concepts ids
        (3) qmask that specifices whether a node is a question concept
        (4) amask that specifices whether a node is a answer concept
    to the output path in python pickle format

    grounded_path: str
    cpnet_graph_path: str
    cpnet_vocab_path: str
    output_path: str
    num_processes: int
    """
    print(f'generating adj data for {keywords_path}...')
    check_path(output_path)

    global concept2id, id2concept, relation2id, id2relation, cpnet_simple, cpnet
    if any(x is None for x in [concept2id, id2concept, relation2id, id2relation]):
        #load_resources(cpnet_vocab_path)
        concept2id = dict(kb.kb_vocab)
        id2concept = dict(kb.invert_kb_vocab)
        relation2id = dict(kb.relation2id)
        id2relation = dict(kb.id2relation)
        
    if cpnet is None or cpnet_simple is None:
        load_cpnet(cpnet_graph_path )

    keywords_dict = pickle.load(open(keywords_path, 'rb'))
    #format keywords_dict[idx] = {"cq_word":set(), "answerA":set(), "answerB":set(), "answerC":set() }

    qa_data = []
    for value in keywords_dict:
        for key, words in value.items():
            if key.startswith("answer"):
                mcp ={"qc":value['cq_word'], "ac": words}
                q_ids = set(concept2id[c] for c in mcp["qc"])
                a_ids = set(concept2id[c] for c in mcp["ac"])
                q_ids = q_ids - a_ids
                qa_data.append((q_ids, a_ids))

    with Pool(num_processes) as p:
        res = list(tqdm(p.imap(concepts_to_adj_matrices_2hop_all_pair, qa_data), total=len(qa_data)))

    # res is a list of tuples, each tuple consists of four elements (adj, concepts, qmask, amask)
    with open(output_path, 'wb') as fout:
        pickle.dump(res, fout)

    print(f'adj data saved to {output_path}')
    print()


def generate_adj_data_from_grounded_concepts(grounded_path, cpnet_graph_path, cpnet_vocab_path, output_path, num_processes):
    """
    This function will save
        (1) adjacency matrics (each in the form of a (R*N, N) coo sparse matrix)
        (2) concepts ids
        (3) qmask that specifices whether a node is a question concept
        (4) amask that specifices whether a node is a answer concept
    to the output path in python pickle format

    grounded_path: str
    cpnet_graph_path: str
    cpnet_vocab_path: str
    output_path: str
    num_processes: int
    """
    print(f'generating adj data for {grounded_path}...')

    global concept2id, id2concept, relation2id, id2relation, cpnet_simple, cpnet
    if any(x is None for x in [concept2id, id2concept, relation2id, id2relation]):
        load_resources(cpnet_vocab_path)
    if cpnet is None or cpnet_simple is None:
        load_cpnet(cpnet_graph_path)

    qa_data = []
    with open(grounded_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            dic = json.loads(line)
            q_ids = set(concept2id[c] for c in dic['qc'])
            a_ids = set(concept2id[c] for c in dic['ac'])
            q_ids = q_ids - a_ids
            qa_data.append((q_ids, a_ids))

    with Pool(num_processes) as p:
        res = list(tqdm(p.imap(concepts_to_adj_matrices_2hop_all_pair, qa_data), total=len(qa_data)))

    # res is a list of tuples, each tuple consists of four elements (adj, concepts, qmask, amask)
    with open(output_path, 'wb') as fout:
        pickle.dump(res, fout)

    print(f'adj data saved to {output_path}')
    print()


#################### adj to sparse ####################

def coo_to_normalized_per_inst(data):
    adj, concepts, qm, am, max_node_num = data
    ori_adj_len = len(concepts)
    concepts = torch.tensor(concepts[:min(len(concepts), max_node_num)])
    adj_len = len(concepts)
    qm = torch.tensor(qm[:adj_len], dtype=torch.uint8)
    am = torch.tensor(am[:adj_len], dtype=torch.uint8)
    ij = adj.row
    k = adj.col
    n_node = adj.shape[1]
    n_rel = 2 * adj.shape[0] // n_node
    i, j = ij // n_node, ij % n_node
    mask = (j < max_node_num) & (k < max_node_num)
    i, j, k = i[mask], j[mask], k[mask]
    i, j, k = np.concatenate((i, i + n_rel // 2), 0), np.concatenate((j, k), 0), np.concatenate((k, j), 0)  # add inverse relations
    adj_list = []
    for r in range(n_rel):
        mask = i == r
        ones = np.ones(mask.sum(), dtype=np.float32)
        A = sparse.csr_matrix((ones, (k[mask], j[mask])), shape=(max_node_num, max_node_num))  # A is transposed by exchanging the order of j and k
        adj_list.append(normalize_sparse_adj(A, 'coo'))
    adj_list.append(sparse.identity(max_node_num, dtype=np.float32, format='coo'))
    return ori_adj_len, adj_len, concepts, adj_list, qm, am


def coo_to_normalized(adj_path, output_path, max_node_num, num_processes):
    print(f'converting {adj_path} to normalized adj')

    with open(adj_path, 'rb') as fin:
        adj_data = pickle.load(fin)
    data = [(adj, concepts, qmask, amask, max_node_num) for adj, concepts, qmask, amask in adj_data]

    ori_adj_lengths = torch.zeros((len(data),), dtype=torch.int64)
    adj_lengths = torch.zeros((len(data),), dtype=torch.int64)
    concepts_ids = torch.zeros((len(data), max_node_num), dtype=torch.int64)
    qmask = torch.zeros((len(data), max_node_num), dtype=torch.uint8)
    amask = torch.zeros((len(data), max_node_num), dtype=torch.uint8)

    adj_data = []
    with Pool(num_processes) as p:
        for i, (ori_adj_len, adj_len, concepts, adj_list, qm, am) in tqdm(enumerate(p.imap(coo_to_normalized_per_inst, data)), total=len(data)):
            ori_adj_lengths[i] = ori_adj_len
            adj_lengths[i] = adj_len
            concepts_ids[i][:adj_len] = concepts
            qmask[i][:adj_len] = qm
            amask[i][:adj_len] = am
            adj_list = [(torch.LongTensor(np.stack((adj.row, adj.col), 0)),
                         torch.FloatTensor(adj.data)) for adj in adj_list]
            adj_data.append(adj_list)

    torch.save((ori_adj_lengths, adj_lengths, concepts_ids, adj_data), output_path)

    print(f'normalized adj saved to {output_path}')
    print()

# if __name__ == '__main__':
#     generate_adj_matrices_from_grounded_concepts('./data/csqa/grounded/train.grounded.jsonl',
#                                                  './data/cpnet/conceptnet.en.pruned.graph',
#                                                  './data/cpnet/concept.txt',
#                                                  '/tmp/asdf', 40)
