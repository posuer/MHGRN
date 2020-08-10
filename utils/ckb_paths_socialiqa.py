import os
import sys
import json
import pickle
import re
import argparse
import multiprocessing
from tqdm import tqdm
import numpy as np
from scipy.spatial import distance
import nltk
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
import string

from utils.connected_kb import Connected_KB
from utils.utils import check_path

import spacy
spacy_nlp = spacy.load("en_core_web_sm")
multi_subj_count = 0

def get_words_from_text(text): #for path ranking. exclude Person X etc. in ATOMIC
    stopwords = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than']
    persons = ["personx", "persony", "personz", "___", "x", "y", "z"]
    # and Punctuation
    keywords = [word for word in nltk.word_tokenize(text.lower()) if word not in stopwords+persons + list(string.punctuation)]
    return keywords

def get_embeddings(word_dict=None, vec_file= None, emb_size=300):
    data_dir = "data/connected_kb/"

    if os.path.exists(os.path.join(data_dir, "glove_pathRanking.pickle")):
        word_vectors = pickle.load(open(os.path.join(data_dir, "glove_pathRanking.pickle"), 'rb'))
    elif vec_file:
        word_vectors = np.random.uniform(-0.1, 0.1, (len(word_dict), emb_size))	
        f = open(vec_file, 'r', encoding='utf-8')
        vec = {}
        for line in f:
            line = line.split()
            vec[line[0]] = np.array([float(x) for x in line[-emb_size:]])
        f.close()  
        for key in word_dict:
            low = key.lower()
            if low in vec:
                word_vectors[word_dict[key]] = vec[low]

        pickle.dump(word_vectors, open(os.path.join(data_dir, "glove_pathRanking.pickle"), 'wb'))
        
    #print("word embedding loaded")
    return word_vectors


def question_type_mapping(context_text, question_text):
    #"oEffect","oReact","oWant","xAttr","xEffect","xIntent","xNeed","xReact","xWant"
    global multi_subj_count
    rel_type = None
    context_text = context_text[0].upper() + context_text[1:]
    question_text_lower = question_text.lower()
    if "describe" in question_text_lower or "think of" in question_text_lower:
        return "xAttr"
    elif "need to do before" in question_text_lower:
        return "xNeed"
    elif "why" in question_text_lower:
        return "xIntent"
    elif "feel" in question_text_lower:
        rel_type = 'React'
    elif "want" in question_text_lower or "do next" in question_text_lower:
        rel_type = 'Want'
    elif "happen" in question_text_lower or "experiencing" in question_text_lower:
        rel_type = 'Effect'
    
    if rel_type and "other" in question_text_lower:
        return "o"+rel_type
    if rel_type and "'s" in question_text_lower and "what's" not in question_text_lower:
        #print(context_text, question_text)
        return "o"+rel_type

    doc = spacy_nlp(context_text)
    sub_toks = [str(tok) for tok in doc if (tok.dep_.startswith("nsubj")) and str(tok)[0].isupper() and str(tok).lower() not in stop_words ]
    #person_toks = [ent.text.lower() for ent in doc.ents if ent.label_ == "PERSON"]
    #obj_toks = set([tok.lower() for tok in doc if (tok.dep_ in ["dobj", "pobj"])])
    question_toks = set(nltk.word_tokenize(question_text))
    #doc_question = spacy_nlp(question_text)
    #question_toks = [str(tok) for tok in doc_question if tok.dep_.startswith("nsubj") and str(tok).lower() not in stop_words]
    
    if rel_type and sub_toks and question_toks and question_toks & set([sub_toks[0]]):
        return "x"+rel_type
    elif rel_type and sub_toks:
        #print("OTHER", context_text, question_text)
        return "o"+rel_type
    #if rel_type:
    #    print(context_text, question_text)
    if rel_type :#and len(sub_toks)>1
        multi_subj_count+=1
        #print(context_text, question_text)
    return ["x"+rel_type, "o"+rel_type] if rel_type else None


def socialiqa_keywords_pathRanking(word_dict, filename):
    '''
    #Output format:
        {
            id: {"cq_word":set(), "answerA":set(), "answerB":set(), "answerC": set(), "all_ans_word": set()},
            ...
        }

    '''
    data_dir = "data/connected_kb/socialiqa/"
    question_type_mapping_NA_count = 0
    #if os.path.exists(os.path.join(data_dir, "socialIQa_v1.4_"+filename+"_keywords_pathRanking.pickle")):
    #    all_keywords_dict = pickle.load(open(os.path.join(data_dir, "socialIQa_v1.4_"+filename+"_keywords_pathRanking.pickle"), 'rb'))
    #else:
    filepath = os.path.join(data_dir, "socialIQa_v1.4_"+filename+".jsonl")
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        all_keywords_dict = [dict() for _ in range(len(lines))]
        
        for idx, line in enumerate(lines):
            path_result = []
            item = json.loads(line.strip())

            context = item["context"]
            question = item["question"]
            answers_dict = {"answerA":item["answerA"], "answerB":item["answerB"],"answerC":item["answerC"]}
            all_keywords_dict[idx] = {"cq_word":set(), "answerA":set(), "answerB":set(), "answerC":set() , "question_type":None}
            
            all_keywords_dict[idx]["cq_word"] = set(word for word in get_words_from_text(context) if word in word_dict)# "cq_word" only have context words
            
            q_type = question_type_mapping(context, question)
            if not q_type: question_type_mapping_NA_count += 1
            all_keywords_dict[idx]["question_type"] = q_type

            for key, value in answers_dict.items():
                all_keywords_dict[idx][key] = set(word for word in get_words_from_text(value) if word in word_dict)
            
            # keywords appear in ConceptNet
            all_keywords_dict[idx]["context_kb_words"] = set(word for word in kb.get_keywords_from_text(context) if word in kb.only_word_dict)
            all_keywords_dict[idx]["question_kb_words"] = set(word for word in kb.get_keywords_from_text(question) if word in kb.only_word_dict)
                        
    pickle.dump(all_keywords_dict, open(os.path.join(data_dir, "socialIQa_v1.4_"+filename+"_keywords_Q2Rel.pickle"), 'wb'))
    print("Num. of failed question to relation mapping: ",question_type_mapping_NA_count)
    print("Num. of subj and obj: ",multi_subj_count)

    return all_keywords_dict


def rank_path(kb, KBQA_word_dict, keywords, path_list, top_k, word_embeddings):

    #if len(path_list) <= top_k: #removed: ranking all paths for later analyzing
    #    return path_list
    keywords_q, keywords_a, question_type = (keywords["cq_word"], keywords["answerA"], keywords["question_type"])
    
    qa_keywords = keywords_q | keywords_a
    qa_average_embedding = np.average([word_embeddings[KBQA_word_dict[word]] for word in qa_keywords if word in KBQA_word_dict ], axis=0)
    
    similarity_score_list = []
    filter_path_list = []
    for path in path_list:
        path_keywords_list = []
        filter_flag = True
        include_ATOMIC_flag = False
        for hop_id, hop in enumerate(path):
            hop['rel_w_parent'] = hop['rel_w_parent'] if hop['rel_w_parent'] else 'cn_atomic' #if relation is None, then its 'cn_atomic'
            ##################
            # filtering path #
            ##################
            if hop_id == 0 and hop["kb_type"] == 'conceptnet' and hop["key"] not in keywords["context_kb_words"] and hop["key"] in keywords["question_kb_words"] : 
                filter_flag = False #remove source keyword from question
                break
            if hop['rel_w_parent'] not in kb.rel_grouping: #if this path contains invalid rel
                filter_flag = False #print("Skip this path due to invalid relation:", hop['rel_w_parent'] )
                break
            #if hop['rel_w_parent'] == 'cn_atomic' and hop['kb_type'] =='atomic_infer': #Only ConceptNet to ATOMIC Event (no CN to Infer)
            if hop_id >= 2 and path[hop_id-2]['kb_type']=='atomic_event' and path[hop_id-1]['kb_type']=='atomic_infer' and hop['kb_type']=='atomic_event': 
                #去掉 event - infer -event, 保留 concept-infer-event
                filter_flag = False
                break
            if question_type and hop['rel_w_parent'] in kb.atomic_rel_types and hop['rel_w_parent'] not in question_type:
                filter_flag = False # Only all relations in this path match question type
                break
            if hop["kb_type"] == 'atomic_event': # if there is ATOMIC node in path
                include_ATOMIC_flag = True

            if hop["kb_type"] == 'conceptnet':
                path_keywords_list.extend(get_words_from_text(hop["key"].replace('_',' ')))
            if hop["kb_type"] == 'atomic_event':
                path_keywords_list.extend(get_words_from_text(hop["key"]))
            if hop["kb_type"] == 'atomic_infer':
                path_keywords_list.extend(list(kb.ATOMIC_Infers[hop["key"]]["conceptnet"]))
            if hop["rel_w_parent"] and hop["rel_w_parent"] != 'cn_atomic':
                path_keywords_list.append(kb.relation_keyword_mapping( hop["rel_w_parent"] ))
       
        if filter_flag and include_ATOMIC_flag:
            filter_path_list.append(path)
            path_average_embedding = np.average([word_embeddings[KBQA_word_dict[kw]] for kw in path_keywords_list if kw in KBQA_word_dict ], axis=0)
            similarity_score_list.append(distance.cosine(qa_average_embedding, path_average_embedding))

    top_k_paths = []
    for count, index in enumerate(np.argsort(np.array(similarity_score_list))[::-1]):
        if count == top_k: break
        top_k_paths.append(filter_path_list[index])
    
    return top_k_paths

def merge_path(input_paras):
    global task_vocab_dir
    KBQA_word_dict, paths_answer, keywords, word_embeddings, top_k = input_paras
    #KBQA_word_dict = dict(KBQA_word_dict)
    kb = Connected_KB()
    '''
    KBQA_word_dict = kb.get_full_word_vocab()
    if task_vocab_dir: # add vocab from QA task
        with open(task_vocab_dir, 'r') as task_vocab_file:
            task_vocab = json.loads(task_vocab_file)
        for key in task_vocab:
            if key not in KBQA_word_dict:
                word_dict[word] = len(KBQA_word_dict)
    '''
    #vec_file = "data/glove/glove.6B.300d.txt"
    #word_embeddings = get_embeddings(KBQA_word_dict, vec_file)

    pruned_path = []

    #print(keywords_qa)
    for keywrod_pair in paths_answer: #for each keyword pair in one answer's paths
        keywrod_pair_paths = keywrod_pair["paths"]
        ranked_keywrod_pair_list = rank_path(kb, KBQA_word_dict, keywords, keywrod_pair_paths, top_k, word_embeddings)
        
        merged_path_list_nodes = []
        merged_path_list = []
        for path in ranked_keywrod_pair_list: #for each path in one keyword pair's paths
            node_list = []
            rel_list = []
            for hop_id, hop in enumerate(path):
                node_list.append(kb.kb_vocab[(hop['kb_type'],hop['key'])])
                if hop_id == 0: #first node, no relation
                    continue 
                
                hop['rel_w_parent'] = hop['rel_w_parent'] if hop['rel_w_parent'] else 'cn_atomic' #if relation is None, then its 'cn_atomic'
                # if hop['rel_w_parent'] not in kb.rel_grouping: #if this path contains invalid rel
                #     node_list = []
                #     #print("Skip this path due to invalid relation:", hop['rel_w_parent'] )
                #     break
                # if hop['rel_w_parent'] == 'cn_atomic' and hop['kb_type'] =='atomic_infer': #Only ConceptNet to ATOMIC Event (no CN to Infer)
                #     node_list = []
                #     break

                grouped_rel = kb.rel_grouping[hop['rel_w_parent']]
                this_hop_dir = hop['rel_direction']
                if grouped_rel.startswith('*'): # if start with *, reverse the dirction eg:('causes': 'causes', 'motivatedbygoal': '*causes',)
                    this_hop_dir = 0 if this_hop_dir else 1
                    grouped_rel = grouped_rel[1:]
                if this_hop_dir == 1:#if opposite direction, get opposite id by adding length of all relations
                    rel_list.append([kb.relation2id[grouped_rel] + len(kb.relation2id)])
                else: 
                    rel_list.append([kb.relation2id[grouped_rel]])
                
            if node_list in merged_path_list_nodes:
                idx = merged_path_list_nodes.index(node_list)
                assert len(merged_path_list[idx]['rel']) == len(rel_list)
                id_list = [i for i in range(len(merged_path_list[idx]['rel']))]
                for hop_id, merged_rel, new_rel in zip(id_list, merged_path_list[idx]['rel'], rel_list):
                    merged_path_list[idx]['rel'][hop_id] = list(set(merged_rel).union(set(new_rel)))
            elif node_list:
                merged_path_list.append({"path":node_list, "rel":rel_list})
                merged_path_list_nodes.append(node_list)

        if merged_path_list:
            pruned_path.append({"qc": kb.lemmatize_word(keywrod_pair['cq_word']), "ac": kb.lemmatize_word(keywrod_pair['a_word']), "pf_res": merged_path_list})
        
    return pruned_path


def prune_path(kb, vec_file, trn_dev_tst, num_machine=1, machine_id="00", top_k = 100, emb_size=300):
    '''
    Splited file into pieces and run on mutiple machines

    file input: {idx: {'answerA':[{"cq_word":cq_word, "a_word":a_word, "paths":paths}], 
        'answerB':list(), 'answerC':list()}
    output: ranked, pruned paths, each line for one answer's paths
    [
        {"qc": word, "ac": word, "pf_res": merged_path_list}
        ...
    ]
    '''
    KBQA_word_dict = kb.get_full_word_vocab()
    if task_vocab_dir: # add vocab from QA task
        with open(task_vocab_dir, 'r') as task_vocab_file:
            task_vocab = json.load(task_vocab_file)
        for key in task_vocab:
            if key not in KBQA_word_dict:
                KBQA_word_dict[key] = len(KBQA_word_dict)

    vec_file = "data/glove/glove.6B.300d.txt"
    word_embeddings = get_embeddings(KBQA_word_dict, vec_file)

    cores = multiprocessing.cpu_count()
    print("Runing on", cores, "CPUs.")

    data_dir = 'data/connected_kb/'+"socialiqa/paths/raw_paths"
    output_dir = 'data/connected_kb/'+"socialiqa/paths/"

    if trn_dev_tst == "dev":
        writer = open(os.path.join(output_dir, "socialIQa_v1.4_dev_pruned_path_Q2Rel_noCN.jsonl"), 'w')
        keywords = socialiqa_keywords_pathRanking(KBQA_word_dict, 'dev')
    elif trn_dev_tst == "tst":    
        writer = open(os.path.join(output_dir, "socialIQa_v1.4_tst_pruned_path_Q2Rel_noCN.jsonl"), 'w')
        keywords = socialiqa_keywords_pathRanking(KBQA_word_dict, 'tst')    
    else:
        writer = open(os.path.join(output_dir, "socialIQa_v1.4_trn_pruned_path_Q2Rel_noCN.jsonl."+machine_id), 'w')
        keywords = socialiqa_keywords_pathRanking(KBQA_word_dict, 'trn')

    exit()
    file_list_all = os.listdir(data_dir)
    file_list_all.sort()

    file_list = []
    for file_name in file_list_all:
        if file_name.split("_")[2] == trn_dev_tst:
            #num_splited_file = max(int(file_name.split("_")[3][1:]), num_splited_file)
            file_list.append(file_name)
    
    num_splited_file = len(file_list)
    raw_file_id_head = 0
    raw_file_id_tail = num_splited_file
    if num_machine>1:
        num_file_each_machine = int(num_splited_file / num_machine)
        raw_file_id_head = int(machine_id) * num_file_each_machine
        raw_file_id_tail = min((int(machine_id)+1) * num_file_each_machine, num_splited_file)
    file_list = file_list[raw_file_id_head:raw_file_id_tail]
    #print(file_list)
    for file_split_name in file_list:
        print(file_split_name, "pruned paths start...")

        all_paths = pickle.load(open(os.path.join(data_dir, file_split_name), 'rb'))

        multi_process_input = []
        for idx, paths_qa in tqdm(all_paths.items(), total=len(all_paths)):
            #print(str(paths_qa)[:300])
            multi_process_input.append((KBQA_word_dict, paths_qa['answerA'], keywords[idx], word_embeddings, top_k))
            multi_process_input.append((KBQA_word_dict, paths_qa['answerB'], keywords[idx], word_embeddings, top_k))
            multi_process_input.append((KBQA_word_dict, paths_qa['answerC'], keywords[idx], word_embeddings, top_k))
        '''
        for this_input in tqdm(multi_process_input[:6], total=len(multi_process_input)):
            pruned_path = merge_path(this_input)
            if pruned_path:
                writer.write(json.dumps(pruned_path) + '\n')
            else:
                writer.write('[]\n')
        
        '''
        with multiprocessing.Pool(cores) as p:
            for pruned_path in tqdm(p.imap(merge_path, multi_process_input), total=len(multi_process_input)):
                if pruned_path:
                    writer.write(json.dumps(pruned_path) + '\n')
                else:
                    writer.write('[]\n')
        
        print(file_split_name, "pruned paths saved")
        writer.flush()

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    #parser.add_argument('--statement_path', default=None)
    #parser.add_argument('--raw_path_path', default=None)
    #parser.add_argument('--kb', default='connected_kb', choices=['connected_kb', 'cpnet'])
    parser.add_argument('--task_vocab_dir', default=None)
    parser.add_argument('--split', default='train', choices=['trn', 'dev', 'tst'], required=True)
    parser.add_argument('-ds', '--dataset', default='csqa', choices=['csqa', 'socialiqa', 'obqa'], required=True)
    parser.add_argument('--top_k',type=int, default=100, help="Keep top n paths for each keywrod pair when prune path")  
    parser.add_argument('--num_machine',type=int, default=1)
    parser.add_argument('--machine_id',type=str, default='00', help="must be 00, 01, 02 ...") 
    
    args = parser.parse_args()
    parser.set_defaults(task_vocab_dir=f'./data/{args.dataset}/statement/vocab.json',
                        )
    args = parser.parse_args()

    global task_vocab_dir
    task_vocab_dir = args.task_vocab_dir

    print("Loading KB ...")
    kb = Connected_KB()

    #socialiqa_keywords_load(kb, 'tst')
    #socialiqa_keywords_load(kb, 'dev')
    #socialiqa_keywords_load(kb, 'trn')

    #print("Start searching for", filename)
    #path_retrieve(kb, filename)
    glove_path = "data/glove/glove.6B.300d.txt"
    prune_path(kb, glove_path, args.split, args.num_machine, args.machine_id, args.top_k)
