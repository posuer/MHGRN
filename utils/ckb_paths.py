import os
import sys
import json
import pickle
import multiprocessing
from tqdm import tqdm
import numpy as np
from scipy.spatial import distance

from utils.connected_kb import Connected_KB
from utils.utils import check_path

def get_embeddings(word_dict=None, vec_file= None, emb_size=300):
    data_dir = "data/connected_kb/"

    if os.path.exists(os.path.join(data_dir, "glove.pickle")):
        word_vectors = pickle.load(open(os.path.join(data_dir, "glove.pickle"), 'rb'))
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

        pickle.dump(word_vectors, open(os.path.join(data_dir, "glove.pickle"), 'wb'))
        
    #print("word embedding loaded")
    return word_vectors


def ground_multi_process(line):
    word_embeddings = get_embeddings()

    item = json.loads(line.strip())

    context = item["context"]
    question = item["question"]
    answers_dict = {"answerA":item["answerA"], "answerB":item["answerB"],"answerC":item["answerC"]}
    
    all_keywords_dict = {"cq_word":set(), "answerA":set(), "answerB":set(), "answerC":set() }

    cq_words = kb.get_keywords_from_text(context+' '+question)
    for word in cq_words:
        if word in kb.ConceptNet:
            all_keywords_dict["cq_word"].add(('conceptnet',word))
        
        # for kb_type, kb_sub in kb.ConceptNet[word].items():
        #     if kb_type in ['atomic_event', 'atomic_infer']:
        #         for kb_key in kb_sub:
        #             all_keywords_dict[idx]["cq_word"].add((kb_type, kb_key))

    #all_keywords_dict[idx]["all_ans_word"] = kb.get_keywords_from_text(' '.join(answers_dict.values())) #found paths were not splited for each answer
    qa_keywords = all_keywords_dict["cq_word"]
    #print(qa_keywords)
    qa_average_embedding = np.average([word_embeddings[kb.only_word_dict[word]] for word in qa_keywords if word in kb.only_word_dict ], axis=0)


    for key, value in answers_dict.items():
        a_words = kb.get_keywords_from_text(value)
        atomic_list = set()
        for word in a_words:
            if word in kb.ConceptNet:
                all_keywords_dict[key].add(('conceptnet',word)) #add Concept Net
                for kb_type, kb_sub in kb.ConceptNet[word].items():
                    if kb_type in ['atomic_event', 'atomic_infer']:
                        for kb_key in kb_sub:
                            atomic_list.add((kb_type, kb_key))
        similarity_score_list = []
        atomic_list = list(atomic_list)
        for kb_type, kb_key in atomic_list:
            path_keywords_list = []
            if kb_type == 'atomic_event':
                path_keywords_list.extend(list(kb.ATOMIC_Events[kb_key]["conceptnet"]))
            if kb_type == 'atomic_infer':
                path_keywords_list.extend(list(kb.ATOMIC_Infers[kb_key]["conceptnet"]))
            path_average_embedding = np.average([word_embeddings[kb.only_word_dict[kw]] for kw in path_keywords_list if kw in kb.only_word_dict ], axis=0)

            similarity_score_list.append(distance.cosine(qa_average_embedding, path_average_embedding))
        top_k_paths = []
        for count, index in enumerate(np.argsort(np.array(similarity_score_list))[::-1]):
            if count == 3: break
            top_k_paths.append(atomic_list[index])
        
        all_keywords_dict[key].update(top_k_paths) # add ranked top n ATOMIC
    return all_keywords_dict

def socialiqa_keywords_ground(kb, filename):
    '''
    #Output format:
        {
            id: {"cq_word":set(), "answerA":set(), "answerB":set(), "answerC": set(), "all_ans_word": set()},
            ...
        }

    '''
    data_dir = "data/connected_kb/socialiqa/"

    if os.path.exists(os.path.join(data_dir, "socialIQa_v1.4_"+filename+"_ground_keywords.pickle")):
        all_keywords_dict = pickle.load(open(os.path.join(data_dir, "socialIQa_v1.4_"+filename+"_ground_keywords.pickle"), 'rb'))
    else:
        #word_embeddings = get_embeddings(kb.only_word_dict)
        cores = multiprocessing.cpu_count()
        print("Runing on", cores, "CPUs.")
        filepath = os.path.join(data_dir, "socialIQa_v1.4_"+filename+".jsonl")
        all_keywords_dict = []
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            with multiprocessing.Pool(cores) as p:
                for grounded_keys_qa in tqdm(p.imap(ground_multi_process, lines), total=len(lines)):
                    all_keywords_dict.append(grounded_keys_qa)
            '''
            for idx, line in tqdm(enumerate(lines), total=len(lines)):

                item = json.loads(line.strip())

                context = item["context"]
                question = item["question"]
                answers_dict = {"answerA":item["answerA"], "answerB":item["answerB"],"answerC":item["answerC"]}
                
                all_keywords_dict[idx] = {"cq_word":set(), "answerA":set(), "answerB":set(), "answerC":set() }

                cq_words = kb.get_keywords_from_text(context+' '+question)
                for word in cq_words:
                    if word in kb.ConceptNet:
                        all_keywords_dict[idx]["cq_word"].add(('conceptnet',word))
                    
                    # for kb_type, kb_sub in kb.ConceptNet[word].items():
                    #     if kb_type in ['atomic_event', 'atomic_infer']:
                    #         for kb_key in kb_sub:
                    #             all_keywords_dict[idx]["cq_word"].add((kb_type, kb_key))

                #all_keywords_dict[idx]["all_ans_word"] = kb.get_keywords_from_text(' '.join(answers_dict.values())) #found paths were not splited for each answer
                qa_keywords = all_keywords_dict[idx]["cq_word"]
                #print(qa_keywords)
                qa_average_embedding = np.average([word_embeddings[kb.only_word_dict[word]] for word in qa_keywords if word in kb.only_word_dict ], axis=0)


                for key, value in answers_dict.items():
                    a_words = kb.get_keywords_from_text(value)
                    atomic_list = set()
                    for word in a_words:
                        if word in kb.ConceptNet:
                            all_keywords_dict[idx][key].add(('conceptnet',word)) #add Concept Net
                            for kb_type, kb_sub in kb.ConceptNet[word].items():
                                if kb_type in ['atomic_event', 'atomic_infer']:
                                    for kb_key in kb_sub:
                                        atomic_list.add((kb_type, kb_key))
                    similarity_score_list = []
                    atomic_list = list(atomic_list)
                    for kb_type, kb_key in atomic_list:
                        path_keywords_list = []
                        if kb_type == 'atomic_event':
                            path_keywords_list.extend(list(kb.ATOMIC_Events[kb_key]["conceptnet"]))
                        if kb_type == 'atomic_infer':
                            path_keywords_list.extend(list(kb.ATOMIC_Infers[kb_key]["conceptnet"]))
                        path_average_embedding = np.average([word_embeddings[kb.only_word_dict[kw]] for kw in path_keywords_list if kw in kb.only_word_dict ], axis=0)
        
                        similarity_score_list.append(distance.cosine(qa_average_embedding, path_average_embedding))
                    top_k_paths = []
                    for count, index in enumerate(np.argsort(np.array(similarity_score_list))[::-1]):
                        if count == 3: break
                        top_k_paths.append(atomic_list[index])
                    
                    all_keywords_dict[idx][key].update(top_k_paths) # add ranked top n ATOMIC
            '''
        pickle.dump(all_keywords_dict, open(os.path.join(data_dir, "socialIQa_v1.4_"+filename+"_ground_keywords.pickle"), 'wb'))
    
    return all_keywords_dict


def csqa_keywords_load(kb, input_file, output_file):
    '''
    #Output format:
        {
            id: {"cq_word":set(), "answerA":set(), "answerB":set(), "answerC": set(), "all_ans_word": set()},
            ...
        }

    '''
    check_path(output_file)

    with open(input_file, 'r', encoding='utf-8') as f:

        lines = f.readlines()
        all_keywords_dict = [dict() for _ in range(len(lines))]
        for idx, line in enumerate(lines):
            json_dic = json.loads(line)

            answers_dict = {"answerA":json_dic["question"]["choices"][0]["text"], 
                "answerB":json_dic["question"]["choices"][1]["text"],
                "answerC":json_dic["question"]["choices"][2]["text"],
                "answerD":json_dic["question"]["choices"][3]["text"],
                "answerE":json_dic["question"]["choices"][4]["text"],
                }
            
            all_keywords_dict[idx] = {"cq_word":set(), "answerA":set(), "answerB":set(), "answerC":set(), "answerD":set(), "answerE":set() }

            all_keywords_dict[idx]["cq_word"] = set(('conceptnet',word) for word in kb.get_keywords_from_text(json_dic["question"]["stem"]) if word in kb.only_word_dict)
            for key, value in answers_dict.items():
                all_keywords_dict[idx][key] = set(('conceptnet',word) for word in kb.get_keywords_from_text(value) if word in kb.only_word_dict)
                if len(all_keywords_dict[idx]["cq_word"]) == 0 and len(all_keywords_dict[idx][key]) == 0:
                    question_concept = json_dic["question"]["question_concept"].replace(" ", "_")
                    all_keywords_dict[idx]["cq_word"] = {('conceptnet',question_concept)} if question_concept in kb.only_word_dict else set()
                    if len(all_keywords_dict[idx]["cq_word"]) == 0:
                        print("No keywords!", idx, all_keywords_dict)
                        exit() 
    pickle.dump(all_keywords_dict, open(output_file, 'wb'))

    return all_keywords_dict

def socialiqa_keywords_load(kb, filename):
    '''
    #Output format:
        {
            id: {"cq_word":set(), "answerA":set(), "answerB":set(), "answerC": set(), "all_ans_word": set()},
            ...
        }

    '''
    data_dir = "data/connected_kb/socialiqa/"

    if os.path.exists(os.path.join(data_dir, "socialIQa_v1.4_"+filename+"_keywords.pickle")):
        all_keywords_dict = pickle.load(open(os.path.join(data_dir, "socialIQa_v1.4_"+filename+"_keywords.pickle"), 'rb'))
    else:
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
                
                all_keywords_dict[idx] = {"cq_word":set(), "answerA":set(), "answerB":set(), "answerC":set() }

                all_keywords_dict[idx]["cq_word"] = set(('conceptnet',word) for word in kb.get_keywords_from_text(context+' '+question) if word in kb.only_word_dict)
                all_keywords_dict[idx]["all_ans_word"] = kb.get_keywords_from_text(' '.join(answers_dict.values())) #found paths were not splited for each answer
                
                for key, value in answers_dict.items():
                    all_keywords_dict[idx][key] = set(('conceptnet',word) for word in kb.get_keywords_from_text(value) if word in kb.only_word_dict)
                
        pickle.dump(all_keywords_dict, open(os.path.join(data_dir, "socialIQa_v1.4_"+filename+"_keywords.pickle"), 'wb'))
    
    return all_keywords_dict

def rank_path(kb, keywords_q, keywords_a, path_list, top_k, word_embeddings):

    if len(path_list) <= top_k:
        return path_list
    
    qa_keywords = keywords_q | keywords_a
    #print(qa_keywords)
    qa_average_embedding = np.average([word_embeddings[kb.only_word_dict[word]] for word in qa_keywords if word in kb.only_word_dict ], axis=0)
    
    similarity_score_list = []

    for path in path_list:
        path_keywords_list = []

        for hop in path:
            if hop["kb_type"] == 'conceptnet':
                path_keywords_list.extend(kb.get_keywords_from_text(hop["key"].replace('_',' ')))
            if hop["kb_type"] == 'atomic_event':
                path_keywords_list.extend(list(kb.ATOMIC_Events[hop["key"]]["conceptnet"]))
            if hop["kb_type"] == 'atomic_infer':
                path_keywords_list.extend(list(kb.ATOMIC_Infers[hop["key"]]["conceptnet"]))
            if hop["rel_w_parent"]:
                path_keywords_list.append(kb.relation_keyword_mapping( hop["rel_w_parent"] ))
        
        #print(path_keywords_list)
        path_average_embedding = np.average([word_embeddings[kb.only_word_dict[kw]] for kw in path_keywords_list if kw in kb.only_word_dict ], axis=0)
        
        similarity_score_list.append(distance.cosine(qa_average_embedding, path_average_embedding))

    top_k_paths = []
    for count, index in enumerate(np.argsort(np.array(similarity_score_list))[::-1]):
        if count == top_k: break
        top_k_paths.append(path_list[index])
    
    return top_k_paths

def merge_path(input_paras):
    paths_answer, keywords_cq, keywords_ans, top_k = input_paras
    kb = Connected_KB()
    vec_file = "data/glove/glove.6B.300d.txt"
    word_embeddings = get_embeddings(kb.only_word_dict, vec_file)

    pruned_path = []

    #print(keywords_qa)
    for keywrod_pair in paths_answer: #for each keyword pair in one answer's paths
        keywrod_pair_paths = keywrod_pair["paths"]
        ranked_keywrod_pair_list = rank_path(kb, keywords_cq, keywords_ans, keywrod_pair_paths, top_k, word_embeddings)
        
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
                if hop['rel_w_parent'] not in kb.rel_grouping: #if this path contains invalid
                    node_list = []
                    #print("Skip this path due to invalid relation:", hop['rel_w_parent'] )
                    break
                grouped_rel = kb.rel_grouping[hop['rel_w_parent']]
                this_hop_dir = hop['rel_direction']
                if grouped_rel.startswith('*'): # if start with #, reverse the dirction eg:('causes': 'causes', 'motivatedbygoal': '*causes',)
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


        with multiprocessing.Pool(cores) as p:
            for pruned_path in tqdm(p.imap(merge_path, multi_process_input), total=len(multi_process_input)):
                if pruned_path:
                    writer_dict[trn_dev_tst].write(json.dumps(pruned_path) + '\n')
                else:
                    writer_dict[trn_dev_tst].write('[]\n')
        #'''
        print(file_split_name, "pruned paths saved")
        
    if writer_trn: writer_trn.close()
    if writer_dev: writer_dev.close()
    if writer_tst: writer_tst.close()
        

def prune_path(kb, vec_file, top_k = 100, emb_size=300):
    '''
    file input: {idx: {'answerA':[{"cq_word":cq_word, "a_word":a_word, "paths":paths}], 
        'answerB':list(), 'answerC':list()}
    output: ranked, pruned paths, each line for one answer's paths
    [
        {"qc": word, "ac": word, "pf_res": merged_path_list}
        ...
    ]
    '''
    #word_embeddings = get_embeddings(kb.only_word_dict, vec_file)

    cores = multiprocessing.cpu_count()
    print("Runing on", cores, "CPUs.")

    data_dir = 'data/connected_kb/'+"socialiqa/paths/raw_paths"

    output_dir = 'data/connected_kb/'+"socialiqa/paths/"
    writer_trn = open(os.path.join(output_dir, "socialIQa_v1.4_trn_pruned_path.jsonl"), 'w')
    writer_dev = open(os.path.join(output_dir, "socialIQa_v1.4_dev_pruned_path.jsonl"), 'w')
    writer_tst = open(os.path.join(output_dir, "socialIQa_v1.4_tst_pruned_path.jsonl"), 'w')
    writer_dict = {"trn":writer_trn, "dev":writer_dev, "tst":writer_tst,}

    keywords_trn = socialiqa_keywords_load(kb, 'trn')
    keywords_dev = socialiqa_keywords_load(kb, 'dev')
    keywords_tst = socialiqa_keywords_load(kb, 'tst')
    keywords_dict = {"trn":keywords_trn, "dev":keywords_dev, "tst":keywords_tst,}

    file_list = os.listdir(data_dir)
    file_list.sort()
    file_list = file_list
    #print(file_list)
    for file_split_name in file_list:
        print(file_split_name, "pruned paths start...")

        all_paths = pickle.load(open(os.path.join(data_dir, file_split_name), 'rb'))
        trn_dev_tst = file_split_name.split('_')[2]

        multi_process_input = []
        for idx, paths_qa in tqdm(all_paths.items(), total=len(all_paths)):
            #print(str(paths_qa)[:300])
            multi_process_input.append((paths_qa['answerA'], keywords_dict[trn_dev_tst][idx]["cq_word"], keywords_dict[trn_dev_tst][idx]["answerA"], top_k))
            multi_process_input.append((paths_qa['answerB'], keywords_dict[trn_dev_tst][idx]["cq_word"], keywords_dict[trn_dev_tst][idx]["answerB"], top_k))
            multi_process_input.append((paths_qa['answerC'], keywords_dict[trn_dev_tst][idx]["cq_word"], keywords_dict[trn_dev_tst][idx]["answerC"], top_k))#, word_embeddings

        with multiprocessing.Pool(cores) as p:
            for pruned_path in tqdm(p.imap(merge_path, multi_process_input), total=len(multi_process_input)):
                if pruned_path:
                    writer_dict[trn_dev_tst].write(json.dumps(pruned_path) + '\n')
                else:
                    writer_dict[trn_dev_tst].write('[]\n')
        #'''
        print(file_split_name, "pruned paths saved")
        
    writer_trn.close()
    writer_dev.close()
    writer_tst.close()
        

def multi_processing(idx, anw_choice, cq_word, a_word, max_hops):
    kb_sub = Connected_KB()
    return idx, anw_choice, cq_word, a_word, kb_sub.search_path_byEntity(cq_word, a_word, max_hops)

def path_retrieve(kb, filename, num_hop=3):
    '''
    output: {idx: {'answerA':[{"cq_word":cq_word, "a_word":a_word, "paths":paths}], 
        'answerB':list(), 'answerC':list()}
    '''
    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cores)
    print("Runing on", cores, "CPUs.")
    
    data_dir = 'data/connected_kb/'+"socialiqa/paths/"
    
    all_qa = socialiqa_keywords_load(kb, filename.split('_')[0])
    this_split = int(filename.split('_s')[1])
    this_split_range = (this_split*750, min((this_split+1)*750, len(all_qa)))

    all_query_list = []
    #idx = 0
    for idx in range(this_split_range[0], this_split_range[1]):
        for cq_word in all_qa[idx]['cq_word']:
            for a_word in all_qa[idx]['answerA']:
                all_query_list.append((idx, 'answerA', cq_word, a_word, num_hop))
            for a_word in all_qa[idx]['answerB']:
                all_query_list.append((idx, 'answerB', cq_word, a_word, num_hop))
            for a_word in all_qa[idx]['answerC']:
                all_query_list.append((idx, 'answerC', cq_word, a_word, num_hop))
        #idx += 1
        
    # multi processing for all QA pair
    all_result = {idx: {'answerA':list(), 'answerB':list(), 'answerC':list()} for idx in range(this_split_range[0], this_split_range[1])}
    print(len(all_qa), this_split_range)
    for idx, anw_choice, cq_word, a_word, paths in pool.starmap(multi_processing, all_query_list):
        if paths:
            all_result[idx][anw_choice].append({"cq_word":cq_word, "a_word":a_word, "paths":paths})

        #if idx % 300 == 0:
        #    pickle.dump(all_result, open(os.path.join(data_dir, "socialIQa_v1.4_"+filename+"_knowledgePath.pickle"), 'wb'))
    
    #with open(os.path.join(data_dir, "socialIQa_v1.4_"+filename+"_knowledgePath_debug.txt"), 'w') as w:
    #    print(all_result, file=w)
    pickle.dump(all_result, open(os.path.join(data_dir, "socialIQa_v1.4_"+filename+"_Path.pickle"), 'wb'))
    

if __name__ == "__main__":
    #from connected_kb import Connected_KB
    print("Loading KB ...")
    kb = Connected_KB()

    socialiqa_keywords_load(kb, 'tst')
    socialiqa_keywords_load(kb, 'dev')
    socialiqa_keywords_load(kb, 'trn')

    #print("Start searching for", filename)
    #path_retrieve(kb, filename)
    

    # print("processing:", 'tst')
    # socialiqa_keywords_ground(kb, 'tst')
    # print("processing:", 'dev')
    # socialiqa_keywords_ground(kb, 'dev')
    # print("processing:", 'trn')
    # socialiqa_keywords_ground(kb, 'trn')


