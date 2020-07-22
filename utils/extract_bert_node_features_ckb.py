import torch
import torch.nn as nn
import pickle
from tqdm import tqdm
import json
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer
import argparse
import os
import time
#try:
#    from .utils import check_path
#    from .utils.connected_kb import Connected_KB
from utils.connected_kb import Connected_KB
from utils.utils import check_path

#except:
#    from utils import check_path
#    from utils.connected_kb import Connected_KB


global id2concept

def load_resources(cpnet_vocab_path):
    with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
        return [w.strip() for w in fin]


def format_infer(inf):
    inf = inf.lower()
    inf = inf.replace("personx's", "his").replace("persony's", "her")
    inf = inf.replace("person x's", "his").replace("person y's", "her")
    inf = inf.replace("person x", "he").replace("person y", "she").replace("person z", "the person") \
                     .replace(" x ", " he ").replace(" y ", " she ").replace(" z ", " the person ") \
                     .replace("personx", "he").replace("persony", "she").replace("personz", "the person") \
                     .strip()
    if inf.endswith(" x"):
        inf = inf.replace(" x", " he") 
    if inf.endswith(" y"):
        inf = inf.replace(" y", " she")
    if inf.endswith(" z"):
        inf = inf.replace(" z", " the person")
        
    if inf.startswith("x "):
        inf = inf.replace("x ", "he ") 
    if inf.startswith("y "):
        inf = inf.replace("y ", "she ")
    if inf.startswith("z "):
        inf = inf.replace("z ", "the person ")
    return inf

def format_event(inf):
    inf = inf.lower()#.replace('___','[MASK]')
    inf = inf.replace("personx's", "his").replace("persony's", "her")
    inf = inf.replace("personx", "he").replace("persony", "her").replace("personz", "the person") \
                     .strip()
    return inf

def convert_qa_concept_to_bert_input(tokenizer, question, answer, concept, max_seq_length):
    qa_tokens = tokenizer.tokenize('Q: ' + question + ' A: ' + answer)
    concept_tokens = tokenizer.tokenize(concept)
    qa_tokens = qa_tokens[:max_seq_length - len(concept_tokens) - 3]
    tokens = [tokenizer.cls_token] + qa_tokens + [tokenizer.sep_token] + concept_tokens + [tokenizer.sep_token]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    segment_ids = [0] * (len(qa_tokens) + 2) + [1] * (len(concept_tokens) + 1)

    assert len(input_ids) == len(segment_ids) == len(input_ids)

    # padding
    padding_id = 0 if args.model=='bert' else 1 #args.model=='roberta'
    pad_len = max_seq_length - len(input_ids)
    input_mask = [1] * len(input_ids) + [0] * pad_len
    input_ids += [padding_id] * pad_len
    segment_ids += [0] * pad_len
    span = (len(qa_tokens) + 2, len(qa_tokens) + 2 + len(concept_tokens))

    assert span[1] + 1 == len(tokens)
    assert max_seq_length == len(input_ids) == len(segment_ids) == len(input_mask)

    return input_ids, input_mask, segment_ids, span


def extract_bert_node_features_from_adj(args, cpnet_vocab_path, statement_path, adj_path, output_path, max_seq_length, device, batch_size, layer_id=-1, cache_path=None, use_cache=True, distr_size=1, distr_nth="00"):
    global id2concept

    check_path(output_path)

    print('extracting from triple strings')
    
    if args.model == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', do_lower_case=True)
        model = BertModel.from_pretrained('bert-large-uncased', output_hidden_states=True)#.to(device)
    elif args.model == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained("roberta-large", do_lower_case=True)
        model = RobertaModel.from_pretrained('roberta-large', output_hidden_states=True)#.to(device)
    
    #if torch.cuda.device_count() > 1:
        #print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        #model = nn.DataParallel(model) 
        #TODO need to check really using multi GPU
    model.to(device)
    
    #model = DistributedDataParallel(model) # device_ids will include all GPU devices by default
    model.eval()

    #print("sleeping for 1.5 hours" )
    #time.sleep(60*60*1.5)
    
    if use_cache and os.path.isfile(cache_path):
        print('Loading cached inputs.')
        with open(cache_path, 'rb') as fin:
            all_input_ids, all_input_mask, all_segment_ids, all_span, offsets = pickle.load(fin)
        print('Loaded')
    else:

        with open(adj_path, 'rb') as fin:
            adj_data = pickle.load(fin)

        offsets = [0]
        all_input_ids, all_input_mask, all_segment_ids, all_span = [], [], [], []
        with open(statement_path, 'r') as fin:
            statement_lines = fin.readlines()
            for line in tqdm(statement_lines, total=len(statement_lines), desc='Calculating alignments'):
                dic = json.loads(line)
                question = dic['question']['stem']
                for choice in dic['question']['choices']:
                    answer = choice['text']
                    adj, concepts_ids, _, _ = adj_data.pop(0)
                    if args.kb =='cpnet':
                        concepts = [id2concept[c].replace('_', ' ') for c in concepts_ids]
                    elif args.kb =='connected_kb':
                        concepts = []
                        for c in concepts_ids:
                            kb_type, kb_key = id2concept[c]
                            if kb_type == 'conceptnet':
                                concepts.append(kb_key.replace('_', ' '))
                            elif kb_type == 'atomic_event':
                                concepts.append(format_event(kb_key))
                            elif kb_type == 'atomic_infer':
                                concepts.append(format_infer(kb_key))

                    offsets.append(offsets[-1] + len(concepts))
                    for concept in concepts:
                        input_ids, input_mask, segment_ids, span = convert_qa_concept_to_bert_input(tokenizer, question, answer, concept, max_seq_length)
                        all_input_ids.append(input_ids)
                        all_input_mask.append(input_mask)
                        all_segment_ids.append(segment_ids)
                        all_span.append(span)

        assert len(adj_data) == 0
        check_path(cache_path)
        with open(cache_path, 'wb') as fout:
            pickle.dump((all_input_ids, all_input_mask, all_segment_ids, all_span, offsets), fout)
        print('Inputs dumped')

    
    if distr_size>1: #split data and run on multiple machine
        this_distr = int(distr_nth)
        total_size = len(all_input_ids)
        this_distr_size = total_size//distr_size
        this_distr_head = this_distr*this_distr_size
        this_distr_tail = min((this_distr+1)*this_distr_size, total_size)
        all_input_ids, all_input_mask, all_segment_ids, all_span = [torch.tensor(x, dtype=torch.long) for x in [all_input_ids[this_distr_head: this_distr_tail], all_input_mask[this_distr_head: this_distr_tail], all_segment_ids[this_distr_head: this_distr_tail], all_span[this_distr_head: this_distr_tail]]]
    else:
        all_input_ids, all_input_mask, all_segment_ids, all_span = [torch.tensor(x, dtype=torch.long) for x in [all_input_ids, all_input_mask, all_segment_ids, all_span]]
    all_span = all_span.to(device)

    concept_vecs = []
    n = all_input_ids.size(0)

    with torch.no_grad():
        for a in tqdm(range(0, n, batch_size), total=n // batch_size + 1, desc='Extracting features'):
            b = min(a + batch_size, n)
            if args.model == 'bert':
                batch = [x.to(device) for x in [all_input_ids[a:b], all_input_mask[a:b], all_segment_ids[a:b]]]
            elif args.model == 'roberta':
                batch = [x.to(device) for x in [all_input_ids[a:b], all_input_mask[a:b]]]
            #print(batch)
            outputs = model(*batch)
            hidden_states = outputs[-1][layer_id]
            mask = torch.arange(max_seq_length, device=device)[None, :]
            mask = (mask >= all_span[a:b, 0, None]) & (mask < all_span[a:b, 1, None])
            pooled = (hidden_states * mask.float().unsqueeze(-1)).sum(1)
            pooled = pooled / (all_span[a:b, 1].float() - all_span[a:b, 0].float() + 1e-5).unsqueeze(1)
            concept_vecs.append(pooled.cpu())
        concept_vecs = torch.cat(concept_vecs, 0).numpy()
    if distr_size>1: 
        res = [concept_vecs[i] for i in range(len(concept_vecs))]
        output_path=output_path+distr_nth
        #import pdb; pdb.set_trace()
        #print("size:", concept_vecs.size )
        print("actually output path:", output_path, "\n Group by offsets in data_utils.py ")
    else:
        res = [concept_vecs[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)]

    with open(output_path, 'wb') as fout:
        pickle.dump(res, fout)

    print('done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--statement_path', default=None)
    parser.add_argument('--adj_path', default=None)
    parser.add_argument('--output_path', default=None)
    parser.add_argument('--split', default='train', choices=['train', 'dev', 'test'], required=True)
    parser.add_argument('-ds', '--dataset', default='csqa', choices=['csqa', 'socialiqa', 'obqa'], required=True)
    parser.add_argument('--layer_id', type=int, default=-1)
    parser.add_argument('--max_seq_length', type=int, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--cpnet_vocab_path', default='./data/cpnet/concept.txt')
    parser.add_argument('--cache_path', default=None)
    
    parser.add_argument('--gpu', default=True)
    parser.add_argument('--model', default='bert', choices=['bert', 'roberta'])
    parser.add_argument('--path_hop', default='2hop', choices=['2hop', '3hop'])
    parser.add_argument('--kb', default='connected_kb', choices=['connected_kb', 'cpnet'])
    parser.add_argument('--max_node_num',type=int, default=None, help=" ")  
    parser.add_argument('--distr_size',type=int, default=1)
    parser.add_argument('--distr_nth',type=str, default='00', help="must be 00, 01, 02 ...") 
    

    args = parser.parse_args()

    kb_data_dir = "connected_kb/" if args.kb == "connected_kb" else ""
    adj_graph_file = ".pruned.pk.npath20" if args.path_hop == "3hop" else ".pk"
    mxnode_file = ".mxnode"+str(args.max_node_num) if args.max_node_num and args.dataset == "socialiqa" else ""
    parser.set_defaults(statement_path=f'./data/{args.dataset}/statement/{args.split}.statement.jsonl',
                        adj_path=f'./data/{kb_data_dir}{args.dataset}/graph/{args.split}.graph.adj{adj_graph_file}{mxnode_file}',
                        output_path=f'./data/{kb_data_dir}{args.dataset}/features/{args.split}.{args.model}.layer{args.layer_id}.{args.path_hop}.pk',
                        cache_path=f'./data/{kb_data_dir}{args.dataset}/features/{args.split}.{args.model}.inputs.{args.path_hop}.pk')
    args = parser.parse_args()
    print(args)
    
    if args.kb == "connected_kb":
        ckb = Connected_KB()
        id2concept = ckb.invert_kb_vocab
    else:
        id2concept = load_resources(cpnet_vocab_path=args.cpnet_vocab_path)
    #torch.distributed.init_process_group(backend="nccl")
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")

    extract_bert_node_features_from_adj(args, cpnet_vocab_path=args.cpnet_vocab_path,
                                        statement_path=args.statement_path,
                                        adj_path=args.adj_path,
                                        output_path=args.output_path,
                                        max_seq_length=args.max_seq_length,
                                        device=device,
                                        batch_size=args.batch_size,
                                        layer_id=args.layer_id,
                                        cache_path=args.cache_path,
                                        distr_size=args.distr_size,
                                        distr_nth=args.distr_nth,
                                        )
