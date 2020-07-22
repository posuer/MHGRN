import os
import csv
import pickle
import nltk
import networkx as nx
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer 

# from utils.conceptnet import load_merge_relation, merged_relations

relation_groups = [
    'atlocation/locatednear',
    'capableof',
    'causes/causesdesire/*motivatedbygoal',
    'createdby',
    'desires',
    #'antonym/distinctfrom',
    #'hascontext',
    'hasproperty',
    'hassubevent/hasfirstsubevent/haslastsubevent/hasprerequisite/entails/mannerof',
    'isa/instanceof/definedas',
    'madeof',
    'notcapableof',
    'notdesires',
    'partof/*hasa',
    #'relatedto/similarto/synonym',
    'usedfor',
    'receivesaction',
]

merged_relations = [
    #'antonym',
    'atlocation',
    'capableof',
    'causes',
    'createdby',
    'isa',
    'desires',
    'hassubevent',
    'partof',
    #'hascontext',
    'hasproperty',
    'madeof',
    'notcapableof',
    'notdesires',
    'receivesaction',
    #'relatedto',
    'usedfor',
]

# discard: ["relatedto", "synonym", "antonym", hasContext, "derivedfrom", "formof", "etymologicallyderivedfrom", "etymologicallyrelatedto", "externalurl"] 
def load_merge_relation():
    relation_mapping = dict()
    for line in relation_groups:
        ls = line.strip().split('/')
        rel = ls[0]
        for l in ls:
            if l.startswith("*"):
                relation_mapping[l[1:]] = "*" + rel
            else:
                relation_mapping[l] = rel
    return relation_mapping


'''
path hop structure
{
"kb_type": this_kb_type,
"key": key,
"rel_w_parent": rel_w_parent,
"rel_direction": direction,
}
'''
class Connected_KB(object):
    def __init__(self, data_dir = 'data/'):
        self.data_dir = data_dir + "connected_kb/"

        self.ConceptNet = {}
        self.ATOMIC_Events = {}
        self.ATOMIC_Infers = {}
        self.G = nx.MultiGraph()

        self.kb_vocab, self.invert_kb_vocab = (dict(), dict()) #keys: (kb_type, key)

        self.rel_grouping = self.__relation_grouping() 
        self.relation2id, self.id2relation = self.__relation_id_map()

        self.lemmatizer = WordNetLemmatizer() 
        '''
        try:
            self.lemmatizer.lemmatize("visits")
            nltk.pos_tag(nltk.word_tokenize("PersonX abuses PersonX's power".lower()))
        except:
            nltk.download('wordnet')
            nltk.download('punkt')
            nltk.download('averaged_perceptron_tagger')
        '''
        self.kb_load()
        self.vocab_load()
        self.only_word_dict = dict([(key[1],value) for key, value in self.kb_vocab.items() if key[0]=='conceptnet'])

    def __relation_grouping(self, ):
        rel_grouping = load_merge_relation() #conceptnet, * means opposite relation direction 
        for i in ["oEffect","oReact","oWant","xAttr","xEffect","xIntent","xNeed","xReact","xWant",'cn_atomic','nothasproperty']:
            rel_grouping[i] = i
        return rel_grouping

    def relation_keyword_mapping(self, rel):
        if rel not in self.rel_grouping:
            return None
        rel = self.rel_grouping[rel]
        if rel.startswith("*"): rel = rel[1:]
        map_dict =  {"oEffect":"effect","oReact":"feel","oWant":"want","xAttr":"is","xEffect":"effect","xIntent":"want","xNeed":"need","xReact":"feel","xWant":"want",
            'antonym':'antonym',
            'atlocation':'locate',
            'capableof':'capable',
            'causes':'cause',
            'createdby':'create',
            'isa':'kind',
            'desires':'desire',
            'hassubevent':'event',
            'partof':'part',
            'hascontext':'context',
            'hasproperty':'property',
            'madeof':'made',
            'notcapableof':'capable',
            'notdesires':'desire',
            'receivesaction':'is',
            'relatedto':'related',
            'usedfor':'used',
            'nothasproperty':'property'
            }

        return map_dict[rel]


    def __relation_id_map(self):
        relation2id = {}
        all_rels = ['cn_atomic','nothasproperty'] + merged_relations + ["oEffect","oReact","oWant","xAttr","xEffect","xIntent","xNeed","xReact","xWant"]
        for key in all_rels:
            #if key not in relation2id:
            relation2id[key] = len(relation2id)
            #relation2id['*'+key] = len(all_rels) + len(relation2id)
        id2relation = {r: i for i, r in relation2id.items()}

        return relation2id, id2relation
    
    def __format_atomic_infer(self, inf):
        inf = inf.lower().replace("person x", "personx").replace("person y", "persony").replace("person z", "personz") \
                        .replace(" x ", " personx ").replace(" y ", " persony ").replace(" z ", " personz ") \
                        .strip()
        if inf.endswith(" x"):
            inf = inf.replace(" x", " personx") 
        if inf.endswith(" y"):
            inf = inf.replace(" y", " persony")
        if inf.endswith(" z"):
            inf = inf.replace(" z", " personz")
            
        if inf.startswith("x "):
            inf = inf.replace("x ", "personx ") 
        if inf.startswith("y "):
            inf = inf.replace("y ", "persony ")
        if inf.startswith("z "):
            inf = inf.replace("z ", "personz ")
        return inf

    def __get_keywords_from_text(self, text): #for linking KBs
        stopwords = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than']
        persons = ["personx", "persony","personz", "___"]
        keywords = [word for (word, pos) in nltk.pos_tag(nltk.word_tokenize(text.lower())) if pos[:2] in ['NN', 'JJ', 'VB', "RB"] and word not in stopwords+persons ]
        keywords_lem = [self.lemmatizer.lemmatize(word) for word in keywords] 
        return keywords_lem

    def get_keywords_from_text(self, text): #for QA text
        stopwords = ['get', "n't",'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than']
        keywords = [word for (word, pos) in nltk.pos_tag(nltk.word_tokenize(text.lower())) if pos[:2] in ['NN', 'JJ', 'VB', "RB"] and word not in stopwords ]
        keywords_lem = set([self.lemmatizer.lemmatize(word) for word in keywords])
        return keywords_lem

    def lemmatize_word(self, word):

        word_lem = self.lemmatizer.lemmatize(word)
        if word_lem in self.ConceptNet.keys():
            return word_lem

        if word_lem not in self.ConceptNet.keys() and word_lem == word:  #fix failed lemmatization. eg: enjoys
            for pos_tag in ['a','v','n']:
                word_lem = self.lemmatizer.lemmatize(word, pos=pos_tag)
                if word_lem in self.ConceptNet.keys():
                    return word_lem
        
        return None
   
    def search_path_byEntity1(self, start_word, end_word, max_hops): 
        start_word = self.lemmatize_word(start_word)
        end_word = self.lemmatize_word(end_word)
        if not (start_word and end_word):
            return None

        paths = nx.all_simple_paths(self.G, source=('conceptnet', start_word), target=('conceptnet',end_word), cutoff=max_hops)
        return paths
    
    
    def search_path_byEntity(self, start_word, end_word, max_hops): 
        start_word = self.lemmatize_word(start_word)
        end_word = self.lemmatize_word(end_word)
        if not (start_word and end_word):
            return None

        connected_KB ={
            'conceptnet':self.ConceptNet,
            'atomic_event':self.ATOMIC_Events,
            'atomic_infer':self.ATOMIC_Infers
        }
        visited = set()
        all_path = []

        def search_all_path(path, this_kb_type, key, rel_w_parent, direction):
            visited.add((this_kb_type, key))
            path.append({
                    "kb_type": this_kb_type,
                    "key": key,
                    "rel_w_parent": rel_w_parent,
                    "rel_direction": direction,
                    })
            
            if this_kb_type == 'conceptnet' and key == end_word: #if found destination in conceptnet node
                all_path.append(list(path))
            elif end_word in connected_KB[this_kb_type][key]['conceptnet']: #if found destination in event or infer node
                if len(path)>=2 and path[-2]["kb_type"] != 'conceptnet': #avoid last two hop be "concept - event/infer", when less than maxhops
                    all_path.append(list(path))
            elif len(path) <= max_hops and len(all_path) <= 1000:
                this_node = connected_KB[this_kb_type][key]

                for kb_type, relations in this_node.items():
                    if this_kb_type == 'conceptnet':
                        if kb_type == 'conceptnet':
                            for neighbor in relations:
                                if (kb_type, neighbor["tail"]) not in visited:
                                    search_all_path(path, kb_type, neighbor["tail"], neighbor["rel"], neighbor["direction"])

                        elif kb_type == 'atomic_event' or kb_type == 'atomic_infer' and len(path)!=max_hops: #avoid last two hop be "concept - event/infer", purne searching space
                            for neighbor in relations:
                                if (kb_type, neighbor) not in visited:
                                    search_all_path(path, kb_type, neighbor, None, None)
                            
                    elif this_kb_type == 'atomic_event' or this_kb_type == 'atomic_infer':
                        if kb_type == 'conceptnet':
                            if len(path)>=2 and path[-2]["kb_type"] == 'conceptnet':
                                continue #avoid path like "conceptnet - event/infer - conceptnet"
                            for neighbor in relations:
                                if (kb_type, neighbor) not in visited:
                                    search_all_path(path, kb_type, neighbor, None, None)
                            
                        elif kb_type == 'atomic_event':
                            for neighbor in relations:
                                if (kb_type, neighbor["tail"]) not in visited:
                                    search_all_path(path, kb_type, neighbor["tail"], neighbor["rel"], None)
                            
                        elif kb_type == 'atomic_infer':
                            for infer_type, neighbor_list in relations.items():
                                if infer_type == 'prefix': continue
                                for neighbor in neighbor_list:
                                    if (kb_type, neighbor) not in visited:
                                        search_all_path(path, kb_type, neighbor, infer_type, None)
            
            path.pop()
            visited.remove((this_kb_type, key))
        
        path = []
        search_all_path(path, 'conceptnet', start_word, None, None)
        return all_path
            


    def retrieve_neighbor_atomic(self):
        pass
    
    def retrieve_neighbor_conceptnet(self):
        pass



    def vocab_load(self):

        if os.path.exists(self.data_dir+"connected_kb_vocab.pickle"):
            self.kb_vocab, self.invert_kb_vocab = pickle.load(open( self.data_dir+"connected_kb_vocab.pickle", "rb" ))
        else:
            for key in self.ConceptNet:
                if ('conceptnet', key) not in self.kb_vocab:
                    self.kb_vocab[('conceptnet', key)] = len(self.kb_vocab)
            for key in self.ATOMIC_Events:
                if ('atomic_event', key) not in self.kb_vocab:
                    self.kb_vocab[('atomic_event', key)] = len(self.kb_vocab)
            for key in self.ATOMIC_Infers:
                if ('atomic_infer', key) not in self.kb_vocab:
                    self.kb_vocab[('atomic_infer', key)] = len(self.kb_vocab)

            for key, idx in self.kb_vocab.items():
                self.invert_kb_vocab[idx] = key
            
            pickle.dump((self.kb_vocab, self.invert_kb_vocab), open( self.data_dir+"connected_kb_vocab.pickle", "wb" ) )

    def construct_nx_graph(self, output_graph_dir):
        triple_set = set()

        def add_to_graph(head, rel, tail, direction = 0):
            if rel not in self.rel_grouping:
                return
            if self.rel_grouping[rel].startswith('*'):
                rel_id = self.relation2id[self.rel_grouping[rel][1:]] + len(self.relation2id)
                rel_id_invert = self.relation2id[self.rel_grouping[rel][1:]]
            else:
                rel_id = self.relation2id[self.rel_grouping[rel]]
                rel_id_invert = self.relation2id[self.rel_grouping[rel]] + len(self.relation2id)
            
            subj = self.kb_vocab[head]
            obj = self.kb_vocab[tail]
            if (rel_id, subj, obj) in triple_set or subj == obj:
                return
            else:
                if direction == 0:
                    triple_set.add((rel_id, subj, obj))
                    triple_set.add((rel_id_invert, obj, subj))

                    self.G.add_edge(subj, obj, rel=rel_id)
                    self.G.add_edge(obj, subj, rel=rel_id_invert)
                elif direction == 1:
                    triple_set.add((rel_id_invert, subj, obj))
                    triple_set.add((rel_id, obj, subj))

                    self.G.add_edge(subj, obj, rel=rel_id_invert)
                    self.G.add_edge(obj, subj, rel=rel_id)

        #if os.path.exists(output_graph_dir):
        #    self.G = pickle.load(open(output_graph_dir, "rb" ) )
        #else:
        for head, cn_event_infer in tqdm(self.ConceptNet.items(),desc="Add ConceptNet to graph"):
            for tail_type, tail_list in cn_event_infer.items():
                if tail_type == "conceptnet":
                    for tail in tail_list:
                        add_to_graph(('conceptnet',head), tail['rel'], ('conceptnet',tail['tail']), tail['direction'])
                elif tail_type == "atomic_event":
                    for tail in tail_list:
                        add_to_graph(('conceptnet',head), 'cn_atomic', ('atomic_event',tail))
                elif tail_type == "atomic_infer":
                    for tail in tail_list:
                        add_to_graph(('conceptnet',head), 'cn_atomic', ('atomic_infer',tail))
        
        for head, cn_event_infer in tqdm(self.ATOMIC_Events.items(),desc="Add ATOMIC Events to graph"):
            for tail_type, tail_list in cn_event_infer.items():
                if tail_type == "conceptnet":
                    for tail in tail_list:
                        add_to_graph(('atomic_event',head), 'cn_atomic', ('conceptnet',tail), 1)
                elif tail_type == "atomic_infer":
                    for rel, tails in tail_list.items():
                        if rel != 'prefix':
                            for tail in tails:
                                add_to_graph(('atomic_event',head), rel, ('atomic_infer',tail))

        for head, cn_event_infer in tqdm(self.ATOMIC_Infers.items(),desc="Add ATOMIC Infers to graph"):
            for tail_type, tail_list in cn_event_infer.items():
                if tail_type == "conceptnet":
                    for tail in tail_list:
                        add_to_graph(('atomic_infer',head), 'cn_atomic', ('conceptnet',tail), 1)
                elif tail_type == "atomic_event":
                    for tail in tail_list:
                        add_to_graph(('atomic_infer',head), tail['rel'], ('atomic_event',tail['tail']), 1)

        pickle.dump(self.G, open( output_graph_dir, "wb" ) )

        print(f"networkx graph file saved to {output_graph_dir}")
        print()
    

    def kb_load(self):
        if os.path.exists(self.data_dir+"ConceptNet_en_QA.pickle") and os.path.exists(self.data_dir+"ATOMIC_v4_Events.pickle") and os.path.exists(self.data_dir+"ATOMIC_v4_Infers.pickle"):# and os.path.exists("ATOMIC_ConceptNet_wholeGraph.pickle"):
            self.ConceptNet = pickle.load(open( self.data_dir+"ConceptNet_en_QA.pickle", "rb" ))
            self.ATOMIC_Events = pickle.load(open( self.data_dir+"ATOMIC_v4_Events.pickle", "rb" ))
            self.ATOMIC_Infers = pickle.load(open( self.data_dir+"ATOMIC_v4_Infers.pickle", "rb" ))
        
        else:
            #ConceptNet
            discarded_rel = ["hascontext", "relatedto", "synonym", "antonym", "derivedfrom", "formof", "etymologicallyderivedfrom", "etymologicallyrelatedto", "externalurl"]
            with open("data/cpnet/conceptnet-assertions-5.6.0.csv",'r',encoding='utf-8') as reader:
                csv_reader = csv.reader(reader, delimiter='\t')
                for line in tqdm(csv_reader):
                    head = line[2].split('/')
                    tail = line[3].split('/')
                    if head[2] == "en" and tail[2] == "en":
                        relation = line[1].split('/')[2].lower()
                        if relation in discarded_rel: continue #pass discarded_rel

                        head_key = head[3].lower()
                        tail_key = tail[3].lower()
                        if head_key not in self.ConceptNet.keys(): self.ConceptNet[head_key] = {"atomic_event":[], "conceptnet":[], "atomic_infer":set()}
                        self.ConceptNet[head_key]["conceptnet"].append({"rel": relation, "tail": tail_key, 'direction':0 })
                        if tail_key not in self.ConceptNet.keys(): self.ConceptNet[tail_key] = {"atomic_event":[], "conceptnet":[], "atomic_infer":set()}
                        self.ConceptNet[tail_key]["conceptnet"].append({"rel": relation, "tail": head_key, 'direction':1 })
                        
                        #self.G.add_edge(('conceptnet',head_key), ('conceptnet',tail_key), rel= relation, head=head_key, tail=tail_key  )


            #ATOMIC
            relations = ["oEffect","oReact","oWant","xAttr","xEffect","xIntent","xNeed","xReact","xWant","prefix"]
            num = 0
            with open("data/atomic/v4_atomic_all_agg.csv") as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for row in tqdm(csv_reader):
                    if num == 0:
                        num +=1 
                        continue
                    event = row[0]
                    infers = [row[i].strip('][').replace('"', "").split(', ') for i in range(1,11)]
                    infers_clear = []
                    for i in infers:
                        infers_clear.append([self.__format_atomic_infer(x) for x in i if x != 'none'])   
                    
                    self.ATOMIC_Events[event]= {"atomic_infer": dict( [(rel, set(infer)) for rel, infer in zip(relations, infers_clear)]), "conceptnet":set()}
                    
                    #for rel, infers in zip(relations, infers_clear):
                        #self.G.add_node(('atomic', event, infer, rel) )
                    #    for infer in infers:
                    #        self.G.add_edge(('atomic_event', event), ("atomic_infer", infer), rel=rel) 

            for key, value in tqdm(self.ATOMIC_Events.items()):
                for prefix in value["atomic_infer"]["prefix"]:
                    word_lem = self.lemmatize_word(prefix)
                    if word_lem:
                        self.ConceptNet[word_lem]["atomic_event"].append(key)
                        self.ATOMIC_Events[key]["conceptnet"].add(word_lem)
                        
                        #self.G.add_edge(('atomic_event', key), ("conceptnet", word_lem)) 

                for inf_key, inf_value in value["atomic_infer"].items():
                    if inf_key != "prefix":
                        for infer_phrase in inf_value:
                            #self.G.add_edge(('conceptnet', word_lem), ('atomic', key, infer_phrase, inf_key) )

                            if infer_phrase not in self.ATOMIC_Infers.keys():  self.ATOMIC_Infers[infer_phrase] = {"conceptnet":set(), "atomic_event":[]}
                            self.ATOMIC_Infers[infer_phrase]["atomic_event"].append({"rel":inf_key,"tail":key})

                            for conceptnet_key in self.__get_keywords_from_text(infer_phrase):
                                word_lem = self.lemmatize_word(conceptnet_key)
                                if word_lem:
                                    self.ATOMIC_Infers[infer_phrase]["conceptnet"].add(word_lem)
                                    self.ConceptNet[word_lem]['atomic_infer'].add(infer_phrase)

                                    #self.G.add_edge(('conceptnet', word_lem), ('atomic_infer', infer_phrase) )


            pickle.dump(self.ConceptNet, open( self.data_dir+"ConceptNet_en_QA.pickle", "wb" ) )
            pickle.dump(self.ATOMIC_Events, open( self.data_dir+"ATOMIC_v4_Events.pickle", "wb" ) )
            pickle.dump(self.ATOMIC_Infers, open( self.data_dir+"ATOMIC_v4_Infers.pickle", "wb" ) )
            #pickle.dump(self.G, open( "ATOMIC_ConceptNet_wholeGraph.pickle", "wb" ) )

            
               

if __name__ == "__main__":
    connected_KB = Connected_KB()
    

    