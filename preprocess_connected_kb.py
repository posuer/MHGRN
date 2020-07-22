import os
import argparse
from multiprocessing import cpu_count


from utils.connected_kb import Connected_KB
from utils.ckb_paths import prune_path, csqa_keywords_load
from utils.convert_socialiqa import convert_to_socialiqa_statement
from utils.tokenization_utils import tokenize_statement_file, make_word_vocab
from utils.grounding import create_matcher_patterns, ground

from utils.embedding import glove2npy, load_pretrained_embeddings_ckb
from utils.graph import generate_graph_ckb, generate_adj_data_from_grounded_concepts_ckb, generate_adj_data_from_pruned_paths_ckb

input_paths = {

    'socialiqa': {
        'train': './data/socialiqa/socialiqa-train-dev/train.jsonl',
        'dev': './data/socialiqa/socialiqa-train-dev/dev.jsonl',
        'test': './data/socialiqa/socialiqa-train-dev/test.jsonl',
        'train-label': './data/socialiqa/socialiqa-train-dev/train-labels.lst',
        'dev-label': './data/socialiqa/socialiqa-train-dev/dev-labels.lst',
        'test-label': './data/socialiqa/socialiqa-train-dev/test-labels.lst',
    },
    'cpnet': {
        'csv': './data/cpnet/conceptnet-assertions-5.6.0.csv',
    },
    'glove': {
        'txt': './data/glove/glove.6B.300d.txt',
    },
    'numberbatch': {
        'txt': './data/transe/numberbatch-en-19.08.txt',
    },
    'transe': {
        'ent': './data/transe/glove.transe.sgd.ent.npy',
        'rel': './data/transe/glove.transe.sgd.rel.npy',
    },
}

output_paths = {
    'cpnet': {
        'csv': './data/cpnet/conceptnet.en.csv',
        'vocab': './data/cpnet/concept.txt',
        'patterns': './data/cpnet/matcher_patterns.json',
        'unpruned-graph': './data/cpnet/conceptnet.en.unpruned.graph',
        'pruned-graph': './data/cpnet/conceptnet.en.pruned.graph',
    },
    'conncected_kb': {
        'pruned-graph': './data/connected_kb/connected_kb_nxGraph.pickle',
        'simple-graph': './data/connected_kb/connected_kb_simple_Graph.pickle',
    },
    'glove': {
        'npy': './data/glove/glove.6B.300d.npy',
        'vocab': './data/glove/glove.vocab',
    },
    'numberbatch': {
        'npy': './data/transe/nb.npy',
        'vocab': './data/transe/nb.vocab',
        'concept_npy': './data/connected_kb/transe/concept.nb.npy'
    },

    'socialiqa': {
        'statement': {
            'train': './data/socialiqa/statement/train.statement.jsonl',
            'dev': './data/socialiqa/statement/dev.statement.jsonl',
            'test': './data/socialiqa/statement/test.statement.jsonl',
            'train-fairseq': './data/socialiqa/fairseq/official/train.jsonl',
            'dev-fairseq': './data/socialiqa/fairseq/official/valid.jsonl',
            'test-fairseq': './data/socialiqa/fairseq/official/test.jsonl',
            'vocab': './data/socialiqa/statement/vocab.json',
        },
        'tokenized': {
            'train': './data/socialiqa/tokenized/train.tokenized.txt',
            'dev': './data/socialiqa/tokenized/dev.tokenized.txt',
            'test': './data/socialiqa/tokenized/test.tokenized.txt',
        },
        'keywords': {
            'train': './data/connected_kb/socialiqa/socialIQa_v1.4_trn_keywords.pickle',
            'dev': './data/connected_kb/socialiqa/socialIQa_v1.4_dev_keywords.pickle',
            'test': './data/connected_kb/socialiqa/socialIQa_v1.4_tst_keywords.pickle',
        },
        'grounded': {
            'train': './data/connected_kb/socialiqa/socialIQa_v1.4_trn_ground_keywords.pickle',
            'dev': './data/connected_kb/socialiqa/socialIQa_v1.4_dev_ground_keywords.pickle',
            'test': './data/connected_kb/socialiqa/socialIQa_v1.4_tst_ground_keywords.pickle',
        },
        'paths': {
            #'raw-train': './data/socialiqa/paths/train.paths.raw.jsonl',
            #'raw-dev': './data/socialiqa/paths/dev.paths.raw.jsonl',
            #'scores-train': './data/socialiqa/paths/train.paths.scores.jsonl',
            #'scores-dev': './data/socialiqa/paths/dev.paths.scores.jsonl',
            'pruned-train': './data/connected_kb/socialiqa/paths/socialIQa_v1.4_trn_pruned_path.jsonl',
            'pruned-dev': './data/connected_kb/socialiqa/paths/socialIQa_v1.4_dev_pruned_path.jsonl',
            'pruned-test': './data/connected_kb/socialiqa/paths/socialIQa_v1.4_tst_pruned_path.jsonl',
            'adj-train': './data/connected_kb/socialiqa/paths/train.paths.adj.jsonl',
            'adj-dev': './data/connected_kb/socialiqa/paths/dev.paths.adj.jsonl',
            'adj-test': './data/connected_kb/socialiqa/paths/test.paths.adj.jsonl',
        },
        'graph': {
            'train': './data/connected_kb/socialiqa/graph/train.graph.jsonl',
            'dev': './data/connected_kb/socialiqa/graph/dev.graph.jsonl',
            'test': './data/connected_kb/socialiqa/graph/test.graph.jsonl',
            'adj-train': './data/connected_kb/socialiqa/graph/train.graph.adj.pk',
            'adj-dev': './data/connected_kb/socialiqa/graph/dev.graph.adj.pk',
            'adj-test': './data/connected_kb/socialiqa/graph/test.graph.adj.pk',
            'pruned-adj-train': './data/connected_kb/socialiqa/graph/train.graph.adj.pruned.pk',
            'pruned-adj-dev': './data/connected_kb/socialiqa/graph/dev.graph.adj.pruned.pk',
            'pruned-adj-test': './data/connected_kb/socialiqa/graph/test.graph.adj.pruned.pk',
        },
        'triple': {
            'train': './data/connected_kb/socialiqa/triples/train.triples.pk',
            'dev': './data/connected_kb/socialiqa/triples/dev.triples.pk',
            'test': './data/connected_kb/socialiqa/triples/test.triples.pk',
        },
    },
    'csqa': {
        'statement': {
            'train': './data/csqa/statement/train.statement.jsonl',
            'dev': './data/csqa/statement/dev.statement.jsonl',
            'test': './data/csqa/statement/test.statement.jsonl',
            'vocab': './data/csqa/statement/vocab.json',
        },
        'statement-with-ans-pos': {
            'train': './data/csqa/statement/train.statement-with-ans-pos.jsonl',
            'dev': './data/csqa/statement/dev.statement-with-ans-pos.jsonl',
            'test': './data/csqa/statement/test.statement-with-ans-pos.jsonl',
        },
        'tokenized': {
            'train': './data/csqa/tokenized/train.tokenized.txt',
            'dev': './data/csqa/tokenized/dev.tokenized.txt',
            'test': './data/csqa/tokenized/test.tokenized.txt',
        },
        'grounded': {
            'train': './data/csqa/grounded/train.grounded.jsonl',
            'dev': './data/csqa/grounded/dev.grounded.jsonl',
            'test': './data/csqa/grounded/test.grounded.jsonl',
        },
        'keywords': {
            'train': './data/connected_kb/csqa/trn_keywords.pickle',
            'dev': './data/connected_kb/csqa/dev_keywords.pickle',
            'test': './data/connected_kb/csqa/tst_keywords.pickle',
        },
        'paths': {
            'raw-train': './data/csqa/paths/train.paths.raw.jsonl',
            'raw-dev': './data/csqa/paths/dev.paths.raw.jsonl',
            'raw-test': './data/csqa/paths/test.paths.raw.jsonl',
            'scores-train': './data/csqa/paths/train.paths.scores.jsonl',
            'scores-dev': './data/csqa/paths/dev.paths.scores.jsonl',
            'scores-test': './data/csqa/paths/test.paths.scores.jsonl',
            'pruned-train': './data/csqa/paths/train.paths.pruned.jsonl',
            'pruned-dev': './data/csqa/paths/dev.paths.pruned.jsonl',
            'pruned-test': './data/csqa/paths/test.paths.pruned.jsonl',
            'adj-train': './data/csqa/paths/train.paths.adj.jsonl',
            'adj-dev': './data/csqa/paths/dev.paths.adj.jsonl',
            'adj-test': './data/csqa/paths/test.paths.adj.jsonl',
        },
        'graph': {
            'train': './data/csqa/graph/train.graph.jsonl',
            'dev': './data/csqa/graph/dev.graph.jsonl',
            'test': './data/csqa/graph/test.graph.jsonl',
            'adj-train': './data/connected_kb/csqa/graph/train.graph.adj.pk',
            'adj-dev': './data/connected_kb/csqa/graph/dev.graph.adj.pk',
            'adj-test': './data/connected_kb/csqa/graph/test.graph.adj.pk',
            'nxg-from-adj-train': './data/csqa/graph/train.graph.adj.jsonl',
            'nxg-from-adj-dev': './data/csqa/graph/dev.graph.adj.jsonl',
            'nxg-from-adj-test': './data/csqa/graph/test.graph.adj.jsonl',
        },
        'triple': {
            'train': './data/csqa/triples/train.triples.pk',
            'dev': './data/csqa/triples/dev.triples.pk',
            'test': './data/csqa/triples/test.triples.pk',
        },
    },
}



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default=['common', 'csqa'], choices=['connected_kb','common', 'csqa', 'hswag', 'anli', 'exp', 'scitail', 'phys', 'socialiqa', 'obqa', 'make_word_vocab'], nargs='+')
    parser.add_argument('--path_prune_threshold', type=float, default=0.12, help='threshold for pruning paths')
    parser.add_argument('--max_node_num', type=int, default=0, help='maximum number of nodes per graph')
    parser.add_argument('-p', '--nprocs', type=int, default=cpu_count(), help='number of processes to use')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--debug', action='store_true', help='enable debug mode')
    parser.add_argument('--path_prune_topn', type=int, default=None, help='pruning top n paths for each keyword pair')


    args = parser.parse_args()
    if args.debug:
        raise NotImplementedError()

    c_kb = Connected_KB()

    routines = {
        'connected_kb':[ #Gengyu
            #{'func': glove2npy, 'args': (input_paths['glove']['txt'], output_paths['glove']['npy'], output_paths['glove']['vocab'])},
            #{'func': glove2npy, 'args': (input_paths['numberbatch']['txt'], output_paths['numberbatch']['npy'], output_paths['numberbatch']['vocab'], True)},
            #unique id for each node and relation: output_paths['cpnet']['vocab'], output_paths['cpnet']['pruned-graph'],
            
            #{'func': c_kb.construct_nx_graph, 'args': (output_paths['conncected_kb']['pruned-graph'],)},
            #{'func': load_pretrained_embeddings_ckb, 
            # 'args': (output_paths['numberbatch']['npy'], output_paths['numberbatch']['vocab'], c_kb, False, output_paths['numberbatch']['concept_npy'])},
        ],
        'csqa': [ 
            {'func': csqa_keywords_load, 'args': (c_kb, output_paths["csqa"]["statement"]["train"], output_paths["csqa"]["keywords"]["train"])},
            {'func': csqa_keywords_load, 'args': (c_kb, output_paths["csqa"]["statement"]["dev"], output_paths["csqa"]["keywords"]["dev"])},
            {'func': csqa_keywords_load, 'args': (c_kb, output_paths["csqa"]["statement"]["test"], output_paths["csqa"]["keywords"]["test"])},

            {'func': generate_adj_data_from_grounded_concepts_ckb, 'args': (output_paths['conncected_kb']['pruned-graph'], 
                                                                       c_kb, output_paths['csqa']['keywords']['train'], output_paths['csqa']['graph']['adj-train'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts_ckb, 'args': (output_paths['conncected_kb']['pruned-graph'], 
                                                                       c_kb, output_paths['csqa']['keywords']['dev'], output_paths['csqa']['graph']['adj-dev'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts_ckb, 'args': (output_paths['conncected_kb']['pruned-graph'], 
                                                                       c_kb, output_paths['csqa']['keywords']['test'], output_paths['csqa']['graph']['adj-test'], args.nprocs)},

        ],
        'socialiqa': [ #Gengyu
            #tranfer to each qa pair: output_paths['socialiqa']['grounded']['train']
            #path for each qa pair: output_paths['socialiqa']['paths']['pruned-train']
            #{'func': convert_to_socialiqa_statement, 'args': (
            #    input_paths['socialiqa']['train'], input_paths['socialiqa']['train-label'], output_paths['socialiqa']['statement']['train'], output_paths['socialiqa']['statement']['train-fairseq'])},
            #{'func': convert_to_socialiqa_statement,
            # 'args': (input_paths['socialiqa']['dev'], input_paths['socialiqa']['dev-label'], output_paths['socialiqa']['statement']['dev'], output_paths['socialiqa']['statement']['dev-fairseq'])},
            #{'func': convert_to_socialiqa_statement,
            # 'args': (input_paths['socialiqa']['test'], input_paths['socialiqa']['test-label'], output_paths['socialiqa']['statement']['test'], output_paths['socialiqa']['statement']['test-fairseq'])},
            #{'func': tokenize_statement_file, 'args': (output_paths['socialiqa']['statement']['train'], output_paths['socialiqa']['tokenized']['train'])},
            #{'func': tokenize_statement_file, 'args': (output_paths['socialiqa']['statement']['dev'], output_paths['socialiqa']['tokenized']['dev'])},
            #{'func': make_word_vocab, 'args': ((output_paths['socialiqa']['statement']['train'],), output_paths['socialiqa']['statement']['vocab'])},

            #{'func': prune_path, 'args': (c_kb, input_paths['glove']['txt'])},
            #{'func': generate_graph_ckb, 'args': (output_paths['socialiqa']['keywords']['train'], output_paths['socialiqa']['paths']['pruned-train'],
            #                                 c_kb, output_paths['conncected_kb']['pruned-graph'],
            #                                 output_paths['socialiqa']['graph']['train'])},
            # {'func': generate_graph_ckb, 'args': (output_paths['socialiqa']['keywords']['dev'], output_paths['socialiqa']['paths']['pruned-dev'],
            #                                   c_kb, output_paths['conncected_kb']['pruned-graph'],
            #                                   output_paths['socialiqa']['graph']['dev'])},
            # {'func': generate_graph_ckb, 'args': (output_paths['socialiqa']['keywords']['test'], output_paths['socialiqa']['paths']['pruned-test'],
            #                                  c_kb, output_paths['conncected_kb']['pruned-graph'],
            #                                  output_paths['socialiqa']['graph']['test'])},
            #{'func': generate_adj_data_from_grounded_concepts_ckb, 'args': (output_paths['conncected_kb']['pruned-graph'],
            #                                                           c_kb, output_paths['socialiqa']['keywords']['train'], output_paths['socialiqa']['graph']['adj-train'], args.nprocs)},
            #{'func': generate_adj_data_from_grounded_concepts_ckb, 'args': (output_paths['conncected_kb']['pruned-graph'], 
            #                                                           c_kb, output_paths['socialiqa']['keywords']['dev'], output_paths['socialiqa']['graph']['adj-dev'], args.nprocs)},
            #{'func': generate_adj_data_from_grounded_concepts_ckb, 'args': (output_paths['conncected_kb']['pruned-graph'], 
            #                                                           c_kb, output_paths['socialiqa']['keywords']['test'], output_paths['socialiqa']['graph']['adj-test'], args.nprocs)},
            # 2 hop, all path
            # 3 hop, pruned path
            {'func': generate_adj_data_from_pruned_paths_ckb, 'args': (output_paths['conncected_kb']['pruned-graph'], output_paths['socialiqa']['paths']['pruned-train'],
                                                                       c_kb, output_paths['socialiqa']['keywords']['train'], output_paths['socialiqa']['graph']['pruned-adj-train'], args.nprocs, args.path_prune_topn, args.max_node_num, args.seed)},
            {'func': generate_adj_data_from_pruned_paths_ckb, 'args': (output_paths['conncected_kb']['pruned-graph'], output_paths['socialiqa']['paths']['pruned-dev'],
                                                                       c_kb, output_paths['socialiqa']['keywords']['dev'], output_paths['socialiqa']['graph']['pruned-adj-dev'], args.nprocs, args.path_prune_topn, args.max_node_num, args.seed)},
            {'func': generate_adj_data_from_pruned_paths_ckb, 'args': (output_paths['conncected_kb']['pruned-graph'], output_paths['socialiqa']['paths']['pruned-test'],
                                                                       c_kb, output_paths['socialiqa']['keywords']['test'], output_paths['socialiqa']['graph']['pruned-adj-test'], args.nprocs, args.path_prune_topn, args.max_node_num, args.seed)},
        
        ],
        'make_word_vocab': [
            {'func': make_word_vocab, 'args': ((output_paths['socialiqa']['statement']['train'],), output_paths['socialiqa']['statement']['vocab'])},
        ],
    }

    for rt in args.run:
        for rt_dic in routines[rt]:
            rt_dic['func'](*rt_dic['args'])

    print('Successfully run {}'.format(' '.join(args.run)))


if __name__ == '__main__':
    main()