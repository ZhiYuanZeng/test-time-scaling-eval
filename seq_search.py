import sglang as sgl
import json
import os
from sglang.api import set_default_backend
from sglang.test.test_utils import (
    add_common_sglang_args_and_parse,
    select_sglang_backend,
)
import argparse
import re
from utils import DataLoader, parse_answer
from math_evaluator import math_postprocess_v2, loose_equal
import logging
from typing import List, Dict, Any
from contextlib import contextmanager
import signal
from tqdm import tqdm
import random
from copy import deepcopy
from collections import Counter
from eval import get_sorted_and_transposed_nested_strings, eval_acc
from transformers import AutoTokenizer
from sglang import function, system, user, assistant, gen, set_default_backend, OpenAI
from sglang.lang.chat_template import get_chat_template

system_prompt = "You are a helpful and harmless assistant. You should think step-by-step."
instruction_for_scq = "Select the best answer from the following options. Output only the letter corresponding to the correct answer, enclosed in \\boxed\{\}."
instruction_for_gsm8k = "Answer the question and enclose the final answer in \\boxed\{\}."

def backtract(solution):
    if '</think>' in solution:
        solution = solution.split('</think>')[0]
    split_solutions = solution.split("\n\n")
    if len(split_solutions) > 2 and 'final answer' in split_solutions[-2].lower():
        return '\n\n'.join(split_solutions[:-2]) + '\n\n'
    else:
        return '\n\n'.join(split_solutions[:-1]) + '\n\n'

@sgl.function
def sgl_search(s, question,  options, answer, solution, n_sample):
    if options is not None:
        instruction = instruction_for_scq
        option_text = f"Options: {options}\n"
    else:
        instruction = instruction_for_gsm8k
        option_text = ""
    
    # s += f"<im_start>{system_prompt}<im_end><|im_start|>user\n{instruction}\nQuestion: {question}\n{option_text}<|im_end|><|im_start|>assistant\n"
    # is_solved = False
    # assert math_postprocess_v2(solution) != answer
    backtraced_solution = backtract(solution)
    s += system(system_prompt)
    s += user(instruction + 'Question: ' + question)
    s += sgl.assistant_begin()
    s += backtraced_solution
    forks = s.fork(n_sample)
    forks += sgl.gen('choices', choices=['Wait', 'Alternatively'])
    forks += sgl.gen('revision', max_tokens=4096, stop=['<｜end▁of▁sentence｜>', '<|im_end|>'])

    forks.join()

def remove_indices(lst, indices):
    """
    从列表 lst 中删除指定索引 indices 的元素。
    :param lst: 原始列表
    :param indices: 需要删除的索引列表
    :return: 删除后的新列表
    """
    return [item for index, item in enumerate(lst) if index not in indices]

def search(arguments, epoch):
    data_to_save = deepcopy(arguments)
    for d in data_to_save:
        d['search_solution'] = []

    depth = 0
    while depth < epoch:
        print(f'Start running epoch {depth}')
        states = sgl_search.run_batch(
            arguments,
            temperature=0.7,
            num_threads=args.parallel,
            progress_bar=True
        )
        # print(len(arguments), len(states))
        for i in range(len(states)):
            if 'choices' in states[i] and 'revision' in states[i]: # revision can fail because of input is too long
                connection_words = states[i]['choices']
                revisions = states[i]['revision']
                assert len(revisions) == 1
                r, c =  revisions[0], connection_words[0]
                new_solution = backtract(arguments[i]['solution']) + c + r
                arguments[i]['solution'] = new_solution
                data_to_save[i]['search_solution'].append(new_solution)
            else:
                if depth > 0:
                    data_to_save[i]['search_solution'].append(data_to_save[i]['search_solution'][depth-1])
                else:
                    data_to_save[i]['search_solution'].append(arguments[i]['solution'])
        depth += 1
        # arguments = remove_indices(arguments, solved_indices)
    return data_to_save


def search_for_test(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    data = []
    with open(args.input_file, 'r') as f:
        for l in f:
            data.append(json.loads(l))
    if args.num_questions > 0:
        random.seed(42)
        data = random.sample(data, args.num_questions)
    questions = [d['question'] for d in data]
    options = [d.get('options', None) for d in data]
    ref_answers = [d['ref_answer'] for d in data]
    model_solutions = [d['model_solution'] for d in data] # a list of list
    model_solutions, sort_indices = get_sorted_and_transposed_nested_strings(model_solutions, tokenizer=tokenizer, transpose=False)
    transposed_model_solutions = list(map(list, zip(*model_solutions)))
    for i,solutions in enumerate(transposed_model_solutions):
        print('avg len of group {}: {}'.format(i, sum([len(tokenizer.encode(s)) for s in solutions]) / len(solutions)))
    model_answers = [[math_postprocess_v2(s) for s in ms] for ms in model_solutions] # a list of list
    
    arguments = []
    num_groups = len(model_solutions[0])
    correct_counter = [0] * num_groups
    total_counter = [0] * num_groups
    coverage_counter = [0 for _ in questions]

    for q_id in range(len(questions)):
        ref_ans = ref_answers[q_id]
        for i,s in enumerate(model_solutions[q_id]):
            model_ans = math_postprocess_v2(s)
            arguments.append({
                'question': questions[q_id], 
                'n_sample': args.n_sample, 
                'options': options[q_id], 
                'answer':ref_answers[q_id], 
                'solution': s})
            if loose_equal(ref_ans, model_ans):
                correct_counter[i] += 1
                coverage_counter[i] = 1
            total_counter[i] += 1
    for j in range(num_groups):
        print(f"the accuracy of group {j}: {correct_counter[j] / total_counter[j]}")
    print('total acc: ', sum(correct_counter) / sum(total_counter))
    print(f"parallel search coverage: {sum(coverage_counter) / len(coverage_counter)}")
    
    if args.resume and os.path.exists(args.search_file):
        with open(args.search_file, 'r') as f:
            data_to_save = [json.loads(l) for l in f]
        assert len(arguments) == len(data_to_save)
        for a, d in zip(arguments, data_to_save):
            a['solution'] = d['search_solution'][-1]
        finished_epochs = len(data_to_save[0]['search_solution'])
        epoch_to_continue = args.epoch - finished_epochs
        print('loading search solitions of {} epochs, there are {} epochs to continue'.format(finished_epochs, epoch_to_continue))
    else:
        data_to_save = []
        epoch_to_continue = args.epoch
        print('search from initial solution, run {} epochs'.format(args.epoch))
    
    new_data_to_save = search(arguments, epoch_to_continue)
    if len(data_to_save) > 0:
        assert len(data_to_save) == len(new_data_to_save)
        for d, nd in zip(data_to_save, new_data_to_save):
            d['search_solution'].extend(nd['search_solution'])
    else:
        data_to_save = new_data_to_save

    with open(args.search_file, 'w') as f:
        for d in data_to_save:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')
    # probing
    original_answers = [math_postprocess_v2(d['solution']) for d in data_to_save]
    ref_answers = [a['answer'] for a in arguments]

    coverage = [loose_equal(o, r) for o, r in zip(original_answers, ref_answers)]
    # 初始化一个字典来存储每组的统计结果
    group_stats = {group: [] for group in range(num_groups)}
    
    print('=' * 50)
    print('Overall Statistics:')
    total_stick_to_old_answer = 0
    for depth in range(args.epoch):
        search_answers = [math_postprocess_v2(d['search_solution'][depth]) for d in data_to_save]
        search_lens = [len(tokenizer.encode(d['search_solution'][depth])) for d in data_to_save]

        # 初始化每组的计数器
        search_from_wrong_count = [0] * num_groups
        search_from_wrong_to_correct_count = [0] * num_groups
        search_from_correct_count = [0] * num_groups
        search_from_correct_to_wrong_count = [0] * num_groups

        search_from_init_correct_count = [0] * num_groups
        search_from_init_correct_to_wrong_count = [0] * num_groups

        search_from_init_wrong_count = [0] * num_groups
        search_from_init_wrong_to_correct_count = [0] * num_groups
        
        search_correct_count = [0] * num_groups
        search_len_count = [0] * num_groups

        # 初始化总体的计数器
        total_search_from_wrong_count = 0
        total_search_from_wrong_to_correct_count = 0
        total_search_from_correct_count = 0
        total_search_from_correct_to_wrong_count = 0

        total_search_from_init_correct_count = 0
        total_search_from_init_correct_to_wrong_count = 0

        total_search_from_init_wrong_count = 0
        total_search_from_init_wrong_to_correct_count = 0
        total_search_correct_count = 0
        total_len_count = 0
        total_wait_correct_count = 0
        total_wait_count = 0
        total_alternatively_correct_count = 0
        total_alternative_count = 0
        
        for i in range(len(search_answers)):
            group_index = i % num_groups  # 确定当前索引属于哪一组
            search_len_count[group_index] += search_lens[i]
            total_len_count += search_lens[i]

            if depth == 0:
                from_correct = loose_equal(original_answers[i], ref_answers[i])
            else:
                from_correct = loose_equal(math_postprocess_v2(data_to_save[i]['search_solution'][depth-1]), ref_answers[i])
            
            if from_correct:
                search_from_correct_count[group_index] += 1
                total_search_from_correct_count += 1
            else:
                search_from_wrong_count[group_index] += 1
                total_search_from_wrong_count += 1

            to_correct = loose_equal(search_answers[i], ref_answers[i])
            coverage[i] = (coverage[i] or to_correct)
            if to_correct:
                search_correct_count[group_index] += 1
                total_search_correct_count += 1

            if from_correct and (not to_correct):
                search_from_correct_to_wrong_count[group_index] += 1
                total_search_from_correct_to_wrong_count += 1
            if (not from_correct) and to_correct:
                search_from_wrong_to_correct_count[group_index] += 1
                total_search_from_wrong_to_correct_count += 1

            if not loose_equal(original_answers[i], ref_answers[i]):
                search_from_init_wrong_count[group_index] += 1
                total_search_from_init_wrong_count += 1
                if to_correct:
                    search_from_init_wrong_to_correct_count[group_index] += 1
                    total_search_from_init_wrong_to_correct_count += 1

            if loose_equal(original_answers[i], ref_answers[i]):
                search_from_init_correct_count[group_index] += 1
                total_search_from_init_correct_count += 1
                if not to_correct:
                    search_from_init_correct_to_wrong_count[group_index] += 1
                    total_search_from_init_correct_to_wrong_count += 1
            
            if loose_equal(original_answers[i], search_answers[i]) and not loose_equal(original_answers[i], ref_answers[i]):
                total_stick_to_old_answer += 1

        # 将每组的统计结果存储到字典中
        for group in range(num_groups):
            group_size = len(search_answers) // num_groups
            group_stats[group].append({
                "epoch": depth,
                'len_after_search': search_len_count[group] // group_size,
                "acc_after_search": search_correct_count[group] / group_size,
                "search_from_wrong_to_correct_ratio": search_from_wrong_to_correct_count[group] / search_from_wrong_count[group] if search_from_wrong_count[group] != 0 else 0,
                "search_from_correct_to_wrong_ratio": search_from_correct_to_wrong_count[group] / search_from_correct_count[group] if search_from_correct_count[group] != 0 else 0,
                "search_from_init_wrong_to_correct_ratio": search_from_init_wrong_to_correct_count[group] / search_from_init_wrong_count[group] if search_from_init_wrong_count[group] != 0 else 0,
                "search_from_init_correct_to_wrong_ratio": search_from_init_correct_to_wrong_count[group] / search_from_init_correct_count[group] if search_from_init_correct_count[group] != 0 else 0,
                "coverage": sum(coverage[group::num_groups]) / group_size
            })

        # 打印总体的统计结果
        # print(f"Overall Results for epoch {depth}:")
        print(f"epoch: {depth}, acc after search: {total_search_correct_count / len(search_answers)}, \
            search from wrong to correct ratio: {total_search_from_wrong_to_correct_count / total_search_from_wrong_count if total_search_from_wrong_count != 0 else 0}, \
            search from correct to wrong ratio: {total_search_from_correct_to_wrong_count / total_search_from_correct_count if total_search_from_correct_count != 0 else 0}, \
            search from init wrong to correct ratio: {total_search_from_init_wrong_to_correct_count / total_search_from_init_wrong_count if total_search_from_init_wrong_count != 0 else 0}, \
            search from init correct to wrong ratio: {total_search_from_init_correct_to_wrong_count / total_search_from_init_correct_count if total_search_from_init_correct_count != 0 else 0}, \
            coverage: {sum(coverage) / len(coverage)}, \
            len after search: {total_len_count / len(search_answers)}")
    print('=' * 50)
    # 打印每组的统计结果（按组整合）
    for group in range(num_groups):
        print(f"Group {group + 1} Statistics:")
        for stats in group_stats[group]:
            print(f"epoch: {stats['epoch']}, acc after search: {stats['acc_after_search']}, \
                search from wrong to correct ratio: {stats['search_from_wrong_to_correct_ratio']}, \
                search from correct to wrong ratio: {stats['search_from_correct_to_wrong_ratio']}, \
                search from init wrong to correct ratio: {stats['search_from_init_wrong_to_correct_ratio']}, \
                search from init correct to wrong ratio: {stats['search_from_init_correct_to_wrong_ratio']}, \
                coverage: {stats['coverage']}, \
                len after search: {stats['len_after_search']}")
        print('-' * 50)
        print("\n")  # 添加空行分隔不同组
    
    print('ratio of stick to old answer: {}'.format(total_stick_to_old_answer / (total_search_from_init_wrong_count * args.epoch) ))

def main(args):
    search_for_test(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_questions", type=int, default=-1)
    parser.add_argument("--search_file", type=str, default=None)
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--n_sample", type=int, default=1)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--model_class", type=str,  default=None)
    parser.add_argument("--tokenizer_path", type=str, default=None)

    args = add_common_sglang_args_and_parse(parser)
    try:
        backend = select_sglang_backend(args)
        backend.chat_template = get_chat_template(args.model_class)
        sgl.set_default_backend(backend)
    except Exception as e:
        print(e)
        # raise e
    main(args)
