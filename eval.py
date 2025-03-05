import argparse
import json
from transformers import AutoTokenizer
from collections import defaultdict
from utils import parse_answer
from math_evaluator import is_equiv, math_postprocess_v2, loose_equal
from qwen_math_evaluator import math_equal
import re
import random
import os
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false" 

# ... rest of your code

def calculate_average_by_class(data, value_key='value', class_key='class'):
    """
    根据指定的 class 键对数据列表中的 value 进行分类，并计算每个 class 下 value 的平均值。

    Args:
        data: 字典列表，每个字典应包含 value_key 和 class_key。
        value_key: value 的键名，默认为 'value'。
        class_key: class 的键名，默认为 'class'。

    Returns:
        一个字典，键为 class，值为对应的平均值。如果某个 class 没有值，则值为 0。
        如果输入数据格式不正确，则返回 None。
    """
    if not isinstance(data, list):
        print("Error: Input data must be a list.")
        return None

    class_values = defaultdict(list)
    for item in data:
        if not isinstance(item, dict):
            print("Error: Each item in data must be a dictionary.")
            return None
        if value_key not in item or class_key not in item:
            print(f"Error: Each dictionary must contain keys '{value_key}' and '{class_key}'.")
            return None
        try:
            value = item[value_key]
            # 尝试将 value 转换为数值类型，如果转换失败则跳过该项
            float(value)
        except (ValueError, TypeError):
            print(f"Warning: Value '{value}' for class '{item[class_key]}' is not a number and will be ignored.")
            continue
        class_values[item[class_key]].append(value)

    class_averages = {}
    for cls, values in class_values.items():
        if values:
            class_averages[cls] = sum(values) / len(values)
        else:
            class_averages[cls] = 0
    return class_averages

def eval_len(alL_model_solutions, all_difficulty, tokenizer):
    results = []
    for solutions, difficulty in zip(alL_model_solutions, all_difficulty):
        if isinstance(solutions, str):
            avg_len = len(tokenizer.encode(solutions))
        else:
            try:
                avg_len = sum([len(tokenizer.encode(s)) for s in solutions]) / len(solutions)
            except Exception as e:
                print(e)
                continue
        results.append({'len': avg_len, 'difficulty': difficulty})
    overall_len = sum([r['len'] for r in results]) / len(results)
    avg_results = calculate_average_by_class(results, value_key='len', class_key='difficulty')
    avg_results['overall'] = overall_len
    return avg_results

def eval_acc(all_ref_ans, all_model_ans, all_difficulty=None, ignore_parsing_error=False, eval_func=loose_equal):
    results = []
    if all_difficulty is None:
        all_difficulty = [None] * len(all_ref_ans)
    
    for ref_ans, model_ans, difficulty in zip(all_ref_ans, all_model_ans, all_difficulty):
        if ignore_parsing_error:
            model_ans = [ms for ms in model_ans if ms is not None]
        if len(model_ans) > 0:
            avg = sum([eval_func(ms, ref_ans) for ms in model_ans]) / len(model_ans)
            results.append({"correct": avg, "difficulty": difficulty})
        # for ms in model_solution:
        #     if not is_equiv(ref_ans, math_postprocess_v2(ms)):
        #         print('ref answer: {}, model answer: {}'.format(ref_ans, math_postprocess_v2(ms)))

    overall_acc = sum([r['correct'] for r in results]) / len(results)
    avg_results = calculate_average_by_class(results, value_key='correct', class_key='difficulty')
    avg_results['overall'] = overall_acc
    return avg_results

def argsort(lst):
    return sorted(range(len(lst)), key=lambda i: lst[i])

def get_sorted_and_transposed_nested_strings(list_of_lists, tokenizer, transpose=True):
    """
    从嵌套字符串列表中获取每个子列表的字符串，按照 tokenizer 编码后的长度从小到大排序，
    然后将结果转置。

    参数:
        list_of_lists (List[List[str]]): 嵌套的字符串列表
        tokenizer: 用于计算字符串长度的 tokenizer 函数

    返回:
        List[List[str]]: 转置后的嵌套列表，每个子列表中的字符串按照 tokenizer 编码后的长度从小到大排序
        如果输入列表为空返回 []
        如果任何子列表为空，对应位置返回 []

    示例:
        输入: [["a", "bb", "ccc"], ["dd", "eeeee"]]
        输出: [["a", "dd"], ["bb", "eeeee"], ["ccc"]]
    """
    if not list_of_lists:
        return [], []
    all_sort_indices = []
    # 先按照 tokenizer 编码后的长度排序
    sorted_nested_strings = []
    for sublist in list_of_lists:
        if not sublist:
            sorted_nested_strings.append([])
            continue

        # 创建带索引的长度列表，保持原始顺序
        length_with_index = [(len(tokenizer.encode(s)), i, s) for i, s in enumerate(sublist)]
        length_with_index.sort()  # 按长度排序
        sort_indices = argsort([len(tokenizer.encode(s)) for i, s in enumerate(sublist)])
        all_sort_indices.append(sort_indices)

        # 按照 tokenizer 编码后的长度从小到大排序
        sorted_sublist = [s for _, _, s in length_with_index]
        sorted_nested_strings.append(sorted_sublist)
    if not transpose:
        return sorted_nested_strings, all_sort_indices
    # 使用 zip 转置嵌套列表
    # 注意：如果子列表长度不一致，zip 会截断到最短的子列表长度
    # 使用 itertools.zip_longest 可以避免截断，并用 None 填充
    from itertools import zip_longest

    transposed_list = list(zip_longest(*sorted_nested_strings, fillvalue=None))

    # 将元组转换为列表，并移除填充的 None
    transposed_list = [
        [item for item in sublist if item is not None] for sublist in transposed_list
    ]

    return transposed_list, all_sort_indices

def filter_wt_indices(lst, indices):
    return [x for i,x in enumerate(lst) if i in indices]

def count_optimal_solution_len(model_solutions, ref_answers, tokenizer, eval_func=loose_equal):
    correct_len_counter = 0
    incorrect_len_counter = 0
    random_len_counter = 0
    total_counter = 0
    for ms, ref in zip(model_solutions, ref_answers):
        correct_solutions = [s for s in ms if eval_func(math_postprocess_v2(s), ref)]
        incorrect_solutions = [s for s in ms if s not in correct_solutions]
        if len(correct_solutions) > 0 and len(incorrect_solutions) > 0:
            correct_len_counter += sum([len(tokenizer.encode(cs)) for cs in correct_solutions]) / len(correct_solutions)
            incorrect_len_counter += sum([len(tokenizer.encode(ics)) for ics in incorrect_solutions]) / len(incorrect_solutions)   
            total_counter += 1
    return correct_len_counter / total_counter, incorrect_len_counter / total_counter

def calculate_averages(lst):
    n = len(lst)
    
    # 前一半的平均值
    if n % 2 == 0:
        first_half = lst[:n//2]
    else:
        first_half = lst[:n//2]
    first_half_avg = sum(first_half) / len(first_half)
    
    # 中间值
    if n % 2 == 1:
        middle_value = lst[n//2]
    else:
        middle_value = (lst[n//2 - 1] + lst[n//2]) / 2
    
    # 后一半的平均值
    if n % 2 == 0:
        second_half = lst[n//2:]
    else:
        second_half = lst[n//2 + 1:]
    second_half_avg = sum(second_half) / len(second_half)
    
    return first_half_avg, middle_value, second_half_avg

def count_correct_solutions(solutions, ref_answers, eval_func, tokenizer):
    num_tokens = 0
    num_solutions = 0
    for s, a in zip(solutions, ref_answers):
        if eval_func(math_postprocess_v2(s), a):
            num_tokens += len(tokenizer.encode(s))
            num_solutions += 1
    return num_solutions, num_tokens


def main(args):
    input_file = args.input_file
    eval_func = loose_equal if args.eval_func == 'loose' else is_equiv
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    with open(input_file, 'r') as f:
        data = [json.loads(l) for l in f]
        print('number of examples: ', len(data))
        data = [d for d in data if all([s is not None for s in d['model_solution']])]
        print('after filtering None, number of examples: ', len(data))

        all_ref_ans = [d['ref_answer'] for d in data]
        all_model_solutions = [d['model_solution'] if isinstance(d['model_solution'], list) else [d['model_solution'],] for d in data]
        all_model_ans = [[math_postprocess_v2(s) for s in ms] for ms in all_model_solutions]
        
        parsing_erro_count = 0
        for ms in all_model_solutions:
            for s in ms:
                if 'boxed' not in s:
                    parsing_erro_count += 1
        print('parsing_erro_rate: {} %'.format(100 * parsing_erro_count / (len(all_model_solutions) * len(all_model_solutions[0]))))
        all_difficulty = [d['difficulty'] for d in data]

    avg_acc = eval_acc(all_ref_ans, all_model_ans, all_difficulty, ignore_parsing_error=False, eval_func=eval_func)
    avg_len = eval_len(all_model_solutions, all_difficulty, tokenizer)
    print('average acc:' + '-'*20)
    print(avg_acc)
    print('average len' + '-'*20)
    print(avg_len)

    correct_len, in_correct_len = count_optimal_solution_len(all_model_solutions, all_ref_ans, tokenizer, )
    print('"correct": {}, "incorrect": {}'.format(correct_len, in_correct_len))

    len_sorted_solutions = get_sorted_and_transposed_nested_strings(all_model_solutions, tokenizer)[0]

    all_num_correct_solution, all_num_correct_tokens = [], []
    for solutions in len_sorted_solutions:
        num_correct_solution, num_correct_tokens = count_correct_solutions(solutions, all_ref_ans, eval_func, tokenizer)
        all_num_correct_solution.append(num_correct_solution)
        all_num_correct_tokens.append(num_correct_tokens)
    print('correct tokens distribution in all groups:', (np.array(all_num_correct_tokens) / sum(all_num_correct_tokens)).tolist())
    print('correct solutions distribution in all groups:', (np.array(all_num_correct_solution) / sum(all_num_correct_solution)).tolist())

    # keep_indices = [i for i in range(len(data)) if all(['boxed' in s[i] for s in len_sorted_solutions])]
    # len_sorted_solutions = [filter_wt_indices(s, keep_indices) for s in len_sorted_solutions]
    # all_difficulty = filter_wt_indices(all_difficulty, keep_indices)
    # all_ref_ans = filter_wt_indices(all_ref_ans, keep_indices)
    len_sorted_ans = [[[math_postprocess_v2(s),] for s in solutions] for solutions in len_sorted_solutions]
    len_sorted_acc = [eval_acc(all_ref_ans, ans, all_difficulty, ignore_parsing_error=False, eval_func=eval_func) for ans in len_sorted_ans]

    first_half_avg, middle_value, second_half_avg = calculate_averages([item['overall'] for item in len_sorted_acc])
    print(f'short acc: {first_half_avg}, middle: {middle_value}, long acc: {second_half_avg}')

    len_sorted_len = [eval_len(solutions, all_difficulty, tokenizer) for solutions in len_sorted_solutions]
    print(len_sorted_acc)
    print('-' * 50)
    print(len_sorted_len)


    wait_count = [sum([s.lower().count('wait') for s in solutions]) / len(solutions) for solutions in len_sorted_solutions]
    alternative_count = [sum([s.lower().count('alternatively') for s in solutions]) / len(solutions) for solutions in len_sorted_solutions]
    print(f'number of occur of wait: {wait_count}')
    print(f'number of occur of alternative: {alternative_count}')
    
    solutions_to_debug = [(q, s) for q, ms in zip([d['question'] for d in data], all_model_solutions) for s in ms if 'boxed' not in s]
    # with open('/tmp/debug.txt','w') as f:
    #     for q,s in solutions_to_debug:
    #         f.write(q + '\n' + s + '#' * 50 + '\n' * 5)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--eval_func", type=str, choices=['opencompass', 'loose'], default='loose')
    parser.add_argument("--tokenizer_path", type=str, default='loose')

    args = parser.parse_args()
    main(args)