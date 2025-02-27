import random
import os
import json
import re
from math_evaluator import math_postprocess_v2
import csv
from colorama import Fore, Style, init

class DataLoader():
    
    def __init__(self, dataset, subset='train', num_examples=-1, seed=42):
        self.subset = subset
        self.num_examples = num_examples
        self.seed = seed
        self.dataset = dataset

    def load_data(self, data_dir=None):
        if data_dir is not None:
            kwargs = {'data_dir': data_dir}
        else:
            kwargs = {}
        if self.dataset == 'scq':
            return self.load_scq_data(**kwargs)
        elif self.dataset == 'gsm8k':
            return self.load_gsm8k_data(**kwargs)
        elif self.dataset == 'math':
            return self.load_math_data(**kwargs)
        elif self.dataset == 'numia':
            return self.load_numia_data(**kwargs)
        elif self.dataset == 'aime':
            return self.load_aime_data(**kwargs)
        elif self.dataset == 'omini':
            return self.load_omini_data(**kwargs)
        elif self.dataset == 'gpqa':
            return self.load_gpqa_data(**kwargs)
        else:
            raise NotImplementedError

    def load_scq_data(self, data_dir="data/scq5k/TAL-SCQ5K-EN", ):
        with open(os.path.join(data_dir, f"{self.subset}.jsonl"), 'r') as f:
            data = [json.loads(line) for line in f]

        if self.num_examples != -1:
            random.seed(self.seed)
            data = random.sample(data, self.num_examples)
        
        questions, answers, solutions, all_candidate_answers, all_difficulty = [],[],[],[],[]
        for d in data:
            try:
                questions.append(d['problem'])
                candidate_answers = ["{}:{}".format(candidate[0]["aoVal"], candidate[0]["content"]).strip() for candidate in d['answer_option_list']]
                candidate_answers = '\n'.join(candidate_answers)
                answers.append(d['answer_value'])
                solutions.append(d['answer_analysis'])
                all_candidate_answers.append(candidate_answers)
                all_difficulty.append(d['difficulty'])
                assert d['qtype'] == 'single_choice'
            except Exception as e:
                print(d)
                print(e)
                print('Error Loading Data' + '!'*50)
        return questions, answers, solutions, all_candidate_answers, all_difficulty

    def load_gsm8k_data(self, data_dir="data/gsm8k/"):
        with open(os.path.join(data_dir, f"{self.subset}.jsonl"), 'r') as f:
            data = [json.loads(line) for line in f]

        if self.num_examples != -1:
            random.seed(self.seed)
            data = random.sample(data, self.num_examples)

        questions, answers, solutions, all_candidate_answers, all_difficulty = [],[],[],[],[]
        for d in data:
            try:
                questions.append(d['question'])
                solutions.append(d['answer'])
                answer = d['answer'].split('####')[1].strip()
                answers.append(answer)
                all_candidate_answers.append(None)
                all_difficulty.append(None)
            except Exception as e:
                print(d)
                print(e)
                print('Error Loading Data' + '!'*50)
        return questions, answers, solutions, all_candidate_answers, all_difficulty

    def load_math_data(self, data_dir="data/math/"):
        def find_and_read_jsonl_files(directory, subset='train'):
            files_data = []
            
            # 遍历文件夹中的所有文件
            for root, dirs, files in os.walk(directory):
                for file in files:
                    # 查找文件名中包含'subset'并且是.jsonl文件
                    if subset in file and file.endswith('.jsonl'):
                        file_path = os.path.join(root, file)
                        print(f"Processing file: {file_path}")
                        
                        # 读取jsonl文件
                        with open(file_path, 'r', encoding='utf-8') as f:
                            for line in f:
                                try:
                                    data = json.loads(line.strip())
                                    files_data.append(data)
                                except json.JSONDecodeError as e:
                                    print(f"Error decoding JSON in file {file_path}: {e}")
            
            return files_data
        if self.subset == 'train':
            data = find_and_read_jsonl_files(data_dir, subset=self.subset)
        else:
            with open(os.path.join(data_dir, 'math500_test.jsonl'), 'r') as f:
                data = [json.loads(l.strip()) for l in f]
                assert len(data) == 500
        
        if self.num_examples != -1:
            random.seed(self.seed)
            data = random.sample(data, self.num_examples)
            
        questions = [item['problem'] for item in data]
        solutions = [item['solution'] for item in data]
        answers = [parse_answer(item['solution']) for item in data]
        all_difficulty = [item['level'] for item in data]
        all_candidate_answers = [None for item in data]

        return questions, answers, solutions, all_candidate_answers, all_difficulty

    def load_numia_data(self, data_dir="data/numia_10000"):
        with open(os.path.join(data_dir, f"{self.subset}.jsonl"), 'r') as f:
            data = [json.loads(line) for line in f]

        if self.num_examples != -1:
            random.seed(self.seed)
            data = random.sample(data, self.num_examples)

        questions, answers, solutions, all_candidate_answers, all_difficulty = [],[],[],[],[]
        for d in data:
            try:
                questions.append(d['problem'])
                solutions.append(d['solution'])
                answer = math_postprocess_v2(d['solution'])
                answers.append(answer)
                all_candidate_answers.append(None)
                all_difficulty.append(None)
            except Exception as e:
                print(d)
                print(e)
                print('Error Loading Data' + '!'*50)
        return questions, answers, solutions, all_candidate_answers, all_difficulty
    
    def load_aime_data(self, data_dir="data/aime"):
        assert self.subset == 'test'
        with open(os.path.join(data_dir, f"{self.subset}.jsonl"), 'r') as f:
            data = [json.loads(line) for line in f]

        if self.num_examples != -1:
            random.seed(self.seed)
            data = random.sample(data, self.num_examples)

        questions, answers, solutions, all_candidate_answers, all_difficulty = [],[],[],[],[]
        for d in data:
            try:
                questions.append(d['problem'])
                solutions.append(d['solution'])
                answer = d['answer']
                answers.append(answer)
                all_candidate_answers.append(None)
                all_difficulty.append(None)
            except Exception as e:
                print(d)
                print(e)
                print('Error Loading Data' + '!'*50)
        return questions, answers, solutions, all_candidate_answers, all_difficulty
    
    def load_omini_data(self, data_dir="./data/omini-math"):
        assert self.subset == 'test'
        with open(os.path.join(data_dir, f"{self.subset}.jsonl"), 'r') as f:
            data = [json.loads(line) for line in f]

        if self.num_examples != -1:
            random.seed(self.seed)
            data = random.sample(data, self.num_examples)

        questions, answers, solutions, all_candidate_answers, all_difficulty = [],[],[],[],[]
        for d in data:
            try:
                questions.append(d['problem'])
                solutions.append(d['solution'])
                answer = d['answer']
                answers.append(answer)
                all_candidate_answers.append(None)
                all_difficulty.append(None)
            except Exception as e:
                print(d)
                print(e)
                print('Error Loading Data' + '!'*50)
        print('Data Examle:' + '-' * 50)
        print('Question:\n', questions[0])
        print('-' * 30)
        print('ref answer:\n', answers[0])
        print('-' * 30)
        return questions, answers, solutions, all_candidate_answers, all_difficulty
    
    def load_gpqa_data(self, data_dir="./data/gpqa"):
        cnt = 0
        data = []
        name = "gpqa_diamond.csv"
        with open(os.path.join(data_dir, name), 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                if row[7] == 'Question':
                    continue
                cnt = cnt + 1
                question = row[7]
                # 第一个是正确选项
                options = [row[8], row[9], row[10], row[11]]
                shuffle_patterns = ['ABCD', 'BCDA', 'CDAB', 'DABC']  # 更新选项顺序
                c = shuffle_patterns[cnt % 4]
                line = {'question': question}
                ground_truth = options[0]
                for i in range(4):
                    line['ABCD'[i]] = options[ord(c[i]) - ord('A')]
                for i in range(4):
                    if line['ABCD'[i]] == ground_truth:
                        line['answer'] = 'ABCD'[i]
                        break
                data.append(line)
        if self.num_examples != -1:
            random.seed(self.seed)
            data = random.sample(data, self.num_examples)
        
        questions = [d['question'] for d in data]
        answers = [d['answer'] for d in data]
        options = [f"A: {d['A']}\nB: {d['B']}\nC: {d['C']}\nD: {d['D']}" for d in data]
        solutions = [None for d in data]
        all_difficulty = [None for d in data]

        print('Data Examle:' + '-' * 50)
        print('Question:\n', questions[0])
        print('-' * 30)
        print('Options:\n', options[0])
        print('-' * 30)
        print('ref answer:\n', answers[0])
        print('-' * 30)
        return questions, answers, solutions, options, all_difficulty

def load_rollout_data(data_path, num_examples, seed=42):
    with open(data_path, 'r') as f:
        data = [json.loads(line) for line in f]
    if num_examples > 0:
        random.seed(seed)
        data = random.sample(data, num_examples)
    questions = [d['question'] for d in data]
    ref_answers = [d['ref_answer'] for d in data]
    ref_solutions = [d['ref_solution'] for d in data]
    all_difficulty = [d['difficulty'] for d in data]
    all_options = [d.get('options') for d in data]

    model_answers = [d['model_answer'] for d in data]
    model_solutions = [d['model_solution'] for d in data]
    return questions, all_options, ref_answers, ref_solutions, all_difficulty, model_answers, model_solutions

import re

def parse_answer(text):
    """
    Parse LaTeX expressions to extract content within \boxed{} command,
    handling nested braces and complex expressions.
    
    Args:
        text (str): LaTeX text containing \boxed{} expression
        
    Returns:
        str: Content within \boxed{}, or None if not found
    """
    def find_matching_brace(s, start):
        """Helper function to find matching closing brace"""
        count = 1
        i = start
        while i < len(s) and count > 0:
            if s[i] == '{':
                count += 1
            elif s[i] == '}':
                count -= 1
            i += 1
        return i - 1 if count == 0 else -1

    # Find \boxed command
    boxed_match = re.search(r'\\boxed\{', text)
    if not boxed_match:
        return None
    
    # Get start position of content (after opening brace)
    start_pos = boxed_match.end()
    
    # Find matching closing brace
    end_pos = find_matching_brace(text, start_pos)
    if end_pos == -1:
        return None
        
    # Extract content between braces
    return text[start_pos:end_pos]


def highlight_words(text, words_to_highlight, color=Fore.RED):
    """
    高亮显示文本中的某些词，同时保留 LaTeX 公式的完整性
    :param text: 原始文本
    :param words_to_highlight: 需要高亮的词（列表）
    :param color: 高亮颜色（默认红色）
    :return: 高亮后的文本

    # 示例用法
    text = r"这是一个包含 LaTeX 公式的字符串：$\frac{a}{b}$ 和 $\sum_{i=1}^n i^2$，以及一些普通文本。"
    words_to_highlight = ["LaTeX", "文本"]
    highlighted_text = highlight_words(text, words_to_highlight, color=Fore.GREEN)

    print(highlighted_text)
    """
    # 将文本拆分为 LaTeX 公式部分和非公式部分
    parts = []
    in_latex = False
    start = 0

    for i, char in enumerate(text):
        if char == '$':
            if not in_latex:
                # 进入 LaTeX 公式部分
                parts.append((text[start:i], False))  # 非公式部分
                start = i
                in_latex = True
            else:
                # 退出 LaTeX 公式部分
                parts.append((text[start:i + 1], True))  # 公式部分
                start = i + 1
                in_latex = False

    # 添加最后一部分
    if start < len(text):
        parts.append((text[start:], False))

    # 高亮非公式部分中的目标词
    highlighted_parts = []
    for part, is_latex in parts:
        if not is_latex:
            for word in words_to_highlight:
                part = part.replace(word, f"{color}{word}{Style.RESET_ALL}")
        highlighted_parts.append(part)

    # 合并所有部分
    color_str = ''.join(highlighted_parts)
    print(color_str)
    return color_str
    
def load_jsonl(file):
    with open(file, 'r') as f:
        return [json.loads(l) for l in f]