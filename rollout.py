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
from math_evaluator import math_postprocess_v2
import logging
from typing import List, Dict, Any
from contextlib import contextmanager
import signal
from tqdm import tqdm
from sglang import function, system, user, assistant, gen, set_default_backend, OpenAI
from transformers import AutoTokenizer
from sglang.lang.chat_template import get_chat_template

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('processing.log'),
        logging.StreamHandler()
    ]
)

system_prompt = "You are a helpful and harmless assistant. You should think step-by-step."
adaptive_prompt = """
Before answering the question, please measure the difficulty of the given question.
If the question is difficult for you, provide a comprehensive and detailed solution to the question by outlining numerous well-defined steps. 
Otherwise, please provide a concise solution to the question and focus only on the key steps needed to reach the answer.
"""
# compress_prompt = "Solve the given question more straightforward and with less thinking."
compress_prompt = "Please provide a concise solution to the question. Focus only on the key steps needed to reach the answer. Avoid unnecessary revisions while ensuring accuracy."
scale_prompt = "Provide a comprehensive and detailed solution to the question by outlining numerous well-defined steps. After that, provide a concise critical evaluation of the proposed solution. If there are errors, clearly explain them and offer a revised, correct solution, also concisely."
instruction_for_scq = "Select the best answer from the following options. Output only the letter corresponding to the correct answer, enclosed in \\boxed\{\}."
instruction_for_gsm8k = "Answer the question and enclose the final answer in \\boxed\{\}"

# tokenizer = AutoTokenizer.from_pretrained("/cpfs01/user/xingshuhao.dispatch/zyzeng/llm_ddd/models/r1-14b")

@sgl.function
def rollout(s, question, options, answer, n_sample, prompt_mode, max_token_limit):
    if options is not None:
        instruction = instruction_for_scq
        option_text = f"Options: {options}\n"
    else:
        instruction = instruction_for_gsm8k
        option_text = ""
    
    if prompt_mode == 'short':
        aux_prompt = compress_prompt
    elif prompt_mode == 'long':
        aux_prompt = scale_prompt
    else:
        aux_prompt = ""
    
    s += system(system_prompt)
    s += user(f'{instruction}\n{aux_prompt}\nQuestion: {question}\n{option_text}')
    s += sgl.assistant_begin()
    print(s.text())
    forks = s.fork(n_sample)
    forks += sgl.gen('answer', max_tokens=max_token_limit, stop=['<｜end▁of▁sentence｜>', '<|im_end|>'])

    forks.join()

def post_process_batch(states: List[Dict], questions: List[str], solutions: List[str], 
                 answers: List[str], all_difficulty: List[str], 
                 all_candidate_answers: List[str], ids: List[int], n_sample) -> List[Dict]:
    """Process a batch of responses and prepare them for saving."""
    batch_data = []
    for i in range(len(states)):
        item = dict(
            question=questions[i],
            ref_solution=solutions[i],
            ref_answer=answers[i],
            difficulty=all_difficulty[i],
            options=all_candidate_answers[i]
        )
        model_answers, model_solutions = [], []
        if 'answer' in states[i]:
            for text in states[i]['answer']:
                answer = math_postprocess_v2(text)
                model_solutions.append(text)
                model_answers.append(answer)
        else:
            model_solutions.append(None)
            model_answers.append(None)
            
        item['model_answer'] = model_answers
        item['model_solution'] = model_solutions
        item['id'] = ids[i]
        batch_data.append(item)
    return batch_data

def save_batch(batch_data: List[Dict[str, Any]], output_file: str, file_mode: str = 'a'):
    """Simply save a batch of data to file.
    
    Args:
        batch_data: List of dictionaries containing the data to save
        output_file: Path to the output file
        file_mode: File mode ('w' for write, 'a' for append)
    """
    try:
        with open(output_file, file_mode) as f:
            for item in batch_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        logging.info(f"Successfully saved batch of {len(batch_data)} items")
    except Exception as e:
        logging.error(f"Error saving batch: {str(e)}")
        raise

class GracefulExit(Exception):
    pass

@contextmanager
def graceful_exit_handler():
    def signal_handler(signum, frame):
        raise GracefulExit()
    
    # Register the signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        yield
    finally:
        # Restore default handlers
        signal.signal(signal.SIGINT, signal.default_int_handler)
        signal.signal(signal.SIGTERM, signal.default_int_handler)

def main(args):
    loader = DataLoader(dataset=args.dataset, num_examples=args.num_questions, subset=args.subset)
    questions, answers, solutions, all_candidate_answers, all_difficulty = loader.load_data(args.data_dir)
    ids = list(range(len(questions)))
    if not args.overwrite:
        try:
            # Check if file exists and is not empty
            with open(args.output_file, 'r') as f:
                # Read existing data and get processed ids
                processed_ids = set()
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        processed_ids.add(item['id'])
                    except json.JSONDecodeError:
                        continue  # Skip invalid JSON lines
                if processed_ids:  # Only filter if we found processed ids
                    # Create masks for filtering
                    masks = [i not in processed_ids for i in ids]
                    # Filter out already processed data
                    questions = [q for q, mask in zip(questions, masks) if mask]
                    answers = [a for a, mask in zip(answers, masks) if mask]
                    solutions = [s for s, mask in zip(solutions, masks) if mask]
                    all_candidate_answers = [c for c, mask in zip(all_candidate_answers, masks) if mask]
                    all_difficulty = [d for d, mask in zip(all_difficulty, masks) if mask]
                    ids = [i for i, mask in zip(ids, masks) if mask]
        
        except FileNotFoundError:
            # File doesn't exist, will create it when writing
            pass
        except IOError:
            # Handle other potential file reading errors
            print(f"Warning: Could not read {args.output_file}, processing all data")
            pass
    
    if len(questions) == 0:
        print("all data has been processed, do nothng since the input argument is to **NOT OVERWRITE**")
        exit()
    else:
        print("processing {} questions".format(len(questions)))
    
    
    with graceful_exit_handler():
        for i in range(0, len(questions), args.batch_size):
            batch_end = min(i + args.batch_size, len(questions))
            batch_questions = questions[i:batch_end]
            batch_answers = answers[i:batch_end]
            batch_solutions = solutions[i:batch_end]
            batch_candidate_answers = all_candidate_answers[i:batch_end]
            batch_difficulty = all_difficulty[i:batch_end]
            batch_ids = ids[i:batch_end]
            
            # Prepare batch arguments
            arguments = [
                {
                    'question': q,
                    'n_sample': args.n_sample,
                    'options': o,
                    'answer': a,
                    'prompt_mode': args.prompt_mode,
                    'max_token_limit': args.max_token_limit
                }
                for q, a, o in zip(batch_questions, batch_answers, batch_candidate_answers)
            ]
            
            try:
                # Process batch
                states = rollout.run_batch(
                    arguments,
                    temperature=0.7,
                    num_threads=args.parallel,
                    progress_bar=True
                )
                
                # Prepare and save batch data
                batch_data = post_process_batch(
                    states, batch_questions, batch_solutions,
                    batch_answers, batch_difficulty,
                    batch_candidate_answers, batch_ids, args.n_sample
                )
                
                save_batch(
                    batch_data,
                    args.output_file,
                    'w' if args.overwrite and i == 0 else 'a'
                )
                
                logging.info(f"Completed batch {i//args.batch_size + 1}/{(len(questions) + args.batch_size - 1)//args.batch_size}")
                
            except GracefulExit:
                logging.info("Received interrupt signal, saving current batch and exiting...")
                raise
            except Exception as e:
                logging.error(f"Error processing batch {i//args.batch_size + 1}: {str(e)}")
                raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='scq', choices=['scq', 'gsm8k', 'math', 'numia', 'aime', 'omini', 'gpqa'])
    parser.add_argument("--subset", type=str, default='test')
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--n_sample", type=int, default=1)
    parser.add_argument("--num_questions", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--prompt_mode", type=str, choices=['normal', 'short', 'adaptive', 'long'], default='normal')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument("--api", action="store_true")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--model_class", type=str, choices=['qwen', 'deepseek-v3'])
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--max_token_limit", type=int, default=32000)

    try:
        args = add_common_sglang_args_and_parse(parser)
        if args.model:
            backend = sgl.Runtime(model_path=args.model, dp_size=args.dp)
            # if args.model_class is None:
            #     if 'qwq' in args.output_file.lower():
            #         args.model_class = 'qwen'
            #     else:
            #         args.model_class = 'deepseek-v3'
            backend.endpoint.chat_template = get_chat_template(args.model_class)
            set_default_backend(backend)
        elif args.api:
            set_default_backend(OpenAI("deepseek-reasoner"))
        else:
            backend = select_sglang_backend(args)
            # if args.model_class is None:
            #     if 'qwq' in args.output_file.lower():
            #         args.model_class = 'qwen'
            #     else:
            #         args.model_class = 'deepseek-v3'
            backend.chat_template = get_chat_template(args.model_class)
            set_default_backend(backend)
        main(args)
    except GracefulExit:
        logging.info("Program terminated gracefully")
    except Exception as e:
        logging.error(f"Program terminated with error: {str(e)}")
        raise