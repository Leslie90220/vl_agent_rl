import json
from datasets import Dataset
import os
import datasets
import argparse
from tqdm import tqdm

USER_PROMPT = '''
Question (IMPORTANT - This is what you need to answer)
{question}

## Role
You are a question-answering agent in long-document understanding. Your task is to solve the above question by reasoning step by step, and when needed, use the provided tools (search, crop, ocr) to gather missing information.

## Thinking Principles
- You MUST conduct reasoning inside <think> and </think> first every time you get new information.
- If the contextual content does not contain enough information, you should call the **search** tool to retrieve new relevant pages. You may call search in multiple rounds until you find enough information to answer the question.
- If you need more fine-grained page details during the answering process, you MUST call the **crop** tool to crop the page and obtain a more detailed slice.
- If you find that the full page image or cropped image content cannot be directly used, or you need to better understand figures, tables, or formulas, you MUST call the **ocr** tool to extract text from the image.
- If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>.

## When to Stop Searching and Provide Answer
- **IMPORTANT**: You should provide an answer when You have found the specific information needed to answer the question.
- **DO NOT** keep searching indefinitely. If you have relevant information, provide your best answer.
- **When in doubt after multiple searches**, provide an answer based on what you have found rather than continuing to search.

## Tools
### search
- Description: Use this tool to search for relevant pages when the current pages are insufficient to answer
- Example:
<tools_call>
{"type":"search","arguments":{"query":"your search query"}}
</tools_call>

### crop
- Description: Use this tool to crop a section of a page when you need more detailed information
- Example:
<tools_call>
{"type":"crop","arguments":{"image_id":"image_01","region":[x1,y1,x2,y2]}}
</tools_call>
- Note: image_id is the ID of the image (e.g., image_01, image_02); region is [x1,y1,x2,y2] pixel coordinates.

### ocr
- Description: Use this tool to extract text from tables, charts, or formulas in an image.
- Example:
<tools_call>
{"type":"ocr","arguments":{"image_id":"image_03","region":[x1,y1,x2,y2]}}
</tools_call>

## Strict Output Requirement
Every response MUST start with a <think> section. 
- If you need tools: <think>...</think><tools_call>...</tools_call>
- If you have the answer: <think>...</think><answer>...</answer>
- Note:Failure to include <think> is a violation of your core instructions,which should be STRICTLY FORBIDDEN.

## Example Usage
Query: Find the dates of 28th IOC Assembly at UNESCO Paris.
<think>I need to find information about dates of 28th IOC Assembly at UNESCO Paris. Let me search for relevant pages.</think>
<tools_call>
{"type":"search","arguments":{"query":"dates of 28th IOC Assembly at UNESCO Paris"}}
</tools_call>
'''


# all_examples = [example for example in all_examples if example['query'] not in sft_questions]
def convert_dataset(USER_PROMPT,file_list,file_source_list,output_name):
    all_examples = []
    for file_name, source_type in zip(file_list, file_source_list):
        with open(file_name, "r") as f:
            file_data = json.load(f)
            data_list = file_data["examples"]
            for item in data_list:
                item['source'] = source_type
            all_examples.extend(data_list)

    for example in all_examples:
        if example['source'] == 'vidoseek':
            example['reason_type'] = example['meta_info']['query_type']
            example['content_type'] = example['meta_info']['source_type']
        elif example['source'] == 'slidevqa_test':
            query_type = example['meta_info']['query_type']
            if 'Multi-Hop' in query_type:
                example['reason_type'] = 'MultiHop'
            elif 'Single-Hop' in query_type:
                example['reason_type'] = 'SingleHop'
            if 'Non-Span' in query_type:
                example['content_type'] = 'NonSpan'
            elif 'Single-Span' in query_type:
                example['content_type'] = 'SingleSpan'
            elif 'Multi-Span' in query_type:
                example['content_type'] = 'MultiSpan'
        elif example['source'] == 'mmlongdoc':
            example['content_type'] = '####'.join(example['meta_info']['source_type'])
            example['reason_type'] = example['meta_info']['doc_type']
        else:
            example['content_type'] = 'Nan'
            example['reason_type'] = 'Nan'

    dataset = Dataset.from_dict({
        "id": [str(example["uid"]) for example in all_examples],
        "problem": [example["query"] for example in all_examples],
        "prompt": [USER_PROMPT.replace('{question}',example["query"]) for example in all_examples],
        "answer": [example["reference_answer"] for example in all_examples],
        "file_name": [example["meta_info"]["file_name"] for example in all_examples],
        "reference_page": [example["meta_info"]["reference_page"] for example in all_examples],
        "data_source_type": [example["source"] for example in all_examples],
        "query_content_type": [example["content_type"] for example in all_examples],
        "query_reason_type": [example["reason_type"] for example in all_examples]
    })

    def make_map_fn_test(split):
        def process_fn(example, idx):
            prompt = example.pop('prompt')
            answer = example.pop('answer')
            problem = example.pop('problem')
            data_source = example.pop('data_source_type')
            reference_page = example.pop('reference_page')
            file_name = example.pop('file_name')
            # images = example.pop('images')
            query_content_type = example.pop('query_content_type')
            query_reason_type = example.pop('query_reason_type')

            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": prompt,
                }],
                # "images": images,
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'answer': answer,
                    "question": problem,
                    "content_type": query_content_type,
                    "reason_type": query_reason_type,
                    "file_name": file_name,
                    "reference_page": reference_page
                }
            }
            return data
        return process_fn

    test_dataset = dataset.map(function=make_map_fn_test('test'), with_indices=True, num_proc=8)

    test_dataset.to_parquet(f'./data/{output_name}.parquet')


if __name__ == '__main__':
    # 转换 MMLongDoc 数据集
    # convert_dataset(
    #     USER_PROMPT,
    #     ['./data/MMLongDoc.json'],
    #     ['mmlongdoc'],
    #     'mmlongdoc_test'
    # )
    
    # 转换 vidoseek 数据集
    convert_dataset(
        USER_PROMPT,
        ['./data/vidoseek.json'],
        ['vidoseek_bak'],
        'vidoseek_bak_train'
    )