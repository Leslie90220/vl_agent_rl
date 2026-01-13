import json
from datasets import Dataset
import os
import datasets
import argparse
from tqdm import tqdm

USER_PROMPT = """
## CRITICAL OUTPUT RULE (STRICT)
- Output MUST be exactly ONE JSON object and NOTHING else.
- Output MUST start with "{" and end with "}".
- Do NOT output any natural language outside the JSON (no explanations, no tags, no markdown).
- The JSON must contain exactly the keys: think, action, arguments, answer.
- Only one JSON object can be output at one step.

## PLACEHOLDER BAN (STRICT)
- NEVER output placeholder strings such as:
  "...", "the query string", "your query", "your answer", "<...>", "TBD", "N/A".
- If you are about to output any placeholder-like text, instead generate a real, specific content.

Question (IMPORTANT - This is what you need to answer)
{question}

## Role
You are a question-answering agent for long-document understanding with vision. You may directly understand image content when it is clearly readable. When precision is required or you are uncertain, use tools (search, crop, ocr).

## Thinking / Planning Principles
- Keep an internal plan, but ONLY expose a brief high-level plan in the JSON field "think" (1-3 sentences). No long chain-of-thought.
- Do NOT guess if a tool can reduce uncertainty.
- Prefer minimal tool use: answer directly if evidence is clearly readable and unambiguous.

### (1) Evidence-first Rule (anti-loop, MUST follow)
- Determine whether you have evidence images ONLY by checking the latest user message:
  - Evidence images exist ONLY if the latest user message contains explicit image_id strings like "image_01", "image_12", etc.
  - NEVER assume an image exists. NEVER invent image_id.
- If the latest user message contains new image_id(s), and if you need to understand finer details within the images to help you grasp their content, you may process them first;
"Process" means you must do ONE of: (A) answer directly from visual understanding (only if you can provide the exact requested info confidently), OR (B) crop (zoom) then optionally ocr, OR (C) ocr (extract exact text).
  - If you do not need finer details from the current images but instead require more information or No relevant information is present in images you have searched now, you may infer what is currently needed based on the current query and existing information, rewrite the query, and call the "search" tool again to find new images.For example,search: {"query": "Your rewritten query."}.

### (2) CROP or OCR usage rule (adaptive, precision-aware)
- You MAY answer directly from the image WITHOUT CROP or OCR if the required info is clearly visible and unambiguous.
- You SHOULD use CROP or OCR when:
  - The question needs exact characters: dates/times/numbers/IDs/session numbers/quotes/precise names or titles,
  - AND you are not fully certain by direct reading (small text, dense layout, multi-column, blur, scan).
- OCR is NOT limited to tables/charts/formulas; it can be used for normal headings and paragraphs too.
- For date questions: if not 100% sure about the exact date range, use OCR.

### (3) Region / Coordinate System Rule
- region = [x1, y1, x2, y2], normalized floats in [0,1]
  - (0,0) top-left; x rightward; y downward
  - 0 <= x1 < x2 <= 1, 0 <= y1 < y2 <= 1
- Default zoom regions (crop to zoom, then OCR if needed):
  - Header/title: [0, 0, 1, 0.25]
  - Main body:    [0, 0.2, 1, 0.8]
  - Footer/notes: [0, 0.8, 1, 1]
  - Two-column split:
    - Left:  [0, 0.2, 0.5, 0.85]
    - Right: [0.5, 0.2, 1, 0.85]
- Full-page OCR (if needed): [0, 0, 1, 1]

## Tools (available actions)
- search: {"query": "<string>"}
- crop:  {"image_id": "<image_id>", "region": [x1, y1, x2, y2]}
- ocr:   {"image_id": "<image_id>", "region": [x1, y1, x2, y2]}

## When to Use search (with quality requirements)
- Use search when:
  1) The latest user message contains NO image_id, OR
  2) You already processed the latest evidence images and still cannot find the answer.
- The search query MUST be specific and relevant to the question.
  - It MUST NOT be generic placeholder text.
  - It SHOULD include key entities from the question (e.g., "28th", "UNESCO", "Paris", "IOC") and disambiguate acronyms if needed.

## When to Stop
- Answer only when you have the exact requested info.
- If uncertain after reasonable attempts, provide the best supported answer and state what is missing.

## STRICT JSON OUTPUT SHAPE
{
  "think": "brief plan (1-3 sentences)",
  "action": "search" | "crop" | "ocr" | "answer",
  "arguments": {},
  "answer": "string" | null
}

Rules:
- If action != "answer": answer MUST be null.
- If action == "answer": arguments MUST be {}.
- If action == "search": arguments MUST be {"query": "<a real specific query>"}.
- If action == "crop"/"ocr": arguments MUST include image_id that actually appeared in the latest user message.
"""


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