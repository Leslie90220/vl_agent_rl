import torch
import re
import numpy as np
from collections import defaultdict
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from .tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
from verl.utils.tracking import Tracking
import shutil
import requests
from transformers.image_processing_base import BatchFeature
from PIL import Image
from tqdm import tqdm
import json

def process_image(image, max_pixels: int = 2048 * 2048, min_pixels: int = 512 * 512):
    import math
    from io import BytesIO
    from PIL import Image

    if isinstance(image, dict):
        image = Image.open(BytesIO(image['bytes']))
    elif isinstance(image, str):
        image = Image.open(image)


    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != 'RGB':
        image = image.convert('RGB')

    return image

def bytes_to_base64(image_bytes: bytes) -> str:
    """Convert image bytes to base64 string."""
    import base64
    return base64.b64encode(image_bytes).decode('utf-8')

def call_ocr_api(ocr_api_url: str, image_bytes: bytes, timeout: int = 180) -> str:
    """
    Call PaddleOCR API to extract text from image.
    
    Args:
        ocr_api_url: OCR API endpoint URL
        image_bytes: Image bytes to process
        timeout: Request timeout in seconds
        
    Returns:
        Extracted text content from the image
    """
    base64_image = bytes_to_base64(image_bytes)
    payload = {
        "file": base64_image,
        "fileType": 1,
    }
    
    try:
        resp = requests.post(ocr_api_url, json=payload, timeout=timeout)
        resp.raise_for_status()
        result = resp.json()["result"]
        
        # Parse the OCR result
        layout_results = result.get("layoutParsingResults", [])
        if not layout_results:
            return "No text detected in the image."
        
        res0 = layout_results[0]
        parsing_list = res0.get("prunedResult", {}).get("parsing_res_list", [])
        
        # Extract text content from all blocks
        text_contents = []
        for block in parsing_list:
            label = block.get("block_label", "")
            content = block.get("block_content", "")
            if content:
                text_contents.append(f"[{label}] {content}" if label else content)
        
        if text_contents:
            return "\n".join(text_contents)
        else:
            return "No text content extracted from the image."
            
    except requests.exceptions.Timeout:
        return "OCR request timed out. Please try again."
    except requests.exceptions.RequestException as e:
        return f"OCR request failed: {str(e)}"
    except (KeyError, IndexError) as e:
        return f"Failed to parse OCR response: {str(e)}"

@dataclass
class GenerationConfig:
    max_turns: int
    max_prompt_length: int 
    num_gpus: int
    n_agent: int = 1  # Number of agent workers per GPU, total workers = num_gpus * n_agent
    search_url: str = None
    ocr_api_url: str = "http://172.18.0.1:8080/layout-parsing"  # PaddleOCR API URL
    max_model_len: int = 32768  # Qwen2.5-VL supports 32K context
    image_pad_id: int = 151655
    endoftext_id: int = 151643


class LLMGenerationManager:
    def __init__(
        self,
        processor,
        actor_rollout_wg,
        config: GenerationConfig,
        is_validation: bool = False,
    ):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=self.tokenizer.pad_token_id
        ))

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']
    
    def _postprocess_responses_first(self,batch):
        
        responses_str = self.tokenizer.batch_decode(batch.batch['input_ids'], skip_special_tokens=True)
        responses_str = ["<search>"+item.split('Question: ')[1].split(' \n\nassistant\n')[0]+"</search>" for item in responses_str]

        responses = self._batch_tokenize(responses_str)
        return responses, responses_str
        

    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        """Process responses to stop at search operation or answer operation."""
        
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )

        def extract_tags(text):
            # 定义正则表达式，匹配 <answer>...</answer>、<search>...</search>、<think>...</think> 和 <tools_call>...</tools_call>
            pattern = r"<(answer|search|think|bbox|tools_call)>(.*?)</\1>"
            # 使用 findall 方法找到所有匹配的内容
            matches = re.findall(pattern, text, re.DOTALL)
            # 将匹配的内容重新组合成字符串
            result = "\n".join([f"<{tag}>{content}</{tag}>" for tag, content in matches])
            return result

        responses_str = [extract_tags(resp) + self.tokenizer.eos_token for resp in responses_str]

        responses = self._batch_tokenize(responses_str)
        return responses, responses_str
    #处理观察，构建新的文本和图片上下文结果信息
    def _process_next_obs(self, next_obs: List, rollings) -> torch.Tensor:
        """Process next observations from environment."""
        next_obs_str = []
        multi_modal_data = []
        multi_modal_inputs = []
        merge_length = self.processor.image_processor.merge_size**2
        
        # 获取每个样本的问题（用于在观察中提醒模型）
        questions = []
        if 'extra_info' in rollings.non_tensor_batch:
            for extra_info in rollings.non_tensor_batch['extra_info']:
                if extra_info and isinstance(extra_info, dict):
                    questions.append(extra_info.get('question', ''))
                else:
                    questions.append('')
        else:
            questions = [''] * len(next_obs)
        
        # print(self.retrievaled_images)
        for idx, obs_item in enumerate(next_obs):
            # invalid action or error message (string)
            if isinstance(obs_item,str):
                next_obs_str.append(obs_item)
                multi_modal_data.append({'image': []})
                multi_modal_inputs.append(BatchFeature(dict()))
            # new crop with image_id support
            elif isinstance(obs_item, dict) and 'image_id' in obs_item and 'region' in obs_item:
                action_type = obs_item.get('action_type', 'crop')  # 默认为 crop 以保持向后兼容
                
                if action_type == 'ocr':
                    # OCR 动作：提取图片区域中的文本
                    try:
                        image_id = obs_item['image_id']
                        region = obs_item['region']
                        
                        # 检查 image_id 是否有效
                        if image_id < 1 or image_id > len(self.retrievaled_images[idx]):
                            raise ValueError(f"Invalid image_id: {image_id}. Available images: 1 to {len(self.retrievaled_images[idx])}")
                        
                        # 获取对应的原始图片路径
                        target_image_path = self.retrievaled_images[idx][image_id - 1]
                        
                        # 获取对应的展示图片（用于坐标映射）
                        if 'image' not in rollings.non_tensor_batch['multi_modal_data'][idx] or len(rollings.non_tensor_batch['multi_modal_data'][idx]['image']) < image_id:
                            raise ValueError(f"No displayed image available for image_id: {image_id}")
                        
                        displayed_image = rollings.non_tensor_batch['multi_modal_data'][idx]['image'][image_id - 1]
                        width, height = displayed_image.size
                        
                        # 打开原始图片
                        raw_image = Image.open(target_image_path)
                        raw_width, raw_height = raw_image.size
                        
                        # 坐标映射：从展示图片坐标 → 原始图片坐标
                        crop_area = [
                            int(raw_width * region[0] / width), 
                            int(raw_height * region[1] / height), 
                            int(raw_width * region[2] / width), 
                            int(raw_height * region[3] / height)
                        ]
                        crop_area = [max(0, crop_area[0]), max(0, crop_area[1]), min(raw_width, crop_area[2]), min(raw_height, crop_area[3])]
                        
                        # 裁剪图片区域
                        cropped_image = raw_image.crop((crop_area[0], crop_area[1], crop_area[2], crop_area[3]))
                        
                        # 将裁剪后的图片转换为 bytes
                        from io import BytesIO
                        img_byte_arr = BytesIO()
                        cropped_image.save(img_byte_arr, format='PNG')
                        img_bytes = img_byte_arr.getvalue()
                
                        # 调用 OCR API
                        print(f"  [OCR DEBUG] Calling OCR API for image_{image_id:02d} at region {region}")
                        ocr_result = call_ocr_api(self.config.ocr_api_url, img_bytes)
                        print(f"  [OCR DEBUG] OCR result: {ocr_result[:200]}..." if len(ocr_result) > 200 else f"  [OCR DEBUG] OCR result: {ocr_result}")
                        
                        # 构建返回给模型的文本（不包含图片）
                        obs_str = f'\n<|im_start|>user\nocr_result_from_image_{image_id:02d}:\n{ocr_result}\n<|im_end|>\n<|im_start|>assistant\n'
                        next_obs_str.append(obs_str)
                        multi_modal_data.append({'image': []})
                        multi_modal_inputs.append(BatchFeature(dict()))
                        
                    except Exception as e:
                        print(f"  [OCR ERROR] Failed to perform OCR: {e}")
                        next_obs_str.append(f'\n<|im_start|>user\nYour OCR action failed: {str(e)}. Please check the image_id and region values and try again.\n<|im_end|>\n<|im_start|>assistant\n')
                        multi_modal_data.append({'image': []})
                        multi_modal_inputs.append(BatchFeature(dict()))
                else:
                    # Crop 动作：裁剪图片并返回图片
                    try:
                        image_id = obs_item['image_id']
                        region = obs_item['region']
                        
                        # 检查 image_id 是否有效（image_id 从 1 开始，对应 retrievaled_images 索引 0）
                        if image_id < 1 or image_id > len(self.retrievaled_images[idx]):
                            raise ValueError(f"Invalid image_id: {image_id}. Available images: 1 to {len(self.retrievaled_images[idx])}")
                        
                        # 获取对应的原始图片路径（image_id 从 1 开始）
                        target_image_path = self.retrievaled_images[idx][image_id - 1]
                        
                        # 获取对应的展示图片（用于坐标映射）
                        # multi_modal_data 中的图片顺序与 retrievaled_images 一致
                        if 'image' not in rollings.non_tensor_batch['multi_modal_data'][idx] or len(rollings.non_tensor_batch['multi_modal_data'][idx]['image']) < image_id:
                            raise ValueError(f"No displayed image available for image_id: {image_id}")
                    
                        displayed_image = rollings.non_tensor_batch['multi_modal_data'][idx]['image'][image_id - 1]
                        width, height = displayed_image.size
                        
                        # 打开原始图片进行裁剪
                        raw_images_crop = Image.open(target_image_path)
                        raw_width, raw_height = raw_images_crop.size
                        
                        # 坐标映射：从展示图片坐标 → 原始图片坐标
                        if self.is_validation:
                            region = [region[0]-28, region[1]-28, region[2]+28, region[3]+28]
                        crop_area = [
                            int(raw_width * region[0] / width), 
                            int(raw_height * region[1] / height), 
                            int(raw_width * region[2] / width), 
                            int(raw_height * region[3] / height)
                        ]
                        crop_area = [max(0, crop_area[0]), max(0, crop_area[1]), min(raw_width, crop_area[2]), min(raw_height, crop_area[3])]
                        
                        # 执行裁剪
                        input_images_list = [raw_images_crop.crop((crop_area[0], crop_area[1], crop_area[2], crop_area[3]))]
                        raw_images_list = [process_image(image, 128*28*28, 64*28*28) for image in input_images_list]
                        
                        multi_modal_data.append({'image': raw_images_list})
                        image_inputs = self.processor.image_processor(raw_images_list, return_tensors='pt')
                        multi_modal_inputs.append(image_inputs)
                        image_grid_thw = image_inputs['image_grid_thw']
                        
                        obs_str = ''.join([f"<|vision_start|>{self.processor.image_token * (image_grid_thw_item.prod() // merge_length)}<|vision_end|>" for image_grid_thw_item in image_grid_thw])
                        # 在裁剪结果前添加来源信息（使用两位数格式）
                        obs_str = f'\n<|im_start|>user\ncropped_from_image_{image_id:02d}\n' + obs_str + '<|im_end|>\n<|im_start|>assistant\n'
                        next_obs_str.append(obs_str)
                        
                        print(f"  [CROP DEBUG] Successfully cropped image_{image_id:02d} at region {region}")
                        
                    except Exception as e:
                        print(f"  [CROP ERROR] Failed to crop: {e}")
                        next_obs_str.append(f'\n<|im_start|>user\nYour crop action failed: {str(e)}. Please check the image_id and region values and try again.\n<|im_end|>\n<|im_start|>assistant\n')
                        multi_modal_data.append({'image': []})
                        multi_modal_inputs.append(BatchFeature(dict()))
            # ret image
            elif isinstance(obs_item,list) and isinstance(obs_item[0],dict):
                img_file_list = [item['image_file'] for item in obs_item]
                
                # === DEBUG: 打印检索到的图片列表 ===
                print(f"\n[IMAGE RETRIEVAL DEBUG] Sample {idx}:")
                print(f"  Total images returned by search: {len(img_file_list)}")
                print(f"  Already retrieved images: {len(self.retrievaled_images[idx])}")
                print(f"  Current image_id counter: {self.image_id_counters[idx]}")
                for img_idx, img_file in enumerate(img_file_list[:5]):  # 只打印前5个
                    # 提取页面信息
                    page_info = "unknown"
                    if img_file and isinstance(img_file, str):
                        import re as re_module
                        page_match = re_module.search(r'_page_(\d+)', img_file)
                        if page_match:
                            page_info = page_match.group(1)
                    already_shown = "✓ already shown" if img_file in self.retrievaled_images[idx] else "NEW"
                    #print(f"  [{img_idx}] {img_file} (page: {page_info}) [{already_shown}]")
                    print(f"  [{img_idx}] {img_file} [{already_shown}]")
                # === END DEBUG ===
                
                input_images_list = []  # 初始化为空列表
                current_image_id = None  # 当前图片的ID
                for image_item in img_file_list:
                    if image_item not in self.retrievaled_images[idx]:
                        self.retrievaled_images[idx].append(image_item)
                        # input_images_list = img_file_list[:1]
                        input_images_list = [image_item]
                        # 更新图片ID计数器并获取当前ID
                        self.image_id_counters[idx] += 1
                        current_image_id = self.image_id_counters[idx]
                        
                        # === DEBUG: 打印选中的图片 ===
                        print(f"  [SELECTED] image: {image_item} with image_id: image_{current_image_id:02d}")
                        # === END DEBUG ===
                        break
                
                # 如果所有图片都已检索过，跳过图片处理
                if not input_images_list:
                    print(f"  [WARNING] All images already shown! Asking model to answer.")
                    
                    # === DEBUG: 打印所有检索到的图片，并检查是否包含 reference_page ===
                    # 获取 reference_page 和 file_name
                    reference_page = None
                    file_name = None
                    if 'extra_info' in rollings.non_tensor_batch and idx < len(rollings.non_tensor_batch['extra_info']):
                        extra_info = rollings.non_tensor_batch['extra_info'][idx]
                        if extra_info and isinstance(extra_info, dict):
                            reference_page = extra_info.get('reference_page', None)
                            file_name = extra_info.get('file_name', None)
                    
                    # 构建期望的图片名称 (格式: file_name_page 或 file_name_N)
                    # 例如: 054adbf8327de7348cba0375cbaea22cb7f80813_4
                    expected_image_names = set()
                    if file_name and reference_page:
                        # 去掉文件扩展名
                        file_base = file_name.rsplit('.', 1)[0] if '.' in file_name else file_name
                        if isinstance(reference_page, list):
                            for page in reference_page:
                                expected_image_names.add(f"{file_base}_{page}")
                        else:
                            expected_image_names.add(f"{file_base}_{reference_page}")
                    
                    print(f"  [RETRIEVAL SUMMARY] Sample {idx}:")
                    print(f"    File: {file_name}")
                    print(f"    Expected image(s): {expected_image_names if expected_image_names else 'N/A'}")
                    print(f"    Total retrieved images: {len(self.retrievaled_images[idx])}")
                    
                    # 检查每个检索到的图片是否包含期望的图片
                    found_reference = False
                    
                    for img_idx, img_path in enumerate(self.retrievaled_images[idx]):
                        # 从图片路径中提取图片名称 (去掉路径和扩展名)
                        # 例如: ./search_engine/corpus/img/054adbf8327de7348cba0375cbaea22cb7f80813_4.jpg -> 054adbf8327de7348cba0375cbaea22cb7f80813_4
                        img_name = None
                        if img_path and isinstance(img_path, str):
                            import os
                            # 获取文件名（不含路径）
                            basename = os.path.basename(img_path)
                            # 去掉扩展名
                            img_name = basename.rsplit('.', 1)[0] if '.' in basename else basename
                        
                        is_reference = img_name in expected_image_names if img_name else False
                        if is_reference:
                            found_reference = True
                        
                        marker = "✓ REFERENCE" if is_reference else ""
                        print(f"    [image_{img_idx+1:02d}] {img_name} {marker}")
                    
                    if found_reference:
                        print(f"  [RETRIEVAL RESULT] ✓ Reference image found in retrieved images!")
                        # 记录找到参考图片的样本，用于后续追踪
                        if not hasattr(self, '_samples_found_reference'):
                            self._samples_found_reference = set()
                        self._samples_found_reference.add(idx)
                        print(f"  [TRACKING] Added sample {idx} to _samples_found_reference. Current set: {self._samples_found_reference}")
                    else:
                        print(f"  [RETRIEVAL RESULT] ✗ Reference image NOT found! Expected: {expected_image_names}")
                    # === END DEBUG ===
                    
                    # 获取当前样本的问题
                    question_text = questions[idx] if idx < len(questions) else ""
                    if question_text:
                        next_obs_str.append(f'\n<|im_start|>user\nAll retrieved images have already been shown. Please provide your answer to the question: {question_text[:200]}\n<|im_end|>\n<|im_start|>assistant\n')
                    else:
                        next_obs_str.append('\n<|im_start|>user\nAll retrieved images have already been shown. Please provide your answer based on the information you have.\n<|im_end|>\n<|im_start|>assistant\n')
                    multi_modal_data.append({'image': []})
                    multi_modal_inputs.append(BatchFeature(dict()))
                    continue

                raw_images_list = [process_image(image, 512*28*28, 256*28*28) for image in input_images_list]

                multi_modal_data.append({'image': raw_images_list})
                image_inputs = self.processor.image_processor(raw_images_list, return_tensors='pt')

                multi_modal_inputs.append(image_inputs)
                image_grid_thw = image_inputs['image_grid_thw']

                obs_str = ''.join([f"<|vision_start|>{self.processor.image_token * (image_grid_thw_item.prod() // merge_length)}<|vision_end|>" for image_grid_thw_item in image_grid_thw])
                raw_obs_str = f"<|vision_start|>{self.processor.image_token}<|vision_end|>" * len(image_grid_thw) 
                # 在图片前添加 image_id 标识（使用两位数格式，如 image_01, image_02）
                image_id_str = f"image_{current_image_id:02d}"
                
                # 获取当前样本的问题，用于提醒模型
                question_reminder = ""
                if idx < len(questions) and questions[idx]:
                    # 截取问题的前150个字符作为提醒
                    q = questions[idx][:350]
                    question_reminder = f"\n[Reminder] Your task is to answer: {q}"
                
                #模型观察到的图片字符串（包含问题提醒）
                obs_str = f'\n<|im_start|>user\n{image_id_str}\n' + obs_str + question_reminder + '<|im_end|>\n<|im_start|>assistant\n'
                next_obs_str.append(obs_str)
                
                # === DEBUG: 打印实际回传给模型的内容 ===
                print(f"\n[MODEL INPUT DEBUG] Sample {idx} - Content returned to model:")
                print(f"  image_id in text: {image_id_str}")
                print(f"  Number of image tokens: {sum(image_grid_thw_item.prod() // merge_length for image_grid_thw_item in image_grid_thw)}")
                # 打印不含 image token 的文本部分
                text_only = obs_str.replace(self.processor.image_token, '').replace('<|vision_start|>', '[IMG_START]').replace('<|vision_end|>', '[IMG_END]')
                print(f"  Text structure: {text_only[:200]}")
                print(f"  Total images in context so far: {len(self.retrievaled_images[idx])}")
                print(f"  All image_ids shown: [image_01 ... image_{self.image_id_counters[idx]:02d}]")
                # === END DEBUG ===
            else:
                raise ValueError('invalid observation')
        
        next_obs_ids = self.tokenizer(
            next_obs_str, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']

        return next_obs_ids, next_obs_str, multi_modal_data, multi_modal_inputs
    
    def _concat_multi_modal_data(self, rollings, next_obs_multi_modal_data:list, next_obs_multi_modal_inputs:list):
        if not 'multi_modal_inputs' in rollings.non_tensor_batch.keys():
            # Initialize multi_modal_inputs
            rollings.non_tensor_batch['multi_modal_inputs'] = np.empty(len(next_obs_multi_modal_data), dtype=object)
            for idx, item in enumerate(next_obs_multi_modal_inputs):
                rollings.non_tensor_batch['multi_modal_inputs'][idx] = item

            # Use np.empty with explicit assignment to ensure correct length
            multi_modal_data_array = np.empty(len(next_obs_multi_modal_data), dtype=object)
            multi_modal_data_array[:] = next_obs_multi_modal_data
            rollings.non_tensor_batch['multi_modal_data'] = multi_modal_data_array

        else:
            for idx, multi_modal_data_item in enumerate(next_obs_multi_modal_data):
                # Ensure rollings has the 'image' key initialized (may come from data pipeline without 'image' key)
                if 'image' not in rollings.non_tensor_batch['multi_modal_data'][idx]:
                    rollings.non_tensor_batch['multi_modal_data'][idx]['image'] = []
                
                # Check if multi_modal_data_item has 'image' key and it's not empty
                if 'image' in multi_modal_data_item and len(multi_modal_data_item['image']) > 0:
                    # data
                    rollings.non_tensor_batch['multi_modal_data'][idx]['image'].extend(multi_modal_data_item['image'])
                    if 'pixel_values' in rollings.non_tensor_batch['multi_modal_inputs'][idx]:
                        rollings.non_tensor_batch['multi_modal_inputs'][idx]['pixel_values'] = torch.cat((rollings.non_tensor_batch['multi_modal_inputs'][idx]['pixel_values'], next_obs_multi_modal_inputs[idx]['pixel_values']),dim=0)
                        rollings.non_tensor_batch['multi_modal_inputs'][idx]['image_grid_thw'] = torch.cat((rollings.non_tensor_batch['multi_modal_inputs'][idx]['image_grid_thw'], next_obs_multi_modal_inputs[idx]['image_grid_thw']),dim=0)
                    else:
                        rollings.non_tensor_batch['multi_modal_inputs'][idx]['pixel_values'] = next_obs_multi_modal_inputs[idx]['pixel_values']
                        rollings.non_tensor_batch['multi_modal_inputs'][idx]['image_grid_thw'] = next_obs_multi_modal_inputs[idx]['image_grid_thw']

        return rollings
        

    def _update_rolling_state(self, rollings, cur_responses: torch.Tensor, 
                            next_obs_ids: torch.Tensor) -> Dict:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding
        if next_obs_ids.shape[1] != 0:
            new_input_ids = self.tensor_fn.concatenate_with_padding([
                rollings.batch['input_ids'],
                cur_responses,
                next_obs_ids
            ])
        else:
            new_input_ids = self.tensor_fn.concatenate_with_padding([
                rollings.batch['input_ids'],
                cur_responses
            ])
        # Create attention mask
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        
        # Check if this is Qwen2-VL (needs 3D position_ids)
        is_qwen2_vl = (self.processor is not None and 
                      hasattr(self.processor, 'image_processor') and
                      "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__)
        
        if is_qwen2_vl:
            from verl.models.transformers.qwen2_vl import get_rope_index
            
            batch_size = new_input_ids.shape[0]
            position_ids_list = []
            
            for i in range(batch_size):
                input_ids_i = new_input_ids[i]
                attention_mask_i = new_attention_mask[i]
                
                # Ensure input_ids is long dtype
                if input_ids_i.dtype != torch.long:
                    input_ids_i = input_ids_i.long()
                if attention_mask_i.dtype != torch.long:
                    attention_mask_i = attention_mask_i.long()
                
                # Get image_grid_thw from multi_modal_inputs if available
                image_grid_thw = None
                if 'multi_modal_inputs' in rollings.non_tensor_batch:
                    mm_inputs = rollings.non_tensor_batch['multi_modal_inputs'][i]
                    if mm_inputs is not None:
                        if hasattr(mm_inputs, 'get'):
                            image_grid_thw = mm_inputs.get('image_grid_thw', None)
                        elif hasattr(mm_inputs, 'data') and isinstance(mm_inputs.data, dict):
                            image_grid_thw = mm_inputs.data.get('image_grid_thw', None)
                
                # Compute 3D position_ids using get_rope_index
                vision_position_ids = get_rope_index(
                    self.processor,
                    input_ids=input_ids_i,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=None,
                    second_per_grid_ts=None,
                    attention_mask=attention_mask_i,
                )  # (3, seq_length)
                
                # Create text position ids (first row)
                valid_mask = attention_mask_i.bool()
                text_position_ids = torch.ones((1, len(input_ids_i)), dtype=torch.long, device=input_ids_i.device)
                text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item(), device=input_ids_i.device)
                
                # Concatenate text and vision position ids to get (4, seq_length)
                position_ids_i = torch.cat((text_position_ids, vision_position_ids.to(input_ids_i.device)), dim=0)
                position_ids_list.append(position_ids_i)
            
            # Stack to get (batch_size, 4, seq_length)
            new_position_ids = torch.stack(position_ids_list, dim=0)
        else:
            # For non-Qwen2-VL models, use simple 2D position_ids
            new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length (保留尽可能多的内容，只在超过 max_prompt_length 时截断)
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        # Handle both 2D and 3D position_ids when cutting
        if new_position_ids.ndim == 3:
            return DataProto.from_dict({
                'input_ids': new_input_ids[:, -max_len:],
                'position_ids': new_position_ids[:, :, -max_len:],
                'attention_mask': new_attention_mask[:, -max_len:]
            }, rollings.non_tensor_batch)
        else:
            return DataProto.from_dict({
                'input_ids': new_input_ids[:, -max_len:],
                'position_ids': new_position_ids[:, -max_len:],
                'attention_mask': new_attention_mask[:, -max_len:]
            }, rollings.non_tensor_batch)

    def _update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          next_obs_ids: torch.Tensor = None) -> Dict:
        """Update right side state."""
        if next_obs_ids != None and next_obs_ids.shape[1] != 0:
            responses = self.tensor_fn.concatenate_with_padding([
                right_side['responses'],
                cur_responses,
                next_obs_ids
            ], pad_to_left=False)
        else:
            responses = self.tensor_fn.concatenate_with_padding([
                right_side['responses'],
                cur_responses,
            ], pad_to_left=False)
        
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {'responses': responses[:, :max_len]}


    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
            
            NOTE: When using agent_loop with n_agent > 1, the total number of workers
            is num_gpus * n_agent, so we need to pad to that total for proper chunking.
        """
        # Calculate total workers: num_gpus * n_agent (for agent_loop chunking compatibility)
        total_workers = self.config.num_gpus * self.config.n_agent
        if total_workers <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % total_workers
        
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        # Add padding sequences
        padding_size = total_workers - remainder
        padded_batch = {}
        padded_non_tensor_batch = {}

        padded_ids = self.tokenizer(
            ['<|im_start|>user\nHi, who are u?<|im_end|>\n<|im_start|>assistant\n'], 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']
        padded_ids = padded_ids[0]

        pad_input_ids = torch.full_like(active_batch.batch['input_ids'][0], self.config.endoftext_id, dtype=torch.int64) #151643
        pad_input_ids[:len(padded_ids)] = padded_ids
        pad_attention_mask = self.tensor_fn.create_attention_mask(pad_input_ids)
        pad_input_ids = pad_input_ids.unsqueeze(0)
        pad_attention_mask = pad_attention_mask.unsqueeze(0)
        
        # Check if position_ids is 3D/4D (Qwen2-VL uses [batch, 4, seq_len]) or 2D (standard [batch, seq_len])
        active_position_ids = active_batch.batch['position_ids']
        is_qwen2_vl_position_ids = len(active_position_ids.shape) == 3  # [batch, 4, seq_len] for Qwen2-VL
        
        # Get the target sequence length from active batch
        target_seq_len = active_batch.batch['input_ids'].shape[1]
        
        if is_qwen2_vl_position_ids:
            # For Qwen2-VL, create proper 4D position_ids for padding samples
            # Since padding samples have no images, we create simple position_ids directly
            # (Don't use get_rope_index because it filters by attention_mask internally)
            
            pad_attention_mask_long = pad_attention_mask[0].long() if pad_attention_mask.dtype != torch.long else pad_attention_mask[0]
            valid_mask = pad_attention_mask_long.bool()
            num_valid = valid_mask.sum().item()
            device = pad_input_ids.device
            
            # Create text position ids (first row): 0, 1, 2, ... for valid tokens, 1 for padding
            text_position_ids = torch.ones((1, target_seq_len), dtype=torch.long, device=device)
            text_position_ids[0, valid_mask] = torch.arange(num_valid, dtype=torch.long, device=device)
            
            # Create vision position ids (3 rows for t, h, w): same as text for pure text input
            # For text-only input, all 3 dimensions use the same position ids
            vision_position_ids = torch.ones((3, target_seq_len), dtype=torch.long, device=device)
            vision_position_ids[:, valid_mask] = torch.arange(num_valid, dtype=torch.long, device=device).unsqueeze(0).expand(3, -1)
            
            # Concatenate text and vision position ids to get (4, seq_length)
            pad_position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)
            pad_position_ids = pad_position_ids.unsqueeze(0)  # (1, 4, seq_length)
        else:
            # For non-Qwen2-VL models, use simple 2D position_ids
            pad_position_ids = self.tensor_fn.create_position_ids(pad_attention_mask)
        
        padded_batch['attention_mask'] = torch.cat([active_batch.batch['attention_mask'], pad_attention_mask.repeat(padding_size, *[1] * (len(active_batch.batch['attention_mask'].shape) - 1))], dim=0)
        padded_batch['input_ids'] = torch.cat([active_batch.batch['input_ids'], pad_input_ids.repeat(padding_size, *[1] * (len(active_batch.batch['input_ids'].shape) - 1))], dim=0)
        padded_batch['position_ids'] = torch.cat([active_batch.batch['position_ids'], pad_position_ids.repeat(padding_size, *[1] * (len(active_batch.batch['position_ids'].shape) - 1))], dim=0)
        

        for k, v in active_batch.non_tensor_batch.items():
            pad_non_tensor_item = np.empty(padding_size, dtype=object)
            if k == 'raw_prompt_ids':
                list_ids = padded_ids.tolist()
                for idx in range(padding_size):
                    pad_non_tensor_item[idx] = list_ids
            elif k == 'raw_prompt':
                # Create a dummy raw_prompt for padding
                dummy_prompt = [{"role": "user", "content": "Hi, who are u?"}]
                for idx in range(padding_size):
                    pad_non_tensor_item[idx] = dummy_prompt
            elif k == 'multi_modal_inputs':
                for idx in range(padding_size):
                    pad_non_tensor_item[idx] = {}
            elif k == 'multi_modal_data':
                for idx in range(padding_size):
                    pad_non_tensor_item[idx] = {'image': []}
            padded_non_tensor_batch[k] = np.concatenate([v, pad_non_tensor_item])
                
        padded_active_batch = DataProto.from_dict(padded_batch, padded_non_tensor_batch)
        
        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)
        
        # Remove padding from output by slicing (keeps proper DataProto structure)
        # Use DataProto's built-in slicing which properly handles both batch and non_tensor_batch
        original_batch_size = batch_size  # The original batch size before padding
        trimmed_output = padded_output[:original_batch_size]
        
        return trimmed_output

    def _raw_prompt_ids(self, rollings):
        new_raw_prompt_ids = []
        rollings.batch['input_ids'] = rollings.batch['input_ids'].long()
        raw_next_obs_ids = [ids[mask == 1].tolist() for ids, mask in zip(np.array(rollings.batch['input_ids']),  np.array(rollings.batch['attention_mask']))]
        def replace_consecutive_elements(arr, target):
            result = []
            i = 0
            while i < len(arr):
                if arr[i] == target:
                    result.append(target)
                    while i + 1 < len(arr) and arr[i + 1] == target:
                        i += 1
                else:
                    result.append(arr[i])
                i += 1
            return result
        raw_next_obs_ids = [replace_consecutive_elements(row,self.config.image_pad_id) for row in raw_next_obs_ids] #151655
        raw_next_obs_ids = np.array(raw_next_obs_ids, dtype=object)
        rollings.non_tensor_batch['raw_prompt_ids'] = raw_next_obs_ids
        return rollings

    def deactivate_batch(self, active_mask,rollings):
        raw_prompt_ids = rollings.non_tensor_batch['raw_prompt_ids']
        max_model_len = self.config.max_model_len
        curr_active_mask = torch.tensor([len(raw_prompt_ids_item) < max_model_len for raw_prompt_ids_item in raw_prompt_ids], dtype=torch.bool)
        active_mask = active_mask * curr_active_mask
        return active_mask

    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""

        # === DEBUG: 打印样本的参考信息 ===
        print(f"\n{'*'*70}")
        print(f"[BATCH DEBUG] Starting LLM loop with {gen_batch.batch['input_ids'].shape[0]} samples")
        if 'extra_info' in gen_batch.non_tensor_batch:
            for idx, extra_info in enumerate(gen_batch.non_tensor_batch['extra_info'][:3]):  # 只打印前3个
                if extra_info and isinstance(extra_info, dict):
                    question = extra_info.get('question', 'N/A')[:100]
                    answer = extra_info.get('answer', 'N/A')[:100]
                    file_name = extra_info.get('file_name', 'N/A')
                    reference_page = extra_info.get('reference_page', 'N/A')
                    print(f"  [Sample {idx}]")
                    print(f"    Question: {question}...")
                    print(f"    Ground Truth: {answer}...")
                    print(f"    File: {file_name}")
                    print(f"    Reference Page: {reference_page}")
        print(f"{'*'*70}\n")
        # === END DEBUG ===

        # original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
         # 保存原始输入（左侧 prompt）初始化响应序列（右侧，初始为空）
        original_left_side = {'input_ids': initial_input_ids}
        original_right_side = {'responses': initial_input_ids[:, []]}
        # 活跃掩码：标记哪些样本还在交互中（未结束）
        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)
        active_num_list = [active_mask.sum().item()]
        # 滚动状态：包含当前上下文
        rollings = gen_batch
        # rollings_multimodal_data = gen_batch.non_tensor_batch.get('multi_modal_inputs', None)
        # rollings_multimodal_data = gen_batch.non_tensor_batch['multi_modal_inputs']
        # rollings_multimodal_data = None
        raw_prompt_ids = rollings.non_tensor_batch['raw_prompt_ids']
        # 每个样本的检索图像列表
        self.retrievaled_images = [[] for _ in range(gen_batch.batch['input_ids'].shape[0])]
        # 每个样本的图片ID计数器，从1开始
        self.image_id_counters = [0 for _ in range(gen_batch.batch['input_ids'].shape[0])]
        # 追踪哪些样本已经找到参考图片（用于调试）
        self._samples_found_reference = set()

        # Main generation loop主循环
        for step in range(self.config.max_turns):
            # === DEBUG: 打印当前步数和活跃样本数 ===
            print(f"\n{'#'*70}")
            print(f"[LOOP DEBUG] Step {step + 1}/{self.config.max_turns}, Active samples: {active_mask.sum().item()}/{len(active_mask)}")
            print(f"{'#'*70}\n")
            # === END DEBUG ===
            
            if not active_mask.sum():
                break# Step1：所有活跃样本都已结束，退出循环
            #Step 2: 裁剪序列到有效长度
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )

            rollings = self._raw_prompt_ids(rollings)
            #Step 3: 检查序列长度，停用超长样本
            active_mask = self.deactivate_batch(active_mask, rollings)
            if not active_mask.sum():
                break
            # Step 4: 提取活跃样本，只对活跃样本进行生成，节省计算
            if 'multi_modal_inputs' in rollings.non_tensor_batch.keys():
                rollings_active = DataProto.from_dict(
                    tensors={k: v[active_mask] for k, v in rollings.batch.items()},
                    non_tensors={k: v[active_mask] for k, v in rollings.non_tensor_batch.items()}
                )
            else:
                rollings_active = DataProto.from_dict({
                    k: v[active_mask] for k, v in rollings.batch.items()
                })

            # Step 5: 模型生成响应。调用 vLLM 生成，处理多 GPU padding
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info   
            # 后处理：提取 <think>, <search>, <answer>, <tools_call> 标签内容         
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])

            # === DEBUG: 打印模型响应和解析出的动作 ===
            print(f"\n[RESPONSE DEBUG] Step {step + 1} - First response:")
            print(f"{responses_str[0][:500]}...")  # 只打印前500字符
            
            # 解析动作
            cur_actions_debug, contents_debug = self.postprocess_predictions(responses_str)
            action_counts = {}
            for act in cur_actions_debug:
                action_counts[act] = action_counts.get(act, 0) + 1
            print(f"[ACTION DEBUG] Action distribution: {action_counts}")
            
            # 打印所有动作类型（answer, search, crop, ocr, None）
            for idx, (act, content) in enumerate(zip(cur_actions_debug, contents_debug)):
                if act == 'answer':
                    print(f"[ANSWER DEBUG] Sample {idx} answered: {content[:200] if content else 'empty'}")
                elif act == 'search':
                    print(f"[SEARCH DEBUG] Sample {idx} searching: {content if content else 'empty'}")
                elif act == 'crop':
                    # crop content 是一个 dict，包含 image_id 和 region
                    if isinstance(content, dict):
                        image_id = content.get('image_id', 'unknown')
                        region = content.get('region', [])
                        print(f"[CROP DEBUG] Sample {idx} cropping: image_id={image_id}, region={region}")
                    else:
                        print(f"[CROP DEBUG] Sample {idx} cropping: {content}")
                elif act == 'ocr':
                    # ocr content 是一个 dict，包含 image_id 和 region
                    if isinstance(content, dict):
                        image_id = content.get('image_id', 'unknown')
                        region = content.get('region', [])
                        print(f"[OCR DEBUG] Sample {idx} OCR: image_id={image_id}, region={region}")
                    else:
                        print(f"[OCR DEBUG] Sample {idx} OCR: {content}")
                elif act is None:
                    # 打印 None 动作的详细信息，帮助诊断问题
                    resp = responses_str[idx] if idx < len(responses_str) else "N/A"
                    print(f"[NONE ACTION DEBUG] Sample {idx} has no valid action!")
                    print(f"  Response preview: {resp[:300]}...")
                    # 检查是否有部分标签
                    has_tools_call = '<tools_call>' in resp
                    has_search = '<search>' in resp
                    has_answer = '<answer>' in resp
                    print(f"  Tags found: <tools_call>={has_tools_call}, <search>={has_search}, <answer>={has_answer}")
            
            # === 特别关注：检查哪些样本已经找到参考图片但仍在搜索 ===
            if hasattr(self, '_samples_found_reference'):
                for sample_idx in self._samples_found_reference:
                    if sample_idx < len(cur_actions_debug):
                        act = cur_actions_debug[sample_idx]
                        resp = responses_str[sample_idx] if sample_idx < len(responses_str) else "N/A"
                        print(f"\n[REFERENCE FOUND BUT NOT ANSWERING] Sample {sample_idx}:")
                        print(f"  Action: {act}")
                        print(f"  Full response: {resp}")
                        if act != 'answer':
                            print(f"  ⚠️ WARNING: Sample {sample_idx} found reference image but did NOT answer!")
            # === END 特别关注 ===
            
            print(f"{'='*60}\n")
            # === END DEBUG ===

            print(responses_str[0])
            
            # === 第一轮强制搜索：如果模型没有输出有效动作，自动用问题作为搜索 query ===
            if step == 0:
                # 检查是否有任何样本没有有效动作
                has_none = any(act is None for act in cur_actions_debug)
                if has_none:
                    print(f"\n[AUTO SEARCH] Step 1: Some samples didn't output valid action, forcing search with question...")
                    # 从 extra_info 中获取问题作为搜索 query
                    forced_responses_str = []
                    for idx in range(len(responses_str)):
                        # 只对 None 动作的样本强制搜索
                        if cur_actions_debug[idx] is None:
                            if 'extra_info' in rollings.non_tensor_batch and idx < len(rollings.non_tensor_batch['extra_info']):
                                extra_info = rollings.non_tensor_batch['extra_info'][idx]
                                if extra_info and isinstance(extra_info, dict):
                                    question = extra_info.get('question', '')
                                    if question:
                                        # 构造强制搜索的响应
                                        forced_response = f"<think>I need to search for relevant information to answer this question.</think>\n<search>{question}</search>" + self.tokenizer.eos_token
                                        forced_responses_str.append(forced_response)
                                        print(f"  [AUTO SEARCH] Sample {idx}: Forcing search for '{question[:80]}...'")
                                        continue
                            # 如果没有问题信息，保持原样
                            forced_responses_str.append(responses_str[idx])
                        else:
                            # 有效动作的样本保持原样
                            forced_responses_str.append(responses_str[idx])
                    
                    responses_str = forced_responses_str
                    responses_ids = self._batch_tokenize(responses_str)
            # === END 第一轮强制搜索 ===
            
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)
            # Execute in environment and process observations
            # Step 6: 执行环境交互
            next_obs, dones = self.execute_predictions(responses_str, self.tokenizer.pad_token, active_mask)
            # Step 7: 更新活跃掩码
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            # Step 8: 处理环境观察（检索结果）
            next_obs_ids, next_obs_str, next_obs_multi_modal_data, next_obs_multi_modal_inputs = self._process_next_obs(next_obs, rollings)
            # Step 9: 拼接多模态数据
            rollings = self._concat_multi_modal_data(
                rollings,
                next_obs_multi_modal_data,
                next_obs_multi_modal_inputs
            )
            # Update states  Step 10: 更新状态          
            rollings = self._update_rolling_state(
                rollings,
                responses_ids,
                next_obs_ids
            )# 更新滚动上下文（用于下一轮生成）
            
            # === DEBUG: 打印当前上下文摘要（只打印第一个样本）===
            if step < 3 or step == self.config.max_turns - 1:  # 只打印前3轮和最后一轮
                print(f"\n[CONTEXT SUMMARY] Step {step + 1} - Sample 0:")
                # 解码当前上下文
                sample_0_ids = rollings.batch['input_ids'][0]
                sample_0_mask = rollings.batch['attention_mask'][0]
                valid_ids = sample_0_ids[sample_0_mask == 1]
                context_text = self.tokenizer.decode(valid_ids, skip_special_tokens=False)
                # 统计 image_id 出现次数
                import re as re_module
                image_ids_in_context = re_module.findall(r'image_(\d+)', context_text)
                print(f"  Context length (tokens): {len(valid_ids)}")
                print(f"  Image IDs found in context: {sorted(set(image_ids_in_context))}")
                print(f"  Number of images in multi_modal_data: {len(rollings.non_tensor_batch.get('multi_modal_data', [{}])[0].get('image', []))}")
                
                # 统计 vision_start 和 vision_end 的数量
                vision_start_count = context_text.count('<|vision_start|>')
                vision_end_count = context_text.count('<|vision_end|>')
                print(f"  <|vision_start|> count: {vision_start_count}")
                print(f"  <|vision_end|> count: {vision_end_count}")
                
                # === 关键检查：问题是否在上下文中 ===
                # 检查 "## Question" 标记是否存在
                has_question_marker = '## Question' in context_text
                # 从 extra_info 获取原始问题
                original_question = ""
                if 'extra_info' in rollings.non_tensor_batch and len(rollings.non_tensor_batch['extra_info']) > 0:
                    extra_info = rollings.non_tensor_batch['extra_info'][0]
                    if extra_info and isinstance(extra_info, dict):
                        original_question = extra_info.get('question', '')[:100]
                
                # 检查问题内容是否在上下文中（取前50个字符匹配）
                question_in_context = original_question[:50] in context_text if original_question else False
                
                print(f"  [QUESTION CHECK] '## Question' marker present: {has_question_marker}")
                print(f"  [QUESTION CHECK] Original question (first 100 chars): {original_question}")
                print(f"  [QUESTION CHECK] Question content in context: {question_in_context}")
                
                if not has_question_marker or not question_in_context:
                    print(f"  [WARNING] ⚠️ QUESTION MAY BE TRUNCATED FROM CONTEXT! ⚠️")
                    print(f"  [WARNING] This could cause the model to search indefinitely without knowing what to answer.")
                # === END 关键检查 ===
                
                # 打印上下文的前500和后500字符（去掉 image token）
                context_preview = context_text.replace(self.processor.image_token, '[IMG]')
                print(f"  Context start: {context_preview[:3500]}...")
                print(f"  Context end: ...{context_preview[-9500:]}")
            # === END DEBUG ===
            
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                next_obs_ids
            )# 更新右侧响应序列（用于最终输出）



        # final LLM rollout # 如果还有活跃样本，再做一次最终生成
        # 在最终轮次，给模型添加一个强制回答的提示
        if active_mask.sum():
            print(f"\n[FINAL ROLLOUT] {active_mask.sum().item()} samples still active after {self.config.max_turns} turns. Forcing final answer...")
            
            # 添加强制回答提示到活跃样本的上下文中
            force_answer_prompt = '\n<|im_start|>user\nYou have reached the maximum number of search turns. You MUST provide your final answer NOW based on all the information you have gathered. Put your answer inside <answer> and </answer> tags.\n<|im_end|>\n<|im_start|>assistant\n'
            force_answer_ids = self.tokenizer(
                [force_answer_prompt], 
                padding='longest',
                return_tensors='pt',
                add_special_tokens=False,
            )['input_ids']
            
            # 为每个样本创建强制回答提示（活跃样本用提示，非活跃样本用空）
            force_answer_ids_batch = []
            for i in range(len(active_mask)):
                if active_mask[i]:
                    force_answer_ids_batch.append(force_answer_ids[0])
                else:
                    force_answer_ids_batch.append(torch.tensor([], dtype=torch.long))
            
            # Pad to same length
            max_len = max(len(ids) for ids in force_answer_ids_batch)
            if max_len > 0:
                padded_force_ids = torch.full((len(force_answer_ids_batch), max_len), self.tokenizer.pad_token_id, dtype=torch.long)
                for i, ids in enumerate(force_answer_ids_batch):
                    if len(ids) > 0:
                        padded_force_ids[i, :len(ids)] = ids
                
                # 更新 rollings 状态，添加强制回答提示
                rollings = self._update_rolling_state(
                    rollings,
                    torch.zeros((len(active_mask), 0), dtype=torch.long),  # 空响应
                    padded_force_ids
                )

            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            rollings = self._raw_prompt_ids(rollings)

            active_mask = self.deactivate_batch(active_mask, rollings)

            if active_mask.sum():

                if 'multi_modal_inputs' in rollings.non_tensor_batch.keys():
                    rollings_active = DataProto.from_dict(
                        tensors={k: v[active_mask] for k, v in rollings.batch.items()},
                        non_tensors={k: v[active_mask] for k, v in rollings.non_tensor_batch.items()}
                    )
                else:
                    rollings_active = DataProto.from_dict({
                        k: v[active_mask] for k, v in rollings.batch.items()
                    })

                gen_output = self._generate_with_gpu_padding(rollings_active)

                meta_info = gen_output.meta_info
                responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
                responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

                # # Execute in environment and process observations
                # 在最终轮次，强制所有样本结束（不管模型输出什么动作）
                _, dones = self.execute_predictions(
                    responses_str, self.tokenizer.pad_token, active_mask, do_search=False
                )
                
                # 强制所有仍然活跃的样本结束
                print(f"[FINAL ROLLOUT] Forcing all remaining {active_mask.sum().item()} samples to terminate.")
                dones = [1] * len(dones)  # 强制所有样本结束

                curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
                active_mask = active_mask * curr_active_mask
                active_num_list.append(active_mask.sum().item())

                original_right_side = self._update_right_side(
                    original_right_side,
                    responses_ids,
                )
        
        print("ACTIVE_TRAJ_NUM:", active_num_list)
        
        # =================== raw prompt ids ===================
        rollings.non_tensor_batch['raw_prompt_ids'] = raw_prompt_ids
        # rollings.non_tensor_batch.pop('raw_prompt_ids')
        # 添加噪声多模态数据（训练时，防止模型依赖图像存在）
        if not self.is_validation:
            rollings, original_right_side = self._add_noisy_multi_modal_data(rollings, original_right_side)
        ### check again
        
        retrievaled_images_array = np.empty(len(self.retrievaled_images), dtype=object)
        for idx in range(len(self.retrievaled_images)):
            retrievaled_images_array[idx] = self.retrievaled_images[idx]
        # 保存检索到的图像路径（用于计算 NDCG）
        rollings.non_tensor_batch['retrievaled_images'] = retrievaled_images_array
        # 组装最终输出
        return self._compose_final_output(original_left_side, original_right_side, meta_info, rollings)
    
    def _add_noisy_multi_modal_data(self, rollings, original_right_side):
        image_padded = Image.new('RGB', (64, 64), (0, 0, 0))

        image_padded = process_image(image_padded, 256*256, 128*128)
        image_inputs = self.processor.image_processor([image_padded], return_tensors='pt')
        image_grid_thw = image_inputs['image_grid_thw']
        merge_length = self.processor.image_processor.merge_size**2
        padded_str = f"\n<|im_start|>user\n<|vision_start|>{self.processor.image_token * (image_grid_thw.prod() // merge_length)}<|vision_end|><|im_end|>"

        padded_str_list = []
        for idx, multi_modal_item in enumerate(rollings.non_tensor_batch['multi_modal_data']):
            # Ensure 'image' key exists
            if 'image' not in multi_modal_item:
                multi_modal_item['image'] = []
            
            if len(multi_modal_item['image']) == 0:
                padded_str_list.append(padded_str)
                rollings.non_tensor_batch['multi_modal_data'][idx]['image'].append(image_padded)
                rollings.non_tensor_batch['multi_modal_inputs'][idx] = image_inputs
            else:
                padded_str_list.append('')
            
        padded_ids = self.tokenizer(
            padded_str_list, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']

        original_right_side = self._update_right_side(
            original_right_side,
            padded_ids
        )
        return rollings, original_right_side


    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict,
                            rollings) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        
        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        
        # Compute proper 3D position_ids for Qwen2-VL (mrope)
        # Check if this is a Qwen2-VL model by checking processor type
        is_qwen2_vl = (self.processor is not None and 
                      hasattr(self.processor, 'image_processor') and
                      "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__)
        
        if is_qwen2_vl:
            from verl.models.transformers.qwen2_vl import get_rope_index
            
            batch_size = final_output['input_ids'].shape[0]
            position_ids_list = []
            
            for i in range(batch_size):
                input_ids_i = final_output['input_ids'][i]
                attention_mask_i = final_output['attention_mask'][i]
                
                # Ensure input_ids is long dtype (required for get_rope_index)
                if input_ids_i.dtype != torch.long:
                    input_ids_i = input_ids_i.long()
                if attention_mask_i.dtype != torch.long:
                    attention_mask_i = attention_mask_i.long()
                
                # Get image_grid_thw from multi_modal_inputs if available
                image_grid_thw = None
                if 'multi_modal_inputs' in rollings.non_tensor_batch:
                    mm_inputs = rollings.non_tensor_batch['multi_modal_inputs'][i]
                    # Handle both dict and BatchFeature types
                    if mm_inputs is not None:
                        if hasattr(mm_inputs, 'get'):
                            # It's a dict or BatchFeature (both have .get method)
                            image_grid_thw = mm_inputs.get('image_grid_thw', None)
                        elif hasattr(mm_inputs, 'data') and isinstance(mm_inputs.data, dict):
                            # It's a BatchFeature with .data attribute
                            image_grid_thw = mm_inputs.data.get('image_grid_thw', None)
                
                # Compute 3D position_ids using get_rope_index
                vision_position_ids = get_rope_index(
                    self.processor,
                    input_ids=input_ids_i,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=None,
                    second_per_grid_ts=None,
                    attention_mask=attention_mask_i,
                )  # (3, seq_length)
                
                # Create text position ids (first row)
                valid_mask = attention_mask_i.bool()
                text_position_ids = torch.ones((1, len(input_ids_i)), dtype=torch.long, device=input_ids_i.device)
                text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item(), device=input_ids_i.device)
                
                # Concatenate text and vision position ids to get (4, seq_length)
                position_ids_i = torch.cat((text_position_ids, vision_position_ids.to(input_ids_i.device)), dim=0)
                position_ids_list.append(position_ids_i)
            
            # Stack to get (batch_size, 4, seq_length)
            final_output['position_ids'] = torch.stack(position_ids_list, dim=0)
        else:
            # For non-Qwen2-VL models, use simple 2D position_ids
            final_output['position_ids'] = self.tensor_fn.create_position_ids(
                final_output['attention_mask']
            )

        final_output = DataProto.from_dict(final_output,rollings.non_tensor_batch)
        final_output.meta_info.update(meta_info)
        
        return final_output
    # 执行动作
    def execute_predictions(self, predictions: List[str], pad_token: str, active_mask=None, do_search=True) -> List[str]:
        """
        Execute predictions across multiple environments.
        NOTE: the function is the actual `step` function in the environment
        NOTE penalty_for_invalid is not included in observation shown to the LLM
        
        Args:
            envs: List of environment instances
            predictions: List of action predictions
            pad_token: Token to use for padding
            
        Returns:
            List of observation strings
        """
        cur_actions, contents = self.postprocess_predictions(predictions)
        next_obs, dones = [], []
        
        crop_list = [content for action, content in zip(cur_actions, contents) if action == 'crop']
        ocr_list = [content for action, content in zip(cur_actions, contents) if action == 'ocr']
        search_queries = [content for action, content in zip(cur_actions, contents) if action == 'search']
        
        # 建立搜索查询到原始响应的映射（用于调试打印）
        search_query_to_response = {}
        search_query_to_sample_idx = {}
        for idx, (action, content) in enumerate(zip(cur_actions, contents)):
            if action == 'search':
                search_query_to_response[content] = predictions[idx]
                search_query_to_sample_idx[content] = idx
        
        if do_search:
            if len(search_queries) > 0:
                batch_size = 100
                search_results = []
                for i in range(0, len(search_queries), batch_size):
                    batch_queries = search_queries[i:i + batch_size]
                    response = requests.get(self.config.search_url, params={"queries": batch_queries})
                    search_results_single_batch = response.json()
                    search_results.extend(search_results_single_batch)
                    
                    # === DEBUG: 打印 search_engine 返回的结果，以及模型的完整 think 和 tools_call ===
                    for query_idx, (query, results) in enumerate(zip(batch_queries, search_results_single_batch)):
                        print(f"\n{'='*60}")
                        print(f"[SEARCH DEBUG] Query {i + query_idx}: {query}")
                        print(f"[SEARCH DEBUG] Number of results: {len(results)}")
                        
                        # 打印模型的完整响应（包含 <think> 和 <tools_call>）
                        if query in search_query_to_response:
                            sample_idx = search_query_to_sample_idx.get(query, 'unknown')
                            full_response = search_query_to_response[query]
                            print(f"[SEARCH DEBUG] Sample {sample_idx} - Model full response:")
                            print(f"============================ Model full response ============================")
                            print(f"{full_response}")
                            print(f"============================ End of model response ============================")
                        
                        if len(results) > 0:
                            for res_idx, res in enumerate(results[:5]):  # 只打印前5个结果
                                image_file = res.get('image_file', 'N/A')
                                # 从文件名提取页面信息 (格式通常是: xxx_page_N.png)
                                page_info = "unknown"
                                if image_file and isinstance(image_file, str):
                                    import re as re_module
                                    page_match = re_module.search(r'_page_(\d+)', image_file)
                                    if page_match:
                                        page_info = page_match.group(1)
                                #print(f"  [Result {res_idx}] image_file: {image_file}, page: {page_info}")
                                print(f"  [Result {res_idx}] image_file: {image_file}")
                        else:
                            print(f"  [SEARCH DEBUG] WARNING: No results returned!")
                        print(f"{'='*60}\n")
                    # === END DEBUG ===
                    
                assert len(search_results) == sum([1 for action in cur_actions if action == 'search'])
            else:
                search_results = []
        else:
            search_results = [''] * sum([1 for action in cur_actions if action == 'search'])

        for i, (action, active) in enumerate(zip(cur_actions, active_mask)):
            
            if not active:
                next_obs.append('')
                dones.append(1)
            else:
                if action == 'answer':
                    # 回答动作 → 结束该样本
                    next_obs.append('')
                    dones.append(1)# 标记为完成
                elif action == 'search':
                    # 搜索动作 → 调用搜索引擎
                    # next_obs.append(f'\n\n<information>{search_results.pop(0).strip()}</information>\n\n')
                    next_obs.append(search_results.pop(0))
                    dones.append(0)
                elif action == 'crop':
                    # 新的裁剪动作 → 支持指定 image_id
                    try:
                        crop_args = crop_list.pop(0)
                        image_id_str = crop_args.get('image_id', '')
                        region = crop_args.get('region', [])
                        
                        # 解析 image_id，格式如 "image_01" 或 "image_1"
                        image_id_match = re.search(r'image_(\d+)', image_id_str)
                        if image_id_match:
                            image_id = int(image_id_match.group(1))
                        else:
                            raise ValueError(f"Invalid image_id format: {image_id_str}")
                        
                        if len(region) == 4 and all(coord >= 0 for coord in region):
                            # 返回包含 image_id 和 region 的字典
                            next_obs.append({'image_id': image_id, 'region': region})
                        else:
                            raise ValueError("Invalid region value")
                    except Exception as e:
                        print(f"[CROP ERROR] {e}")
                        next_obs.append('\n<|im_start|>user\nYour previous crop action is invalid. Please use the correct format: <tools_call>{"type": "crop", "arguments": {"image_id": "image_01", "region": [x1, y1, x2, y2]}}</tools_call>. Please try again.\n<|im_end|>\n<|im_start|>assistant\n')
                    dones.append(0)
                elif action == 'ocr':
                    # OCR 动作 → 提取图片中的文本
                    try:
                        ocr_args = ocr_list.pop(0)
                        image_id_str = ocr_args.get('image_id', '')
                        region = ocr_args.get('region', [])
                        
                        # 解析 image_id，格式如 "image_01" 或 "image_1"
                        image_id_match = re.search(r'image_(\d+)', image_id_str)
                        if image_id_match:
                            image_id = int(image_id_match.group(1))
                        else:
                            raise ValueError(f"Invalid image_id format: {image_id_str}")
                        
                        if len(region) == 4 and all(coord >= 0 for coord in region):
                            # 返回包含 image_id, region 和 action_type 的字典
                            next_obs.append({'action_type': 'ocr', 'image_id': image_id, 'region': region})
                        else:
                            raise ValueError("Invalid region value")
                    except Exception as e:
                        print(f"[OCR ERROR] {e}")
                        next_obs.append('\n<|im_start|>user\nYour previous OCR action is invalid. Please use the correct format: <tools_call>{"type": "ocr", "arguments": {"image_id": "image_01", "region": [x1, y1, x2, y2]}}</tools_call>. Please try again.\n<|im_end|>\n<|im_start|>assistant\n')
                    dones.append(0)
                else:
                    # 无效动作 → 返回错误提示
                    next_obs.append('\n<|im_start|>user\nYour previous action is invalid. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you want to search, you should put the query between <search> and </search>.\nIf you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Please try again.\n<|im_end|>\n<|im_start|>assistant\n')
                    dones.append(0)
            
        assert len(search_results) == 0

        return next_obs, dones
    #解析调用工具
    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[int], List[bool]]:
        """
        Process (text-based) predictions from llm into actions and validity flags.
        
        Args:
            predictions: List of raw predictions
            
        Returns:
            Tuple of (actions list, validity flags list)
        """
        actions = []
        contents = []
                
        for idx, prediction in enumerate(predictions):
            if isinstance(prediction, str): # for llm output
                # === DEBUG: 打印原始预测内容 ===
                if idx == 0:  # 只打印第一个样本的详细信息
                    print(f"\n[PARSE DEBUG] Sample {idx} raw prediction (first 500 chars):")
                    print(f"  {prediction[:500]}")
                # === END DEBUG ===
                
                # 首先尝试匹配新的 <tools_call> 格式
                tools_call_pattern = r'<tools_call>(.*?)</tools_call>'
                tools_call_match = re.search(tools_call_pattern, prediction, re.DOTALL)
                
                if tools_call_match:
                    try:
                        tools_call_content = tools_call_match.group(1).strip()
                        if idx == 0:
                            print(f"  [PARSE DEBUG] Found <tools_call>: {tools_call_content[:200]}")
                        tools_call_json = json.loads(tools_call_content)
                        tool_type = tools_call_json.get('type', '')
                        
                        if tool_type == 'crop':
                            # crop 格式: {"type": "crop", "arguments": {"image_id": "image_01", "region": [x1, y1, x2, y2]}}
                            action = 'crop'
                            content = tools_call_json.get('arguments', {})
                        elif tool_type == 'ocr':
                            # ocr 格式: {"type": "ocr", "arguments": {"image_id": "image_01", "region": [x1, y1, x2, y2]}}
                            action = 'ocr'
                            content = tools_call_json.get('arguments', {})
                        elif tool_type == 'search':
                            # search 格式: {"type": "search", "arguments": {"query": "..."}}
                            action = 'search'
                            content = tools_call_json.get('arguments', {}).get('query', '')
                        else:
                            action = None
                            content = ''
                        if idx == 0:
                            print(f"  [PARSE DEBUG] Parsed action: {action}, content: {str(content)[:100]}")
                    except json.JSONDecodeError as e:
                        if idx == 0:
                            print(f"  [PARSE DEBUG] JSON decode error: {e}")
                        action = None
                        content = ''
                else:
                    # 回退到标签格式 (search, answer)
                    pattern = r'<(search|answer)>(.*?)</\1>'
                    match = re.search(pattern, prediction, re.DOTALL)
                    if match:
                        content = match.group(2).strip()  # Return only the content inside the tags
                        action = match.group(1)
                        if idx == 0:
                            print(f"  [PARSE DEBUG] Found legacy tag <{action}>: {content[:100]}")
                    else:
                        content = ''
                        action = None
                        if idx == 0:
                            # 检查是否有部分匹配
                            has_tools_call_open = '<tools_call>' in prediction
                            has_tools_call_close = '</tools_call>' in prediction
                            has_search_open = '<search>' in prediction
                            has_search_close = '</search>' in prediction
                            has_answer_open = '<answer>' in prediction
                            has_answer_close = '</answer>' in prediction
                            print(f"  [PARSE DEBUG] No valid action found!")
                            print(f"    <tools_call> open: {has_tools_call_open}, close: {has_tools_call_close}")
                            print(f"    <search> open: {has_search_open}, close: {has_search_close}")
                            print(f"    <answer> open: {has_answer_open}, close: {has_answer_close}")
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
            
            actions.append(action)
            contents.append(content)
            
        return actions, contents

