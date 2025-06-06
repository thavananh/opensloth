import sys
import argparse
from loguru import logger
import json
import random
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, PreTrainedTokenizerFast
import os
from typing import List, Dict, Any, Optional
from datasets import Dataset

USE_LEN = True

# Global tokenizer instance, initialized in main()
tokenizer: PreTrainedTokenizerFast

def get_item_lengths(items: List[Dict[str, Any]]) -> List[int]:
    """
    Calculate token lengths for a list of items.
    Returns a list of integers representing the length of each item.
    """
    lengths: List[int] = []
    
    for item in items:
        messages_list: Optional[List[Dict[str, Any]]] = None
        
        if 'messages' in item and isinstance(item.get('messages'), list):
            messages_list = item['messages']
        elif 'conversations' in item and isinstance(item.get('conversations'), list):
            messages_list = item['conversations']
        
        if not messages_list:
            lengths.append(0)
            continue

        try:
            if USE_LEN:
                # Use character length instead of tokenization
                total_chars = 0
                for message in messages_list:
                    if isinstance(message, dict) and 'content' in message:
                        content = message.get('content', '')
                        if isinstance(content, str):
                            total_chars += len(content)
                lengths.append(total_chars)
            else:
                formatted_text: str = tokenizer.apply_chat_template(
                    messages_list,
                    tokenize=False, 
                    add_generation_prompt=False
                )
                lengths.append(len(tokenizer.encode(formatted_text)))
        except Exception as e:
            logger.warning(f'Error processing item. Setting length to 0. Error: {e}')
            lengths.append(0)
    
    return lengths

def pack_by_indices(
    lengths: List[int], 
    max_seq_len: int, 
    shuffle_items: bool = True
) -> List[List[int]]:
    """
    Pack items by their indices based on lengths.
    Returns a list of lists, where each inner list contains indices 
    of items that should be packed together.
    """
    if not lengths:
        logger.info('No lengths provided to packing function.')
        return []

    # Create indices and optionally shuffle
    indices = list(range(len(lengths)))
    if shuffle_items:
        random.shuffle(indices)

    # Create batches of indices based on max_seq_len
    batches: List[List[int]] = []
    current_batch: List[int] = []
    current_batch_len: int = 0
    ignored_count: int = 0

    for idx in indices:
        item_len = lengths[idx]
        
        if item_len > max_seq_len:
            ignored_count += 1
            continue

        if not current_batch or (current_batch_len + item_len <= max_seq_len):
            current_batch.append(idx)
            current_batch_len += item_len
        else:
            batches.append(current_batch)
            current_batch = [idx]
            current_batch_len = item_len

    if current_batch:
        batches.append(current_batch)
    
    logger.info(f'Total batches created: {len(batches)}')
    logger.info(f'Total items considered: {len(lengths)}')
    if ignored_count > 0:
        skip_percentage = (ignored_count / len(lengths)) * 100
        logger.warning(f'Items ignored (too long): {ignored_count} ({skip_percentage:.2f}%)')
    else:
        logger.info('No items were skipped')

    # Plot distribution if we have results
    if batches:
        batch_lengths = [sum(lengths[idx] for idx in batch) for batch in batches]
        total_tokens = sum(batch_lengths)
        logger.info(f'Total tokens in packed data: {total_tokens}')
        
        try:
            plt.figure(figsize=(10, 6))
            num_bins = min(30, len(set(batch_lengths)) // 2 if len(set(batch_lengths)) > 1 else 1)
            num_bins = max(1, num_bins)
            plt.hist(batch_lengths, bins=num_bins, alpha=0.7)
            plt.title('Distribution of Packed Sequence Lengths')
            plt.xlabel('Total Length')
            plt.ylabel('Frequency')
            dist_path = '/tmp/packed_lengths_distribution.png'
            plt.savefig(dist_path)
            plt.close()
            logger.info(f'Saved distribution plot to {dist_path}')
        except Exception as e:
            logger.error(f'Failed to generate plot: {e}')
            
    return batches

def create_packed_items(
    original_items: List[Dict[str, Any]], 
    packed_indices: List[List[int]],
    lengths: List[int]
) -> List[Dict[str, Any]]:
    """
    Create final packed items by concatenating messages based on indices.
    """
    packed_items: List[Dict[str, Any]] = []
    
    for batch_indices in packed_indices:
        if not batch_indices:
            continue
            
        all_messages: List[Dict[str, Any]] = []
        total_len = 0
        
        for idx in batch_indices:
            item = original_items[idx]
            total_len += lengths[idx]
            
            # Get messages from item
            messages_list: Optional[List[Dict[str, Any]]] = None
            if 'messages' in item and isinstance(item.get('messages'), list):
                messages_list = item['messages']
            elif 'conversations' in item and isinstance(item.get('conversations'), list):
                messages_list = item['conversations']
            
            if messages_list:
                all_messages.extend(messages_list)
        
        packed_items.append({
            'messages': all_messages,
            'total_len': total_len
        })
    
    return packed_items

def main():
    global tokenizer 

    parser = argparse.ArgumentParser(description='Fast data packing script')
    parser.add_argument('-i', '--input', required=True, help='Input JSON file path')
    parser.add_argument('-o', '--output', required=True, help='Output JSON file path')
    parser.add_argument('--seq_len', type=int, default=4096, help='Max sequence length')
    parser.add_argument('--workers', type=int, default=1, help='Number of workers')
    parser.add_argument('--tokenizer_name', type=str, default='unsloth/gemma-3-27b-it', help='Tokenizer name')
    parser.add_argument('--log_level', type=str, default='INFO', help='Logging level')
    args = parser.parse_args()

    logger.remove() 
    logger.add(sys.stderr, level=args.log_level.upper())

    # Load tokenizer
    logger.info(f'Loading tokenizer: {args.tokenizer_name}')
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        sys.exit(1)

    # Load data
    logger.info(f'Loading data from: {args.input}')
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            raw_data: List[Any] = json.load(f)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        sys.exit(1)
        
    if not isinstance(raw_data, list):
        logger.error("Input must be a list")
        sys.exit(1)

    # Filter for dict items only
    items = [item for item in raw_data if isinstance(item, dict)]
    logger.info(f'Processing {len(items)} dictionary items')

    if not items:
        logger.error("No valid items to process")
        sys.exit(1)

    # Step 1: Get lengths
    logger.info('Calculating lengths...')
    # if args.workers > 1:
    #     dataset = Dataset.from_list(items)
    #     processed_dataset = dataset.map(
    #         lambda x: {'length': get_item_lengths([x])[0]},
    #         num_proc=args.workers,
    #         desc='Calculating lengths'
    #     )
    #     lengths = [item['length'] for item in processed_dataset]
    # else:
    #     lengths = get_item_lengths(items)

    # Step 2: Get packing indices
    logger.info('Packing by indices...')
    def get_len(messages: List[Dict[str, Any]]) -> int:
        if USE_LEN:
            return sum(len(msg.get('content', '')) for msg in messages if isinstance(msg, dict) and 'content' in msg)
        else:
            formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            return len(tokenizer.encode(formatted_text))
    from tqdm import tqdm
    lengths = [get_len(item['messages']) for item in tqdm(items, desc='Calculating lengths')]
    packed_indices = pack_by_indices(lengths, args.seq_len, shuffle_items=True)

    # Step 3: Create final packed items
    logger.info('Creating packed items...')
    packed_items = create_packed_items(items, packed_indices, lengths)

    # Save results
    logger.info(f'Saving {len(packed_items)} packed items to: {args.output}')
    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(packed_items, f, ensure_ascii=False, indent=4)
    except Exception as e:
        logger.error(f"Error saving output: {e}")
        sys.exit(1)

    logger.info('Script finished successfully.')

if __name__ == '__main__':
    main()
