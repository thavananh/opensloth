# COPY from unsloth_zoo/dataset_utils.py

def _longest_common_sublist(lists):
    """
    Finds the longest common sublist among multiple lists.

    Parameters:
    lists (List[List[int]]): A list of lists.

    Returns:
    List[int]: The longest common sublist. If multiple sublists have the same maximum length,
               one of them is returned. If there's no common sublist, an empty list is returned.
    """
    if not lists: return []

    # Find the minimum length among all lists
    min_len = min(len(lst) for lst in lists)
    if min_len == 0: return []

    def has_common_sublist(length):
        """
        Checks if there's a common sublist of the given length across all lists.

        Returns:
        (bool, List): Tuple of whether such a sublist exists and the sublist itself.
        """
        common = set()
        first = lists[0]
        # Generate all possible sublists of the given length from the first list
        for i in range(len(first) - length + 1):
            sub = tuple(first[i:i + length])
            common.add(sub)
        pass

        # Iterate over the remaining lists and retain only the common sublists
        for lst in lists[1:]:
            current = set()
            for i in range(len(lst) - length + 1):
                sub = tuple(lst[i:i + length])
                if sub in common:
                    current.add(sub)
            common = current
            if not common:
                return False, []
        pass
        
        # If common is not empty, return one of the common sublists
        return True, list(common.pop())
    pass

    left, right = 1, min_len
    result = []

    while left <= right:
        mid = left + (right - left) // 2
        exists, sublist = has_common_sublist(mid)
        if exists:
            result = sublist  # Update result with the latest found sublist
            left = mid + 1    # Try to find a longer sublist
        else:
            right = mid - 1   # Try with a shorter length
    pass

    return result


def _find_common_token_ids(component, tokenizer, force_match = False):
    """
    \n### User:\n\n
    \n\n### User:\n\n
    etc
    we need to find the middle most repeatted part.
    Tokenizers can tokenize newlines or spaces as 1 token!
    """
    right_text = ""
    if   component.endswith (" "): right_text = " "
    elif component.endswith("\n"): right_text = "\n"
    left_text = ""
    if   component.startswith (" "): left_text = " "
    elif component.startswith("\n"): left_text = "\n"
    stripped = component.strip()
    
    # Add current pieces and also newlines
    all_input_ids = []
    if not force_match:
        for left in range(3):
            for right in range(3):
                x = left*left_text + stripped + right*right_text
                x = tokenizer(x, add_special_tokens = False).input_ids
                all_input_ids.append(x)

                x = left*"\n" + stripped + right*"\n"
                x = tokenizer(x, add_special_tokens = False).input_ids
                all_input_ids.append(x)
            pass
        pass
    else:
        x = tokenizer(component, add_special_tokens = False).input_ids
        all_input_ids.append(x)
    pass

    # Old longest common substring is replaced with actual longest common list of numbers
    # substring = _old_longest_common_substring([str(x + [0]) for x in all_input_ids])
    # substring = substring.split(", ")[:-1]
    # substring = [int(x) for x in substring if x.isdigit()]
    substring = _longest_common_sublist([x + [0] for x in all_input_ids])

    # If substring is simply [0], this might be just the original single token
    # Fixes https://github.com/unslothai/unsloth/issues/1290
    # Mistral [INST] [/INST] singular tokens breaks since we output [0] but we need [3] [4]
    if substring == [0] and len(all_input_ids[0]) == 1:
        single_token = all_input_ids[0][0]
        # Confirm single token in every single possible match
        if all(single_token in x for x in all_input_ids):
            substring = [single_token]
    pass

    # Also if substring is original input_ids + [0], then leave it as the original one
    # This happens when no newlines / spaces are used in chat template
    # Eg Phi-4 does not use newlines or spaces
    if (len(set(str(x) for x in all_input_ids)) == 1) and \
        (len(all_input_ids[0]) + 1 == len(substring)) and \
        (all_input_ids[0] == substring[:-1]):

        # Use original un-changed substring
        substring = all_input_ids[0]
    pass
    
    # Also get rest of tokenized string
    original = tokenizer(component, add_special_tokens = False).input_ids
    # Get optional left and right
    for j in range(len(original)):
        if original[j : j + len(substring)] == substring: break
    optional_left  = original[:j]
    optional_right = original[j+len(substring):]
    return substring, optional_left, optional_right


