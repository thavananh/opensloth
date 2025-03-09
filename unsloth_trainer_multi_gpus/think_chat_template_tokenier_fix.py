def fix_think_chat_template_tokenizer(tokenizer):
    """
    Fix the tokenizer chat template to not remove think tags."""
    template_no_remove_think_tags = """{% if not add_generation_prompt is defined %}{% set add_generation_prompt =
    false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false,
    is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if
    message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%-
    endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in
    messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false
    -%}{{'<｜User｜>' + message['content']}}{%- endif %}{%- if message['role'] ==
    'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}{%-
    for tool in message['tool_calls']%}{%- if not ns.is_first
    %}{{'<｜Assistant｜><|tool_calls_begin|><|tool_call_begin|>' + tool['type'] +
    '<|tool_sep|>' + tool['function']['name'] + '\n' + '```json' + '\n' +
    tool['function']['arguments'] + '\n' + '```' + '<|tool_call_end|>'}}{%- set
    ns.is_first = true -%}{%- else %}{{'\n' + '<|tool_call_begin|>' + tool['type'] +
    '<|tool_sep|>' + tool['function']['name'] + '\n' + '```json' + '\n' +
    tool['function']['arguments'] + '\n' + '```' +
    '<|tool_call_end|>'}}{{'<|tool_calls_end|><|end_of_sentence|>'}}{%- endif %}{%-
    endfor %}{%- endif %}{%- if message['role'] == 'assistant' and
    message['content'] is not none %}{%- if ns.is_tool %}{{'<|tool_outputs_end|>' +
    message['content'] + '<|end_of_sentence|>'}}{%- set ns.is_tool = false -%}{%-
    else %}{{'<｜Assistant｜>' + message['content'] + '<|end_of_sentence|>'}}{%- endif %}{%- endif %}{%- if message['role']
    == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first
    %}{{'<|tool_outputs_begin|><|tool_output_begin|>' + message['content'] +
    '<|tool_output_end|>'}}{%- set ns.is_output_first = false %}{%- else
    %}{{'\n<|tool_output_begin|>' + message['content'] + '<|tool_output_end|>'}}{%-
    endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool
    %}{{'<|tool_outputs_end|>'}}{% endif %}{% if add_generation_prompt and not
    ns.is_tool %}{{'<｜Assistant｜><think>
    '}}{% endif %}"""

    tokenizer.chat_template = template_no_remove_think_tags
    return tokenizer