"""
Copyright 2024 Amazon.com, Inc. or its affiliates.  All Rights Reserved.
SPDX-License-Identifier: MIT-0
"""
import json
from langchain.prompts.prompt import PromptTemplate
from langchain_community.llms import Bedrock
from langchain_community.chat_models import BedrockChat
import logging
import os
from timeit import default_timer as timer

# Create the logger
DEFAULT_LOG_LEVEL = logging.NOTSET
DEFAULT_LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
log_level = os.environ.get('LOG_LEVEL')
match log_level:
    case '10':
        log_level = logging.DEBUG
    case '20':
        log_level = logging.INFO
    case '30':
        log_level = logging.WARNING
    case '40':
        log_level = logging.ERROR
    case '50':
        log_level = logging.CRITICAL
    case _:
        log_level = DEFAULT_LOG_LEVEL
log_format = os.environ.get('LOG_FORMAT')
if log_format is None:
    log_format = DEFAULT_LOG_FORMAT
elif len(log_format) == 0:
    log_format = DEFAULT_LOG_FORMAT
# Set the basic config for the logger
logging.basicConfig(level=log_level, format=log_format)


# Function to read the content of the specified file
def read_file(dir_name, file_name):
    logging.info('Reading content from file "{}"...'.format(file_name))
    with open(os.path.join(dir_name, file_name)) as f:
        content = f.read()
    logging.info('Completed reading content from file.')
    return content


# Function to write the content to the specified file
def write_file(dir_name, file_name, content):
    logging.info('Writing content to file "{}"...'.format(file_name))
    with open(os.path.join(dir_name, file_name), 'w') as f:
        f.write(content)
    logging.info('Completed writing content to file.')
    return content


# Function to prepare the prompt
def prepare_prompt(prompt_template_dir, prompt_template_file_name, **kwargs):
    prompt_template_file_path = os.path.join(prompt_template_dir, prompt_template_file_name)
    logging.info('Reading content from prompt template file "{}"...'.format(prompt_template_file_name))
    prompt_template = PromptTemplate.from_file(prompt_template_file_path)
    logging.info('Completed reading content from prompt template file.')
    logging.info('Substituting prompt variables...')
    prompt = prompt_template.format(**kwargs)
    logging.info('Completed substituting prompt variables.')
    return prompt


# Function to invoke the specified Claude 3 LLM through
# the LangChain ChatModel client and using the specified prompt
def invoke_claude_3(model_id, bedrock_rt_client, prompt):
    # Create the LangChain ChatModel client
    logging.info('Creating LangChain ChatModel client for LLM "{}"...'.format(model_id))
    llm = BedrockChat(
        model_id=model_id,
        model_kwargs={
            "temperature": 0,
            "max_tokens": 4000
        },
        client=bedrock_rt_client,
        streaming=False
    )
    logging.info('Completed creating LangChain ChatModel client for LLM.')
    logging.info('Invoking LLM "{}" with specified inference parameters "{}"...'.
                 format(llm.model_id, llm.model_kwargs))
    start = timer()
    prompt_response = llm.invoke(prompt).content
    end = timer()
    logging.info(prompt + prompt_response)
    logging.info('Completed invoking LLM.')
    logging.info('Prompt processing duration = {} second(s)'.format(end - start))
    return prompt_response


# Function to process the steps required for the prompt
def process_prompt(model_id, bedrock_rt_client, prompt_templates_dir, prompt_template_file, source_prompt):
    # Read the prompt template and perform variable substitution
    prompt = prepare_prompt(prompt_templates_dir, prompt_template_file,
                            SOURCE_PROMPT=source_prompt)
    # Invoke the LLM and print the response
    return invoke_claude_3(model_id, bedrock_rt_client, prompt)


# Function to generate the prompts for many models
def generate_common_prompt(decomposed_prompt):
    logging.info('Generating prompt...')
    decomposed_prompt = json.loads(decomposed_prompt)
    prompt = ''
    system_prompts = decomposed_prompt['systemPrompts']
    system_prompts_len = len(system_prompts)
    user_prompts = decomposed_prompt['userPrompts']
    user_prompts_len = len(user_prompts)
    if system_prompts_len > 0:
        # Loop through the system prompts
        for system_prompt in system_prompts:
            prompt += system_prompt + ' '
        prompt = prompt.strip(' ')
        prompt += '\n\n'
    if user_prompts_len > 0:
        # Loop through the user prompts
        for index, user_prompt in enumerate(user_prompts):
            context = user_prompt['context']
            input_data = user_prompt['inputData']
            output_indicator = user_prompt['outputIndicator']
            instruction = user_prompt['instruction']
            if len(context) > 0:
                prompt += context + '\n'
            if len(input_data) > 0:
                prompt += input_data + '\n'
            if len(output_indicator) > 0:
                prompt += output_indicator + ' '
            # Check if this is the last index
            if index == (user_prompts_len - 1):
                if len(instruction) > 0:
                    prompt += '\n\n' + instruction + '\n'
            else:
                if len(instruction) > 0:
                    prompt += instruction + '\n'
            prompt += '--\n'
        prompt = prompt.rstrip('--\n').rstrip(' ')
    logging.info('Completed generating prompt.')
    return prompt


# Function to generate the prompt for Amazon Titan Text models
def generate_amazon_titan_text_prompt(decomposed_prompt):
    return generate_common_prompt(decomposed_prompt)


# Function to generate the prompt for Anthropic Claude 2.x and older models
def generate_anthropic_claude_2_prompt(decomposed_prompt):
    logging.info('Generating prompt...')
    decomposed_prompt = json.loads(decomposed_prompt)
    prompt = '\n\n\nHuman: '
    system_prompts = decomposed_prompt['systemPrompts']
    system_prompts_len = len(system_prompts)
    user_prompts = decomposed_prompt['userPrompts']
    user_prompts_len = len(user_prompts)
    if system_prompts_len > 0:
        # Loop through the system prompts
        for system_prompt in system_prompts:
            prompt += system_prompt + ' '
        prompt = prompt.strip(' ')
        prompt += '\n\n'
    if user_prompts_len > 0:
        # Loop through the user prompts
        for index, user_prompt in enumerate(user_prompts):
            context = user_prompt['context']
            input_data = user_prompt['inputData']
            output_indicator = user_prompt['outputIndicator']
            instruction = user_prompt['instruction']
            if len(context) > 0:
                prompt += context + '\n'
            if len(input_data) > 0:
                prompt += input_data + '\n'
            if len(output_indicator) > 0:
                prompt += output_indicator + ' '
            # Check if this is the last index
            if index == (user_prompts_len - 1):
                if len(instruction) > 0:
                    prompt += '\n\n' + instruction + '\n'
            else:
                if len(instruction) > 0:
                    prompt += instruction + '\n'
            prompt += '\n'
        prompt = prompt.rstrip('\n').rstrip(' ')
        prompt += '\n\nAssistant: '
    logging.info('Completed generating prompt.')
    return prompt


# Function to format the parts of the prompts for
# the Messages API in Anthropic Claude 3 models
def format_anthropic_claude_3_message_parts(role, input_prompts, output_prompt):
    input_prompts_len = len(input_prompts)
    if role == 'system':
        if input_prompts_len > 0:
            system_prompt_content = ''
            for input_prompt in input_prompts:
                system_prompt_content += input_prompt + ' '
            system_prompt_content = system_prompt_content.strip(' ')
            system_prompt = {
                "system": system_prompt_content
            }
            output_prompt += '\n\t' + json.dumps(system_prompt).lstrip('{').rstrip('}') + ','
    else:
        for index, input_prompt in enumerate(input_prompts):
            non_system_prompt_content = (('Carefully read the context provided in the <CONTEXT> '
                                         + 'tag and the input data provided in the <INPUT_DATA> tag '
                                         + 'and then respond to the provided instruction. '
                                         + '<CONTEXT>{}</CONTEXT>. <INPUT_DATA>{}</INPUT_DATA>. {}')
                                         .format(input_prompt['context'],
                                                 input_prompt['inputData'],
                                                 input_prompt['instruction']))
            non_system_prompt = {
                "role": "user",
                "content": non_system_prompt_content
            }
            output_prompt += '\t\t' + json.dumps(non_system_prompt) + ',\n'
            if input_prompts_len > 1:
                non_system_prompt_content = input_prompt['instruction']
                non_system_prompt = {
                    "role": "assistant",
                    "content": non_system_prompt_content
                }
                output_prompt += '\t\t' + json.dumps(non_system_prompt) + ',\n'
    return output_prompt


# Function to generate the prompt for Anthropic Claude 3 Messages API
def generate_anthropic_claude_3_prompt(decomposed_prompt):
    logging.info('Generating prompt...')
    prompt = '{'
    decomposed_prompt = json.loads(decomposed_prompt)
    system_prompts = decomposed_prompt['systemPrompts']
    user_prompts = decomposed_prompt['userPrompts']
    prompt = format_anthropic_claude_3_message_parts('system', system_prompts, prompt)
    prompt += '\n\t"messages": [\n'
    prompt = format_anthropic_claude_3_message_parts('user', user_prompts, prompt)
    prompt = prompt.strip(',\n')
    prompt += '\n\t]\n}'
    logging.info('Completed generating prompt.')
    return prompt


# Function to generate the prompt for AI21 Labs Jurassic models
def generate_ai21_labs_jurassic_prompt(decomposed_prompt):
    return generate_common_prompt(decomposed_prompt)


# Function to generate the prompt for Cohere Command models
def generate_cohere_command_prompt(decomposed_prompt):
    return generate_common_prompt(decomposed_prompt)


# Function to generate the prompt for Meta LLAMA 2 models
def generate_meta_llama_2_prompt(decomposed_prompt):
    logging.info('Generating prompt...')
    decomposed_prompt = json.loads(decomposed_prompt)
    prompt = '<s>[INST] '
    system_prompts = decomposed_prompt['systemPrompts']
    system_prompts_len = len(system_prompts)
    user_prompts = decomposed_prompt['userPrompts']
    user_prompts_len = len(user_prompts)
    if system_prompts_len == 0:
        prompt += '\n'
    else:
        prompt += '<<SYS>>\n'
        # Loop through the system prompts
        for system_prompt in system_prompts:
            prompt += system_prompt + ' '
        prompt = prompt.strip(' ')
        prompt += '<</SYS>>\n\n'
    if user_prompts_len > 0:
        # Loop through the user prompts
        for index, user_prompt in enumerate(user_prompts):
            context = user_prompt['context']
            input_data = user_prompt['inputData']
            output_indicator = user_prompt['outputIndicator']
            instruction = user_prompt['instruction']
            if len(context) > 0:
                prompt += context + ' '
            if len(input_data) > 0:
                prompt += input_data + ' '
            if len(output_indicator) > 0:
                prompt += output_indicator + ' '
            # Check if this is the last index
            if index == (user_prompts_len - 1):
                if len(instruction) > 0:
                    prompt += instruction
                prompt += ' [/INST]'
            else:
                prompt += ' [/INST] {} </s><s>[INST] '.format(instruction)
    logging.info('Completed generating prompt.')
    return prompt


# Function to generate the prompt for Mistral AI models
def generate_mistral_ai_prompt(decomposed_prompt):
    logging.info('Generating prompt...')
    decomposed_prompt = json.loads(decomposed_prompt)
    prompt = '<s>[INST] '
    system_prompts = decomposed_prompt['systemPrompts']
    system_prompts_len = len(system_prompts)
    user_prompts = decomposed_prompt['userPrompts']
    user_prompts_len = len(user_prompts)
    # Loop through the system prompts
    for system_prompt in system_prompts:
        prompt += system_prompt + ' '
    if user_prompts_len > 0:
        # Loop through the user prompts
        for index, user_prompt in enumerate(user_prompts):
            context = user_prompt['context']
            input_data = user_prompt['inputData']
            output_indicator = user_prompt['outputIndicator']
            instruction = user_prompt['instruction']
            if len(context) > 0:
                prompt += context + ' '
            if len(input_data) > 0:
                prompt += input_data + ' '
            if len(output_indicator) > 0:
                prompt += output_indicator + ' '
            # Check if this is the last index
            if index == (user_prompts_len - 1):
                if len(instruction) > 0:
                    prompt += instruction
                prompt += ' [/INST]'
            else:
                prompt += ' [/INST] {} </s> [INST] '.format(instruction)
    logging.info('Completed generating prompt.')
    return prompt


# Convert decomposed prompts to the specified model
def convert_decomposed_prompt(target_model_name, target_prompts_dir, decomposed_source_prompt):
    # Generate the prompt, write to file and print it
    target_prompt_file_prefix = target_model_name.replace(' ', '_')
    logging.info('Converting prompt for {}:'.format(target_model_name))
    match target_model_name:
        case 'Amazon Titan Text':
            target_prompt = generate_amazon_titan_text_prompt(decomposed_source_prompt)
        case 'Anthropic Claude 2':
            target_prompt = generate_anthropic_claude_2_prompt(decomposed_source_prompt)
        case 'Anthropic Claude 3':
            target_prompt = generate_anthropic_claude_3_prompt(decomposed_source_prompt)
        case 'AI21 Labs Jurassic':
            target_prompt = generate_ai21_labs_jurassic_prompt(decomposed_source_prompt)
        case 'Cohere Command':
            target_prompt = generate_cohere_command_prompt(decomposed_source_prompt)
        case 'Meta LLAMA 2':
            target_prompt = generate_meta_llama_2_prompt(decomposed_source_prompt)
        case 'Mistral AI':
            target_prompt = generate_mistral_ai_prompt(decomposed_source_prompt)
    write_file(target_prompts_dir,
               '{}_zero_shot_prompt_1.txt'.format(target_prompt_file_prefix),
               target_prompt)
    logging.info('Converted prompt:\n{}'.format(target_prompt))
