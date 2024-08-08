## Convert GPT text prompts to text prompts for LLMs on Amazon Bedrock

This repository contains samples to convert GPT text prompts to text prompts for Large Language Models (LLMs) on Amazon Bedrock.

### Overview

A [prompt](https://en.wikipedia.org/wiki/Prompt_engineering) to a [Large Language Model (LLM)](https://en.wikipedia.org/wiki/Large_language_model) is an instruction provided to it to produce an output.  There are many text [prompting techniques](https://www.promptingguide.ai/techniques).

This sample will walk you through examples of converting [GPT](https://en.wikipedia.org/wiki/Generative_pre-trained_transformer) text prompts to text prompts for LLMs available on [Amazon Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock.html).

We will use the following process for prompt conversion,

* **Step 1:** Decompose the GPT text prompt into [prompt elements](https://www.promptingguide.ai/introduction/elements) using [Anthropic Claude 3](https://www.anthropic.com/news/claude-3-family) or [Anthropic Claude 3.5 Sonnet](https://www.anthropic.com/news/claude-3-5-sonnet) available on Amazon Bedrock.  These constructs are,
    * Context
    * Input Data
    * Output Indicator
    * Instruction
* **Step 2:** Parse these constructs using a Python script and create the text prompts for the various LLMs available on Amazon Bedrock.

We will cover the following types of prompts,

1. [Zero-Shot Prompting](https://www.promptingguide.ai/techniques/zeroshot)
2. [Few-Shot Prompting](https://www.promptingguide.ai/techniques/fewshot)
3. [Chain-of-Thought Prompting](https://www.promptingguide.ai/techniques/cot)
4. [Self-Consistency](https://www.promptingguide.ai/techniques/consistency)
5. [Generated Knowledge Prompting](https://www.promptingguide.ai/techniques/knowledge)

Note:

* These notebooks should only be run from within an [Amazon SageMaker Notebook instance](https://docs.aws.amazon.com/sagemaker/latest/dg/nbi.html) or within an [Amazon SageMaker Studio Notebook](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated.html)
* These notebooks use text based models along with their versions that were available on Amazon Bedrock at the time of writing. Update these as required.
* At the time of this writing, Amazon Bedrock was only available in [these supported AWS Regions](https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-regions.html). If you are running these notebooks from any other AWS Region, then you have to change the Amazon Bedrock client's region and/or endpoint URL parameters to one of those supported AWS Regions. Follow the guidance in the *Organize imports* section of the individual notebooks.
* On an Amazon SageMaker Notebook instance, these notebooks are recommended to be run with a minimum instance size of *ml.t3.medium* and with *Amazon Linux 2, Jupyter Lab 3* as the platform identifier.
* On an Amazon SageMaker Studio Notebook, these notebooks are recommended to be run with a minimum instance size of *ml.t3.medium* and with *Data Science 3.0* as the image.
* At the time of this writing, the most relevant latest version of the Kernel for running these notebooks were *conda_python3* on an Amazon SageMaker Notebook instance or *Python 3* on an Amazon SageMaker Studio Notebook.

### Disclaimer

This prompt conversion process has been provided to accelerate your migration from GPT to LLMs on Amazon Bedrock. You must review the converted prompts for accuracy, optimize and test them before using in your projects.

### Repository structure

This repository contains

* [A Jupyter Notebook](https://github.com/aws-samples/convert-gpt-text-prompts-to-amazon-bedrock-llm-prompts/blob/main/notebooks/llm_text_prompt_converter.ipynb).

* [A Python script](https://github.com/aws-samples/convert-gpt-text-prompts-to-amazon-bedrock-llm-prompts/blob/main/notebooks/scripts/helper_functions.py) that contains helper functions.

* [A prompt templates folder](https://github.com/aws-samples/convert-gpt-text-prompts-to-amazon-bedrock-llm-prompts/blob/main/notebooks/prompt_templates) that contains the prompts for Anthropic Claude 3 and 3.5 to perform Step 1 of the prompt conversion.

* [A source prompts folder](https://github.com/aws-samples/convert-gpt-text-prompts-to-amazon-bedrock-llm-prompts/blob/main/notebooks/source_prompts/gpt) containing example GPT prompts for the various prompt types mentioned above.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.
