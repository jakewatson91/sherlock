import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_cohere import ChatCohere
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

# Deepseek V3
deepseekv3_llm = ChatOpenAI(
    model_name='deepseek-chat', 
    openai_api_key=os.getenv('DEEPSEEK_API_KEY'), 
    openai_api_base='https://api.deepseek.com',
    max_tokens=128
)

# Deepseek R1 Distilled
deepseekr1_model = HuggingFaceEndpoint(repo_id='deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
                    #  model_kwargs={"temperature": 0.1, "max_length": 2048, "do_sample": True},
                     huggingfacehub_api_token=os.getenv('HF_API_KEY')
    )
deepseekr1_llm = ChatHuggingFace(llm=deepseekr1_model)

# Cohere - 
cohere_llm = ChatCohere(api_key=os.getenv('COHERE_TRIAL_KEY'), model='command-r', max_tokens=128) # TRIAL limited, swap to COHERE_API_KEY for paid

# OpenAI
openai_llm = ChatOpenAI(verbose=True, temperature=0, model_name="gpt-3.5-turbo", max_tokens=128)