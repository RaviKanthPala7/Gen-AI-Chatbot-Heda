import os
import boto3
from langchain_aws import BedrockLLM
from langchain.llms.bedrock import Bedrock
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

bedrock_client = boto3.client(service_name="bedrock-runtime", region_name="ap-northeast-1")

def demo_chatbot():
    # demo_llm = BedrockLLM(
    #     credentials_profile_name='admin',
    #     model_id='anthropic.claude-v2:1',
    #     region_name='ap-northeast-1',
    #     model_kwargs={
    #         "temperature": 0.9,
    #         "top_p": 0.5
    #     }   
    # )
    # return demo_llm
    llm = Bedrock(model_id="anthropic.claude-v2:1", client=bedrock_client, model_kwargs={'max_tokens_to_sample':512})
    return llm
    
#     return demo_llm.invoke(input_text) 
#

# response = demo_chatbot('What is the temperature in London like?')
# print(response)

def demo_memory():
    llm_data=demo_chatbot()
    memory=ConversationBufferMemory(llm=llm_data, max_token_limit=512)
    return memory

def demo_conversation(input_text,memory):
    llm_chain_data=demo_chatbot()
    llm_conversation=ConversationChain(llm=llm_chain_data,memory=memory,verbose=True)

    chat_reply=llm_conversation.predict(input=input_text)
    return chat_reply


