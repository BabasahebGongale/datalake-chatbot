# Import Required Library
import json
import boto3
from typing import Dict
import time
import sqlalchemy
from sqlalchemy import create_engine
from langchain.chains import LLMChain
#from langchain.llms import LLMContentHandler
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts.prompt import PromptTemplate
from pydantic import BaseModel, root_validator
from langchain.sql_database import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts  import PromptTemplate
from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.llms import SagemakerEndpoint
from langchain.llms.bedrock import Bedrock
import pandas as pd

client = boto3.client('glue')
region=client.meta.region_name
# Connect to S3 using Athena

glue_db_name = "housingdb"
glue_databucket_name = "athena-housing-dataset-poc"

# Create the athena SQLAlchemy engine

def get_athena_connection_string(glue_db_name,glue_databucket_name):
    connathena = f"athena.{region}.amazonaws.com" 
    portathena = '443' #Update, if port is different
    schemaathena = glue_db_name #from user defined params
    s3stagingathena = f's3://{glue_databucket_name}/results/'#from cfn params
    wkgrpathena = 'primary'#Update, if workgroup is different
    # Create the athena connection string
    connection_string = f"awsathena+rest://@{connathena}:{portathena}/{schemaathena}?s3_staging_dir={s3stagingathena}/&work_group={wkgrpathena}"
    return connection_string


def get_athen_connection():
    engine_athena = create_engine(get_athena_connection_string(glue_db_name,glue_databucket_name), echo=False,pool_size=10, max_overflow=20)
    athena_connection = engine_athena.connect()
    #print("----Athena connection----\n",athena_connection)
    return athena_connection
    


glue_data_catalog = [glue_db_name]

def get_athena_db(glue_db_name,glue_databucket_name):
    # Athena variables
    dbathena = SQLDatabase.from_uri(get_athena_connection_string(glue_db_name,glue_databucket_name))
    #print('Athen db :', dbathena)
    return dbathena

# AWS Glue set Up
def parse_catalog():
    #Connect to Glue catalog
    #get metadata of redshift serverless tables
    columns_str=''
    #define glue cient
    glue_client = boto3.client('glue')
    try:
        for db in glue_data_catalog:
            response = glue_client.get_tables(DatabaseName =db)
            for tables in response['TableList']:
                #classification in the response for s3 and other databases is different. Set classification based on the response location
                if tables['StorageDescriptor']['Location'].startswith('s3'):  classification='s3' 
                else:  classification = tables['Parameters']['classification']
                for columns in tables['StorageDescriptor']['Columns']:
                        dbname,tblname,colname=tables['DatabaseName'],tables['Name'],columns['Name']
                        columns_str=columns_str+f'\n|{dbname}|{tblname}|{colname}'                     
        #API
        ## Append the metadata of the API to the unified glue data catalog
       # columns_str=columns_str+'\n'+('api|meteo|weather|weather')
    except Exception as e:
        print('Error in parse catalog method :', e)
    return columns_str

glue_catalog = parse_catalog()

#display a few lines from the catalog
#print('\n'.join(glue_catalog.splitlines()[-40:]) )



inference_modifier = {'max_tokens_to_sample':4096, 
                      "temperature":0,
                      "top_k":250,
                      "top_p":1,
                      "stop_sequences": ["\n\nHuman"]
                     }


def get_bedrock_client():
    bedrock_client = boto3.client(service_name='bedrock-runtime',
                       region_name='us-west-2',
                       endpoint_url='https://bedrock-runtime.us-west-2.amazonaws.com')
    #print("------get_bedrock_client----- ")
    #print(bedrock_client)
    return bedrock_client

def get_textgen_llm():
    
    textgen_llm = Bedrock(model_id = "anthropic.claude-v2",
                    client =get_bedrock_client(), 
                    model_kwargs = inference_modifier 
                    )
    
    #print("------get_textgen_llm----- ")
    
    #print(textgen_llm)
    return textgen_llm
    
print('the bedrock client :', get_bedrock_client())


channel = ""

def get_channel():
    return channel

def set_channel(db):
    channel =  db
    return channel

### Test Data
#Function 1 'Infer Channel'
#define a function that infers the channel/database/table and sets the database for querying
dbdf = ""
def identify_channel(query):
    db = {}
    bedrock_client = get_bedrock_client()
    #Prompt 1 'Infer Channel'
    ##set prompt template. It instructs the llm on how to evaluate and respond to the llm. It is referred to as dynamic since glue data catalog is first getting generated and appended to the prompt.
    multi_var_prompt = PromptTemplate(
        input_variables=["query", "glue_catalog"],
        template="""Human: From the table below, find the database (in column database) which will contain the data (in corresponding 
            column_names) to answer the question {query} \n
            {glue_catalog}
            Give your answer as database ==
            Also,give your answer as database.table ==
            Based on the table schema {glue_catalog} below, write a SQL query that would answer the user's answer the question: {query}

            Question: {query}
            SQL Query :
            Display generated SQLQuery as SQLGenerated and enclosed SQLGenerated in square brackets: \n
            Assistant:""" )

    # define llm chain
    llm_chain = multi_var_prompt.format(query=query,glue_catalog=glue_catalog)
    #run the query and save to generated texts
    textgen_llm = get_textgen_llm()
    response = textgen_llm(llm_chain)
    generated_texts = response[response.index('\n')+1:]
   # print('identified channel:', generated_texts)

    #set the channel from where the query can be answered
 #   if 'database' in generated_texts: 

    result = getSqlQuery(generated_texts)
    
    if 'housingdb' in generated_texts:
            set_channel("db")
            #channel='db'
           # print("SET database to athena")
    elif 'api' in generated_texts: 
            channel='api'
           # print("SET database to weather api")        
    else: raise Exception("User question cannot be answered by any of the channels mentioned in the catalog")
    #print("Step complete. Channel is: ", channel)
    
    return result

#Function 2 'Run Query'
#define a function that infers the channel/database/table and sets the database for querying
def run_query(query):
    result,genSqlQuery= identify_channel(query)
    db=get_athena_db(glue_db_name,glue_databucket_name)
    channel = get_channel()
    #call the identify channel function first
    try:
        
        #  input=query
        table_info=db.get_table_info()
        dialect=db.dialect
        response = result
        #print('the result -------->',response)
        ##Prompt 2 'Run Query'
        #after determining the data channel, run the Langchain SQL Database chain to convert 'text to sql' and run the query against the source data channel. 
        #provide rules for running the SQL queries in default template--> table info.

        multi_var_prompt = PromptTemplate(
        input_variables=["response","query"],
        template="""Human: Based on the SQLResult{response}  write a natural language response:
        SQL Response: {response}
        use {query} as reference to write a response.

        Assistant:""" )

        llm_chain = multi_var_prompt.format(response=response,query=query)
        #run the query and save to generated texts
        textgen_llm = get_textgen_llm()
        response =  textgen_llm(llm_chain)
        res_texts = response[response.index('\n')+1:]
        #print('generated_texts -------------->:', generated_texts)
        set_channel("")
    except Exception as e:
        Exception("Please rephrase question in context with database", e)
    
    return res_texts,genSqlQuery,result
        

        
#Try to get the generated SQL from exception 
def getSqlQuery(res):
    #print('SQL getSqlQuery---->',res)
    frm = str(res).find("[")
    to = str(res).find("]")
    
    genSqlQuery = str(res)[frm+1:to]
    res = invokeAthena(genSqlQuery)

    print('Finale output for query -------->',res)
    return res,genSqlQuery

import awswrangler as wr

def invokeAthena(genSqlQuery): 
    #df = pd.read_sql(str(genSqlQuery), get_athen_connection())
    
    df_athena = wr.athena.read_sql_query(str(genSqlQuery), database=glue_db_name)
    return df_athena
    #print(df)
    #return df

def getResponse(df):
    return df

def getDataframe():
    return dbdf 


#Response from Langchain
# query = "get the details of most expensive house"
# response = run_query(query)
# print("----------------------------------------------------------------------")
# print(f'Q: {query}  \nA: {response}')
    
# import gradio as gr
# import os
# import io

# # result , response  = run_query("SELECT * FROM housingdb.poc_housing_dataset WHERE area < 2000")
# # print(response.strip())
# # print(result.strip())

# def summarize(input):
#     response  = run_query(input)
#     return response.strip()

# # def dbresult():
# #     #output = qa.run(input)
# #     result  = getDataframe()
# #     return result.strip()

# gr.close_all()
# demo = gr.Interface(fn=summarize, 
#                     inputs=[gr.Textbox(label="Please write question", lines=6)],
#                     outputs=[gr.Textbox(label="Here is Answer", lines=3)],
#                     title="Question & answer bot for your database",
#                     description="GenAI solution for SQL queries"
#                    )

# #demo.launch(share=True, server_port=int(os.environ['PORT2']))
# demo.launch(share=True)

# #response = run_query(query)



# import streamlit as st
# from PIL import Image

# #from langchain.callbacks import StreamlitCallbackHandler
# import os
# #st.cache_data.clear()


# #page_icon = Image.open("../images/airplane.png",'r')
# image = Image.open("images/gen_ai_banner.jpg",'r')
# # page style
# chat_input = "Hi! I'm an IAG Flight Analyst. How can I help?"
# chat_placeholder = "Ask me about flights!"
# error_message = "Issues with the LLM. Try clearing the message history."

# page_title = "Talk To Housing Database"
# st.set_page_config(page_title=page_title)
# st.title(page_title)
# #st.image(image, width=int(logo_width))

# # app
# if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
#     st.session_state["messages"] = [{"role": "assistant", "content": chat_input}]
    
# # for msg in st.session_state.messages:
# #     st.chat_message(msg["role"]).write(msg["content"])


# user_query = st.chat_input(placeholder=chat_placeholder)    

# if 'chat_history' not in st.session_state:
#     st.session_state.chat_history=[]
# else:
#     for message in st.session_state.chat_history:
#         #query_memory.save_context({'input':message['Human']},{'output':message['Assistant']})
#         answer_memory.save_context({'input':message['Human']},{'output':message['Assistant']})

# if user_query:
#     #message={'Human':user_query,'Assistant':response}
#     #st.session_state.chat_history.append(message)
#     st.session_state.messages.append({"role": "user", "content": user_query})
#     st.chat_message("user").write(user_query)
    
#     with st.chat_message("assistant"):
#        #st_cb = StreamlitCallbackHandler(st.container())
#         try:
#             prompt = f"\n\nHuman: {user_query}\n\nAssistant:"
#             #response = agent.run(prompt, callbacks=[st_cb])
                   
#             response = run_query(user_query)
#             if user_query:
#                 message={'Human':user_query,'Assistant':response}
#                 st.session_state.chat_history.append(message)
#             print('the response on UI :', response)

#         except Exception as e:
#             print(e)
#             st.warning(
#                 error_message
#             )
#         else:
#             st.session_state.messages.append({"role": "assistant", "content": response})
#             st.write(response)