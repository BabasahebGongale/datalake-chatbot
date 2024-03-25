
import streamlit as st
from PIL import Image
from langchain.memory import ConversationBufferMemory

#from langchain.callbacks import StreamlitCallbackHandler
import sql_generator as gen 
import footer as ft 
import os
#st.cache_data.clear()

# Setup memory
answer_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

#page_icon = Image.open("../images/airplane.png",'r')
images = Image.open("images/gen_ai_banner.jpg",'r')

# page style
chat_input = "Hi! I'm Cognizant AI DB Analyst. How can I help?"
chat_placeholder = "Ask me about house!"
error_message = "Issues with the LLM. Try clearing the message history."
page_title = "Talk To Data Lake"
logo_width = 500
st.set_page_config(page_title=page_title)
st.title(page_title)
st.image(images, width=int(logo_width))


# app
if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role": "assistant", "content": chat_input}]
    
    
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


user_query = st.chat_input(placeholder=chat_placeholder)   




if 'chat_history' not in st.session_state:
    st.session_state.chat_history=[]
else:
    for message in st.session_state.chat_history:
        #query_memory.save_context({'input':message['Human']},{'output':message['Assistant']})
        answer_memory.save_context({'input':message['Human']},{'output':message['Assistant']})

if user_query:
    #message={'Human':user_query,'Assistant':response}
    #st.session_state.chat_history.append(message)
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)
    
    with st.chat_message("assistant"):
       #st_cb = StreamlitCallbackHandler(st.container())
        try:
            prompt = f"\n\nHuman: {user_query}\n\nAssistant:"
            #response = agent.run(prompt, callbacks=[st_cb])
                   
            response,genSqlQuery,result = gen.run_query(user_query)
            
            if user_query:
                message={'Human':user_query,'Assistant':response}
                st.session_state.chat_history.append(message)
            print('the response on UI :', response)

        except Exception as e:
            print(e)
            st.warning(
                error_message
            )
        else:
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)
            with st.sidebar:
               # with st.echo():
                    #st.write(genSqlQuery)
                    container = st.container(border=True)
                    container.write("SQL Query : "+genSqlQuery)
                    container.write(result)
            # Now insert some more in the container
            
           
ft.footer()    

