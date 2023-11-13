import streamlit as st

from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMMathChain
from langchain.agents import Tool
import time

from tools import get_tools



st.set_page_config(
    page_title="Garf Knows Everything", page_icon="🐈", initial_sidebar_state="collapsed"
)


st.header("",divider='orange')
st.title(":orange[_Garf_] 🐈 knows everything!")
st.header("",divider='orange')

with st.form("my_form"):
    user_input = st.text_area("Ask me anything about everything! ", "What is the latest version of iOS? What is it raised to 0.43 power?")
    submitted = st.form_submit_button("Submit")


openai_api_key = ''
openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')

if not openai_api_key.startswith('sk-'):
    st.info("Please add your OpenAI API key in the sidebar to continue.")
    st.stop()

llm = OpenAI(temperature=0, streaming=True, openai_api_key=openai_api_key)


llm_math_chain = LLMMathChain.from_llm(llm)

math_tool =  Tool(
            name="Calculator",
            func=llm_math_chain.run,
            description="Useful for when you need to answer questions about math.",
        )

if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history")


if not openai_api_key:
    st.info("Please add your OpenAI API key in the sidebar to continue.")
    st.stop()


tools = get_tools()
tools.append(math_tool)

garf = initialize_agent(
    agent='conversational-react-description',
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=7,
    memory = st.session_state.memory
)


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if submitted:
    user_container = st.empty()
    user_container = user_container.container()
    user_container.chat_message("user").write(user_input)
        
    st.session_state.messages.append({"role": "user", "content": user_input})

    
    with st.chat_message("assistant", avatar="😺"):
        callback = StreamlitCallbackHandler(st.container())
        answer = garf.run(user_input, callbacks=[callback])
        answer_container = st.empty()
    
        complete_answer = ""
        for word in answer.split():
            complete_answer += word + " "
            time.sleep(0.15)
            answer_container.info(complete_answer + "▌")
     
        answer_container.success(complete_answer, icon="🍵")   
        
    if 'python' in user_input.lower():
        st.code(answer, language='python')

        
    st.toast('Ready!', icon='🐾')
    st.session_state.messages.append({"role": "assistant", "content": complete_answer})
    time.sleep(5)
        
