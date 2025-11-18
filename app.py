import streamlit as st
# ✅ CHANGED: Added ChatGoogleGenerativeAI for Gemini support
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os

st.set_page_config(
    page_title="Omni-Bot - Your Intelligent AI Companion",
    page_icon="🤖",
    layout="wide"
)

# ✅ CHANGED: Added GOOGLE_API_KEY for Gemini alongside existing keys
groq_api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
google_api_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
langchain_api_key = st.secrets.get("LANGCHAIN_API_KEY", os.getenv("LANGCHAIN_API_KEY", ""))

if langchain_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key

if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'code_history' not in st.session_state:
    st.session_state.code_history = []
if 'current_mode' not in st.session_state:
    st.session_state.current_mode = "Brainy Buddy"

with st.sidebar:
    st.title("🤖 Omni-Bot Settings")
    
    # ✅ CHANGED: New AI Provider selector dropdown
    st.subheader("🤖 AI Provider")
    ai_provider = st.selectbox(
        "Select AI Provider",
        ["Google Gemini", "Groq"],
        help="Choose between Google Gemini or Groq API"
    )
    
    # ✅ CHANGED: Dynamic API status based on selected provider
    st.subheader("🔑 API Status")
    if ai_provider == "Google Gemini":
        if google_api_key:
            st.success("✅ Gemini API: Configured")
        else:
            st.warning("🔶 Demo Mode - Add GOOGLE_API_KEY")
    else:  # Groq
        if groq_api_key:
            st.success("✅ Groq API: Configured")
        else:
            st.warning("🔶 Demo Mode - Add GROQ_API_KEY")
    
    mode = st.selectbox("Choose Mode", ["Brainy Buddy", "CodeCraft"])
    st.session_state.current_mode = mode
    
    # ✅ CHANGED: Provider-specific model configuration
    if ai_provider == "Google Gemini" and google_api_key:
        # Fixed Gemini model (as per your request)
        engine = "gemini-2.5-flash"
        st.info("📦 Model: gemini-2.5-flash")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
        max_tokens = st.slider("Max Tokens", 50, 2000, 512, 50)
    elif ai_provider == "Groq" and groq_api_key:
        # Groq model dropdown
        engine = st.selectbox("Model", ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"])
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
        max_tokens = st.slider("Max Tokens", 50, 2000, 512, 50)
    else:
        # Demo mode defaults
        engine = "gemini-2.5-flash" if ai_provider == "Google Gemini" else "llama-3.1-8b-instant"
        temperature = 0.7
        max_tokens = 512
    
    st.markdown("---")
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.conversation_history = []
        st.rerun()
    if st.button("Clear Code", use_container_width=True):
        st.session_state.code_history = []
        st.rerun()

st.title("🤖 Omni-Bot - Your Intelligent AI Companion")

# ✅ CHANGED: Provider-specific status banner
if ai_provider == "Google Gemini":
    if not google_api_key:
        st.warning("🔶 **Demo Mode** - Add GOOGLE_API_KEY for full Gemini features")
    else:
        st.success("🚀 **Gemini AI Mode Enabled!**")
else:
    if not groq_api_key:
        st.warning("🔶 **Demo Mode** - Add GROQ_API_KEY for full Groq features")
    else:
        st.success("🚀 **Groq AI Mode Enabled!**")

st.markdown("---")

# ✅ CHANGED: Unified get_llm() function supporting both Gemini and Groq
def get_llm():
    """Initialize and return LLM instance based on selected provider"""
    if ai_provider == "Google Gemini":
        if google_api_key:
            try:
                # Gemini initialization with max_output_tokens parameter
                return ChatGoogleGenerativeAI(
                    model=engine,
                    google_api_key=google_api_key,
                    temperature=temperature,
                    max_output_tokens=max_tokens  # Note: Gemini uses max_output_tokens
                )
            except Exception as e:
                st.error(f"Gemini Error: {str(e)}")
                return None
        return None
    else:  # Groq provider
        if groq_api_key:
            try:
                # Groq initialization with max_tokens parameter
                return ChatGroq(
                    groq_api_key=groq_api_key,
                    model_name=engine,
                    temperature=temperature,
                    max_tokens=max_tokens  # Note: Groq uses max_tokens
                )
            except Exception as e:
                st.error(f"Groq Error: {str(e)}")
                return None
        return None

if st.session_state.current_mode == "Brainy Buddy":
    st.header("💬 Brainy Buddy - Intelligent Conversations")
    
    for message in st.session_state.conversation_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask me anything..."):
        llm = get_llm()
        st.session_state.conversation_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            if not llm:
                # ✅ CHANGED: Demo message shows which API key is needed
                provider_name = "GOOGLE_API_KEY" if ai_provider == "Google Gemini" else "GROQ_API_KEY"
                response = f"Demo: You asked '{prompt}'. Add {provider_name} for AI responses."
                st.markdown(response)
                st.session_state.conversation_history.append({"role": "assistant", "content": response})
            else:
                with st.spinner("🤔 Thinking..."):
                    try:
                        messages = [
                            ("system", "You are Brainy Buddy, a helpful AI assistant."),
                            *[(msg["role"], msg["content"]) for msg in st.session_state.conversation_history[-6:]],
                        ]
                        prompt_template = ChatPromptTemplate.from_messages(messages)
                        chain = prompt_template | llm | StrOutputParser()
                        response = chain.invoke({"question": prompt})
                        st.markdown(response)
                        st.session_state.conversation_history.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

elif st.session_state.current_mode == "CodeCraft":
    st.header("💻 CodeCraft - Your Coding Assistant")
    
    col1, col2 = st.columns(2)
    with col1:
        programming_language = st.selectbox("Language", ["Python", "JavaScript", "Java", "C++", "Go", "Rust"])
    with col2:
        code_task = st.selectbox("Task", ["Write Code", "Debug/Explain", "Optimize", "Code Review"])
    
    if st.session_state.code_history:
        st.subheader("📚 Recent Code")
        for interaction in reversed(st.session_state.code_history[-3:]):
            with st.expander(f"💻 {interaction['task']} - {interaction['language']}"):
                st.code(interaction['question'])
                st.markdown(interaction['response'])
    
    code_prompt = st.text_area("Describe your task:", height=150)
    
    if st.button("Get Code Help 🚀", type="primary"):
        if code_prompt:
            with st.spinner("💻 Working..."):
                llm = get_llm()
                if not llm:
                    # ✅ CHANGED: Demo message shows which API key is needed
                    provider_name = "GOOGLE_API_KEY" if ai_provider == "Google Gemini" else "GROQ_API_KEY"
                    response = f"Demo: {programming_language} code for '{code_prompt}'. Add {provider_name} for real code."
                    st.markdown(response)
                else:
                    try:
                        prompt = ChatPromptTemplate.from_messages([
                            ("system", f"You are a {programming_language} expert. Task: {code_task}. Provide clear code."),
                            ("human", "{question}"),
                        ])
                        chain = prompt | llm | StrOutputParser()
                        response = chain.invoke({"question": code_prompt})
                        st.subheader("💡 Solution")
                        st.markdown(response)
                        st.session_state.code_history.append({
                            'task': code_task,
                            'language': programming_language,
                            'question': code_prompt,
                            'response': response
                        })
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            st.rerun()

st.markdown("---")
# ✅ CHANGED: Footer mentions both AI providers
st.markdown("<div style='text-align: center'><p>Built with Streamlit, LangChain, Google Gemini & Groq</p></div>", unsafe_allow_html=True)
