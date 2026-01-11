import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from agent_core import ask
import time 

st.set_page_config(page_title="Oracle", layout="wide")

# --- GEMINI-DARK STYLING ---
st.markdown("""
    <style>
    .stApp { background-color: #0b192e; color: #e3e3e3; }
    
    /* STICKY COLUMN FIX */
    /* This targets the vertical block inside the second column */
    [data-testid="column"]:nth-child(2) {
        position: sticky;
        top: 1.5rem;
        max-height: calc(100vh - 2rem);
        overflow-y: auto;
        align-self: flex-start;
    }
    /* Chat Bubble Styling */
    .stChatMessage { border-radius: 15px; margin-bottom: 10px; border: 1px solid #1e293b; }
    
    /* Source Inspector Card */
    .source-card {
        background-color: #1e293b;
        border-radius: 16px;
        padding: 20px;
        border: 1px solid #334155;
        color: #f1f5f9;
    }
    
    /* Interactive Sentence Buttons */
    .stButton > button {
        border-radius: 12px;
        border: 1px solid #334155;
        background-color: #0f172a;
        color: #94a3b8;
        text-align: left;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        border-color: #38bdf8;
        color: #38bdf8;
        background-color: #1e293b;
    }
    
    /* The box containing the document text */
    .source-box {
        background-color: #0f172a;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #38bdf8;
        font-family: 'Courier New', Courier, monospace;
        font-size: 0.9rem;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "selected_span" not in st.session_state:
    st.session_state.selected_span = None

st.title("🔮 Oracle")

# --- MAIN LAYOUT ---
chat_col, source_col = st.columns([3, 2], gap="large")

with chat_col:
    chat_container = st.container()
    
    # Display message history
    with chat_container:
        for msg in st.session_state.messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            with st.chat_message(role):
                st.markdown(msg.content)

        # Persistence: Show the clickable spans for the MOST RECENT assistant response
        if st.session_state.last_result:
            st.write("---")
            st.caption("✨ Witty insight detected. Click a sentence to see my proof:")
            for i, span in enumerate(st.session_state.last_result.get("spans", [])):
                if span.get("source"):
                    if st.button(span["sentence"], key=f"span_{i}", use_container_width=True):
                        st.session_state.selected_span = span
                        st.rerun()

    # Chat Input
    user_input = st.chat_input("Seek wisdom from the Oracle...")

    if user_input:
        # 1. Store and Show User Message
        st.session_state.messages.append(HumanMessage(content=user_input))
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_input)
            
            # 2. Generate Assistant Response
            with st.chat_message("assistant"):
                with st.spinner("Consulting the ancient scrolls..."):
                    result = ask(user_input, st.session_state.messages)
                
                # 3. Typewriter Effect
                placeholder = st.empty()
                full_response = ""
                for chunk in result["answer"].split(" "):
                    full_response += chunk + " "
                    time.sleep(0.04)
                    placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)
                
                # 4. Save to State
                st.session_state.last_result = result
                st.session_state.messages.append(AIMessage(content=result["answer"]))
                
                # 5. Rerun to show the interactive buttons
                st.rerun()

# --- SOURCE COLUMN (The "Oracle's Library") ---
# --- SOURCE COLUMN (The Fixed Sidebar) ---
with source_col:
    # Use a container to group the sticky content
    with st.container():
        st.subheader("📚 Library Records")
        
        if st.session_state.selected_span:
            span = st.session_state.selected_span
            src = span["source"]
            
            st.markdown(f"""
                <div class="source-card">
                    <p style="color:#38bdf8; font-weight:bold; margin:0;">VERIFIED SOURCE</p>
                    <div class="source-box">
                        "{src.get('text')}"
                    </div>
                    <p style="font-size: 0.8rem; margin-top: 15px; color: #94a3b8;">
                        <b>File:</b> {src.get('file')}<br>
                        <b>Location:</b> Page {src.get('page')}<br>
                        <b>Confidence:</b> {int(span.get('score', 0) * 100)}%
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            if st.button("Close Source", use_container_width=True):
                st.session_state.selected_span = None
                st.rerun()
        else:
            st.info("The Oracle's evidence will appear here once you click a verified sentence in the chat.")