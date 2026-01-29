import streamlit as st
import requests
import time

# --- Configuration ---
API_URL = "http://127.0.0.1:8000"
st.set_page_config(page_title="Gemini 3 RAG", page_icon="⚡")

# --- Sidebar: Document Upload ---
with st.sidebar:
    st.header("📂 Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF or TXT", type=["pdf", "txt", "png"])
    
    if uploaded_file is not None:
        if st.button("Process Document"):
            with st.spinner("Uploading and processing..."):
                # Use the correct file format for requests
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                try:
                    response = requests.post(f"{API_URL}/documents/upload", files=files)
                    if response.status_code == 200:
                        st.success("✅ Document queued successfully!")
                        st.json(response.json())
                    else:
                        st.error(f"Upload failed: {response.text}")
                except Exception as e:
                    st.error(f"Connection error: {e}")

    st.markdown("---")
    st.markdown("### 📊 System Status")
    if st.button("Check API Health"):
        try:
            res = requests.get(f"{API_URL}/")
            if res.status_code == 200:
                st.success("Online 🟢")
                st.json(res.json())
            else:
                st.error("Offline 🔴")
        except:
            st.error("Connection Failed 🔴")

# --- Main Chat Interface ---
st.title("⚡ Gemini 3 RAG Chat")
st.markdown("Ask questions about your uploaded documents.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📚 View Sources"):
                for idx, source in enumerate(message["sources"]):
                    st.markdown(f"**Chunk {idx+1}** (Score: {source['similarity_score']:.2f})")
                    st.caption(source['content'])

# Chat Input
if prompt := st.chat_input("What is this document about?"):
    # 1. Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Call API
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking... 🧠")
        
        try:
            payload = {
                "question": prompt,
                "top_k": 5,
                "include_sources": True
            }
            start_time = time.time()
            response = requests.post(f"{API_URL}/qa/ask", json=payload)
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                answer = data["answer"]
                sources = data.get("sources", [])
                
                # --- FIX: Construct metrics dictionary from flat response ---
                metrics_display = {
                    "retrieval_ms": data.get("retrieval_time_ms"),
                    "generation_ms": data.get("generation_time_ms"),
                    "total_ms": data.get("total_time_ms"),
                    "confidence": data.get("confidence_score")
                }
                # -----------------------------------------------------------

                # Display Answer
                message_placeholder.markdown(answer)
                
                # Display Sources dropdown
                with st.expander(f"📚 Sources & Metrics ({elapsed:.2f}s)"):
                    st.json(metrics_display)
                    if sources:
                        for idx, source in enumerate(sources):
                            st.markdown(f"**{source['document_name']}** (Score: {source['similarity_score']:.2f})")
                            st.caption(source['content'])
                    else:
                        st.info("No sources cited.")

                # Save to history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": sources
                })
            else:
                message_placeholder.error(f"Error {response.status_code}: {response.text}")
        except Exception as e:
            message_placeholder.error(f"Failed to connect to backend: {e}")