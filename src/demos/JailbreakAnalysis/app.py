import streamlit as st
import numpy as np
import shapiq
from demos.shared.hf_model import HFModelWrapper
from demos.JailbreakAnalysis.JailbreakAnalysisGame import JailbreakGame

st.set_page_config(
    page_title="Shapiq Jailbreak Explainability",
    page_icon="🔍",
    layout="wide",
)

# --- Caching & State ---
@st.cache_resource
def get_model(model_name: str, temperature: float = 0.0):
    return HFModelWrapper(model_name=model_name, device="cuda", temperature=temperature)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "llama-3.3-70b-versatile"

# --- Sidebar Config ---
st.sidebar.title("Configuration")

model_choices = [
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "openai/gpt-oss-safeguard-20b",
    "qwen/qwen3-32b",
    "gemini-2.5-flash",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
]

selected_model = st.sidebar.selectbox("Model", model_choices, index=0)
st.session_state.selected_model = selected_model

temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.1)

# --- Navigation ---
tab_inference, tab_explanation = st.tabs(["💬 Inference", "🔍 Explanation"])

# --- Inference Tab ---
with tab_inference:
    st.markdown("## Chat with the Model")
    st.markdown(f"**Current Model:** `{st.session_state.selected_model}` | **Temperature:** `{temperature}`")
    
    # Display chat
    for msg in st.session_state.chat_history:
        st.chat_message(msg["role"]).write(msg["content"])
        
    prompt = st.chat_input("Enter your prompt...")
    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        with st.chat_message("assistant"):
            try:
                model = get_model(st.session_state.selected_model, temperature=temperature)
                # Note: caching the model uses the previous temp, so we explicitly pass it below.
                stream = model.generate_text_stream(prompt=prompt, chat=True, temperature=temperature)
                response = st.write_stream(stream)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error during inference: {e}")

# --- Explanation Tab ---
with tab_explanation:
    st.markdown("## Explanation with shapiq")
    st.markdown("Analyze the compliance of the model based on Shapley values. This evaluates how individual parts of your prompt influence the model's output compliance.")
    
    # Explanation config
    with st.expander("Explanation Settings", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            scoring_mode = st.selectbox("Value Function", ["llm-as-a-judge", "logprob"], index=0)
            
            judge_model = None
            if scoring_mode == "llm-as-a-judge":
                judge_model = st.selectbox("Judge Model", model_choices, index=model_choices.index("Qwen/Qwen2.5-1.5B-Instruct"))
                
            masking_strategy = st.selectbox("Masking Strategy", ["remove", "mask", "distributional", "generative"], index=0)
            
        with col2:
            segmentation = st.selectbox("Segmentation Level", ["semantic", "sentence", "word", "token"], index=0)
            
            semantic_window = 4
            semantic_threshold = 0.50
            if segmentation == "semantic":
                semantic_window = st.number_input("Segmentation Window Size", value=4, min_value=1)
                semantic_threshold = st.number_input("Similarity Threshold", value=0.50, min_value=0.0, max_value=1.0, step=0.05)
                
    # Input for explanation
    explain_prompt = st.text_area("Prompt to explain", height=100)
    
    if st.button("Explain with shapiq", type="primary"):
        if not explain_prompt:
            st.warning("Please enter a prompt to explain.")
        else:
            with st.status("Running explanation...") as status:
                try:
                    st.write("Loading model...")
                    # We pass temperature=0.0 for deterministic explanations
                    model = get_model(st.session_state.selected_model, temperature=0.0)
                    
                    st.write("Initializing Jailbreak Game...")
                    game = JailbreakGame(
                        model_name=st.session_state.selected_model,
                        input_text=explain_prompt,
                        scoring_mode=scoring_mode,
                        mask_strategy=masking_strategy,
                        segmentation=segmentation,
                        device="cuda",
                        hf_model=model,
                        judge_model_name=judge_model if judge_model else "Qwen/Qwen2.5-1.5B-Instruct",
                        semantic_window=int(semantic_window),
                        semantic_threshold=float(semantic_threshold),
                    )
                    
                    st.write("Calculating compliance score...")
                    full_coalition = np.ones((1, game.n_players))
                    compliance_score = float(game.value_function(full_coalition)[0])
                    
                    st.write("Running Shapiq approximation...")
                    approx = shapiq.KernelSHAP(n=game.n_players, random_state=42)
                    result = approx.approximate(budget=10, game=game)
                    
                    player_values = np.array([float(result[(i,)]) for i in range(game.n_players)])
                    
                    status.update(label="Explanation complete!", state="complete", expanded=False)
                    
                    # Rendering results
                    st.success(f"**Model:** {st.session_state.selected_model}  |  **Compliance Score:** `{compliance_score:+.4f}`")
                    
                    st.markdown("### Shapley Values")
                    
                    # Custom HTML for colorized bars
                    html = "<table><thead><tr><th>Player</th><th>Shapley Value</th><th>Contribution</th></tr></thead><tbody>"
                    for p, val in zip(game.players, player_values, strict=False):
                        bar_len = min(int(abs(val) * 15), 10)
                        bar_color = "#10b981" if val >= 0 else "#ef4444"
                        bar = f'<span style="color:{bar_color}">{"█" * bar_len}</span>'
                        html += f"<tr><td><code>{p}</code></td><td>{val:+.4f}</td><td>{bar}</td></tr>"
                    html += "</tbody></table>"
                    
                    st.html(html)
                    
                except Exception as e:
                    status.update(label="Error during explanation.", state="error")
                    st.error(f"Error during explanation: {e}")
