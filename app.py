import os
import time
import streamlit as st
from typing import List, TypedDict, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

load_dotenv()
import wikipedia

wikipedia.set_user_agent("DeepResearchAgent/1.0 (college_project@example.com)")


# 0. UI CONFIG & CUSTOM CSS (MODERN SAAS THEME)

st.set_page_config(
    page_title="DeepResearch AI", 
    page_icon="⚡", 
    layout="wide",
    initial_sidebar_state="expanded"
)



st.markdown("""
<style>
    /* Main Background & Font Styling */
    .stApp {
        background-color: #0B0F19;
        color: #F3F4F6;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #030712;
        border-right: 1px solid #1F2937;
    }

    /* Glassmorphism Metric Cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.7) 100%);
        border: 1px solid rgba(51, 65, 85, 0.5);
        padding: 18px;
        border-radius: 12px;
        backdrop-filter: blur(12px);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    [data-testid="stMetricValue"] {
        color: #38BDF8 !important;
        font-weight: 700;
    }
    [data-testid="stMetricLabel"] {
        color: #94A3B8 !important;
    }

    /* Sleek Input Fields */
    [data-testid="stTextInput"] input {
        background-color: #111827;
        color: #FFFFFF;
        border: 1px solid #374151;
        border-radius: 8px;
        padding: 10px 14px;
    }
    [data-testid="stTextInput"] input:focus {
        border-color: #38BDF8;
        box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.2);
    }

    /* Primary Buttons */
    .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.6rem 1.2rem;
        transition: all 0.2s ease-in-out;
    }
    .stButton button[kind="primary"]:hover {
        background: linear-gradient(135deg, #2563EB 0%, #1E40AF 100%);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }

    /* Custom Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        border-bottom: 1px solid #1F2937;
        padding-bottom: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #94A3B8;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E293B !important;
        color: #38BDF8 !important;
    }

    /* Divider */
    hr {
        border-color: #1F2937;
    }
</style>
""", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = {}


# 1. STATE & PYDANTIC SCHEMAS

class ResearchState(TypedDict):
    research_topic: str
    search_queries: List[str]
    raw_context: List[str]
    iteration_count: int
    final_report: str
    requires_clarification: bool
    clarification_question: str

# Pydantic Schemas guarantee the LLM doesn't hallucinate JSON keys
class PlannerOutput(BaseModel):
    requires_clarification: bool = Field(description="True ONLY if the topic is incredibly vague (e.g., a single word like 'Apple' or 'Rust').")
    clarification_question: str = Field(description="If requires_clarification is true, the question to ask the user.")
    search_queries: List[str] = Field(description="List of exactly 3 Wikipedia search queries. MUST BE SHORT NOUNS ONLY (e.g., 'Figure AI', 'Humanoid robot'). Never use long sentences.")

class EvaluatorOutput(BaseModel):
    is_sufficient: bool = Field(description="True if the context contains enough facts to write a report.")
    search_queries: List[str] = Field(description="Exactly 2 new specific search queries to fill gaps. MUST BE SHORT NOUNS ONLY (e.g., 'Robotics', 'Artificial intelligence').")

# Initialize Wikipedia tool (Zero rate limits)
api_wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=2500)
search_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

def get_llm():
    
    return ChatGroq(model="openai/gpt-oss-120b", temperature=st.session_state.get("temperature", 0.0))


# 2. AGENT NODES

def planner_node(state: ResearchState):
    st.session_state.ui_status.write("🧠 **Analyzing topic and planning queries...**")
    
    prompt = f"Analyze the topic: '{state['research_topic']}'. If it's a single ambiguous word, flag it. Otherwise, provide 3 specific Wikipedia queries."
    
    
    structured_llm = get_llm().with_structured_output(PlannerOutput)
    response = structured_llm.invoke(prompt)
    
    if response.requires_clarification:
        st.session_state.ui_status.write("🛑 **Topic is ambiguous. Pausing for human clarification.**")
    else:
        st.session_state.ui_status.write(f"✅ **Planned queries:** {', '.join(response.search_queries)}")
        
    return {
        "search_queries": response.search_queries, 
        "iteration_count": 0,
        "requires_clarification": response.requires_clarification,
        "clarification_question": response.clarification_question
    }

def searcher_node(state: ResearchState):
    all_contexts = state.get("raw_context", [])
    for query in state["search_queries"]:
        st.session_state.ui_status.write(f"🔍 **Searching Wikipedia for:** *'{query}'*...")
        try:
            results = search_tool.invoke(query)
            if "No good Wikipedia Search Result was found" in results:
                st.session_state.ui_status.write(f"⚠️ **Warning:** No Wikipedia page found for '{query}'.")
            else:
                all_contexts.append(f"Source ({query}):\n{results}")
        except Exception as e:
           
            st.session_state.ui_status.write(f"⚠️ **Error Details for '{query}':** {str(e)}")
            
    return {"raw_context": all_contexts, "iteration_count": state.get("iteration_count", 0) + 1}

def evaluator_node(state: ResearchState):
    st.session_state.ui_status.write("⚖️ **Evaluating gathered evidence...**")
    context_str = "\n".join(state["raw_context"])
    
    prompt = f"Topic: '{state['research_topic']}'. Context: {context_str}. Is this enough to write a detailed report?"
    
    structured_llm = get_llm().with_structured_output(EvaluatorOutput)
    response = structured_llm.invoke(prompt)
    
    if response.is_sufficient or state["iteration_count"] >= st.session_state.get("max_loops", 2):
        st.session_state.ui_status.write("✅ **Context sufficient. Moving to synthesis.**")
        return {"search_queries": []}
    else:
        st.session_state.ui_status.write(f"🔄 **Found gaps. New queries:** {', '.join(response.search_queries)}")
        return {"search_queries": response.search_queries}

def writer_node(state: ResearchState):
    st.session_state.ui_status.write("✍️ **Drafting final report...**")
    context_str = "\n\n".join(state["raw_context"])
    prompt = f"Write a comprehensive professional research summary on: {state['research_topic']}. Source data: {context_str}. Include Markdown headers."
    
    report = get_llm().invoke(prompt)
    st.session_state.ui_status.update(label="✅ **Execution Complete!**", state="complete", expanded=False)
    return {"final_report": report.content}


# 3. ROUTING & GRAPH COMPILATION

def route_research(state: ResearchState) -> Literal["searcher", "writer"]:
    if len(state["search_queries"]) == 0 or state["iteration_count"] >= st.session_state.get("max_loops", 2):
        return "writer"
    return "searcher"

def route_after_planner(state: ResearchState) -> Literal["searcher", "__end__"]:
    if state.get("requires_clarification", False):
        return END 
    return "searcher"

workflow = StateGraph(ResearchState)
workflow.add_node("planner", planner_node)
workflow.add_node("searcher", searcher_node)
workflow.add_node("evaluator", evaluator_node)
workflow.add_node("writer", writer_node)

workflow.add_edge(START, "planner")
workflow.add_conditional_edges("planner", route_after_planner, {"searcher": "searcher", END: END})
workflow.add_edge("searcher", "evaluator")
workflow.add_conditional_edges("evaluator", route_research)
workflow.add_edge("writer", END)
research_app = workflow.compile()


# 4. STREAMLIT UI

with st.sidebar:
    st.title("⚙️ Agent Settings")
    st.session_state.temperature = st.slider("Model Creativity", 0.0, 1.0, 0.0, 0.1)
    st.session_state.max_loops = st.number_input("Max Reflection Loops", 1, 5, 2)
    
    st.divider()
    st.subheader("📚 Session History")
    if st.session_state.history:
        for past_topic in reversed(st.session_state.history.keys()):
            if st.button(f"📄 {past_topic[:20]}...", key=f"btn_{past_topic}", use_container_width=True):
                st.session_state.current_view = past_topic

st.title("⚡ Autonomous DeepResearch AI")

# Form to prevent "Enter" key bugs

with st.form("research_form"):
    col1, col2 = st.columns([4, 1])
    with col1:
        topic = st.text_input("Topic", placeholder="e.g., Commercial viability of solid-state batteries", label_visibility="collapsed")
    with col2:
        start_btn = st.form_submit_button("🚀 Dispatch Agent", use_container_width=True, type="primary")

if start_btn and topic:
    st.session_state.pop("awaiting_human", None)
    start_time = time.time()
    
    # Status container must be outside the form
    
    st.session_state.ui_status = st.status("Initializing Agent Engine...", expanded=True)
    
    result = research_app.invoke({"research_topic": topic, "iteration_count": 0, "raw_context": []})
    
    if result.get("requires_clarification", False):
        st.session_state.ui_status.update(label="⏸️ **Agent Paused**", state="error", expanded=True)
        st.session_state.awaiting_human = result
    else:
        elapsed = time.time() - start_time
        st.session_state.ui_status.update(label="✅ **Execution Complete!**", state="complete", expanded=False)
        st.session_state.history[topic] = {
            "report": result.get("final_report", ""),
            "iterations": result.get("iteration_count", 0),
            "time": elapsed,
            "sources": len(result.get("raw_context", [])),
            "raw_data": "\n\n---\n\n".join(result.get("raw_context", []))
        }
        st.session_state.current_view = topic

# Human-in-the-Loop UI

if "awaiting_human" in st.session_state:
    st.warning("⚠️ **The agent needs your help to proceed.**")
    st.info(f"**Agent asks:** {st.session_state.awaiting_human['clarification_question']}")
    human_feedback = st.text_input("Your clarification:")
    
    if st.button("Submit & Resume Research", type="primary"):
        refined_topic = f"{st.session_state.awaiting_human['research_topic']} (Clarification: {human_feedback})"
        del st.session_state.awaiting_human
        
        st.session_state.ui_status = st.status("Resuming Agent Engine...", expanded=True)
        start_time = time.time()
        result = research_app.invoke({"research_topic": refined_topic, "iteration_count": 0, "raw_context": []})
        elapsed = time.time() - start_time
        
        st.session_state.ui_status.update(label="✅ **Execution Complete!**", state="complete", expanded=False)
        st.session_state.history[refined_topic] = {
            "report": result.get("final_report", ""),
            "iterations": result.get("iteration_count", 0),
            "time": elapsed,
            "sources": len(result.get("raw_context", [])),
            "raw_data": "\n\n---\n\n".join(result.get("raw_context", []))
        }
        st.session_state.current_view = refined_topic
        st.rerun()

# Dashboard
if "current_view" in st.session_state and st.session_state.current_view in st.session_state.history:
    data = st.session_state.history[st.session_state.current_view]
    st.divider()
    st.subheader(f"Results: {st.session_state.current_view}")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("⏱️ Execution Time", f"{data['time']:.1f} sec")
    m2.metric("🔄 Reflection Loops", data['iterations'])
    m3.metric("🌐 Search Batches", data['sources'])
    
    tab1, tab2 = st.tabs(["📝 Final Report", "🗄️ Raw Context"])
    with tab1:
        st.markdown(data["report"])
    with tab2:
        st.text_area("Wikipedia Raw Data", data["raw_data"], height=400)