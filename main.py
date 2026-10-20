import os
from typing import List, TypedDict
from dotenv import load_dotenv

# 1. LangChain & LangGraph Imports
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

# Load the environment variables (.env file)
load_dotenv()

# 2. State Definition
class ResearchState(TypedDict):
    research_topic: str
    search_queries: List[str]
    raw_context: List[str]
    iteration_count: int
    final_report: str

# 3. Cloud LLM & Tools Setup
# Llama-3.3-70b-versatile is a massively capable, fast model provided freely by Groq
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
api_wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=2500)
search_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

# 4. Planner Node
def planner_node(state: ResearchState):
    prompt = f"""
    You are an expert researcher. Break this topic into 3 simple, distinct web search queries:
    "{state['research_topic']}"
    
    Output ONLY 3 queries, one per line. Do not number them or add any other text.
    """
    response = llm.invoke(prompt)
    
    # Clean the output into a Python list
    queries = [q.strip("- ").strip() for q in response.content.strip().split("\n") if q.strip()][:3]
    print(f"\n[Planner] Generated search queries:\n{queries}")
    
    return {"search_queries": queries, "iteration_count": 0}

# 5. Searcher Node
# 5. Searcher Node (Updated with Error Handling)
# 5. Searcher Node (Updated with Error Handling for Terminal)
def searcher_node(state: ResearchState):
    all_contexts = state.get("raw_context", [])
    for query in state["search_queries"]:
        print(f"[Searcher] 🔍 Searching Wikipedia for: '{query}'...")
        try:
            results = search_tool.invoke(query)
            
            # If Wikipedia returns no results
            if "No good Wikipedia Search Result was found" in results:
                print(f"[Searcher] ⚠️ Warning: No data found for '{query}'.")
            else:
                all_contexts.append(results)
                
        except Exception as e:
            print(f"[Searcher] ⚠️ Error: Search failed for '{query}'.")
            # Tell the LLM that the search failed so it doesn't hallucinate
            all_contexts.append(f"Search failed for '{query}' due to a network error.")
            
    return {"raw_context": all_contexts, "iteration_count": state.get("iteration_count", 0) + 1}

# 6. Writer Node
def writer_node(state: ResearchState):
    print("\n[Writer] Synthesizing data into final report...")
    context_str = "\n\n".join(state["raw_context"])
    
    prompt = f"""
    Write a concise, professional research summary on: {state['research_topic']}
    
    Use the following search data as your source material:
    {context_str}
    
    Provide key takeaways, main findings, and list any URLs found in the text at the bottom.
    Format your response in clean Markdown.
    """
    report = llm.invoke(prompt)
    return {"final_report": report.content}

# 7. Assemble the Graph
workflow = StateGraph(ResearchState)
workflow.add_node("planner", planner_node)
workflow.add_node("searcher", searcher_node)
workflow.add_node("writer", writer_node)

# Define the flow
workflow.add_edge(START, "planner")
workflow.add_edge("planner", "searcher")
workflow.add_edge("searcher", "writer")
workflow.add_edge("writer", END)

# Compile the application
research_app = workflow.compile()

# 8. Run the Agent
if __name__ == "__main__":
    result = research_app.invoke({
        "research_topic": "Latest breakthroughs in solid-state EV batteries for 2026"
    })
    
    print("\n" + "=" * 50)
    print("FINAL RESEARCH REPORT")
    print("=" * 50)
    print(result["final_report"])