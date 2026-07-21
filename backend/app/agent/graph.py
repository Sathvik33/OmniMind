from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from backend.app.router.decision_engine import DecisionEngine
from backend.app.context.context_builder import ContextBuilder
from backend.app.context.prompt_builder import PromptBuilder
from backend.app.context.memory import MemoryManager
from backend.app.llm.llm_router import LLMRouter
from backend.app.guardrails.generation_guard import GenerationGuard
from backend.app.db.database import SessionLocal

class AgentState(TypedDict):
    session_id: int
    query: str
    has_images: bool
    decision: Dict[str, Any]
    retrieved_context: List[Dict[str, Any]]
    final_prompt: str
    final_answer: str

class AgentWorkflow:
    def __init__(self):
        self.graph = StateGraph(AgentState)
        self._build_graph()
        
    def _build_graph(self):
        self.graph.add_node("planner", self._node_planner)
        self.graph.add_node("executor", self._node_executor)
        self.graph.add_node("generator", self._node_generator)
        
        self.graph.set_entry_point("planner")
        self.graph.add_edge("planner", "executor")
        self.graph.add_edge("executor", "generator")
        self.graph.add_edge("generator", END)
        
        self.runner = self.graph.compile()

    def _node_planner(self, state: AgentState) -> AgentState:
        db = SessionLocal()
        try:
            decision_engine = DecisionEngine(db)
            result = decision_engine.process_query(state["query"], state["has_images"])
            state["decision"] = result["decision"]
            state["retrieved_context"] = result["context"]
        finally:
            db.close()
        return state

    def _node_executor(self, state: AgentState) -> AgentState:
        # 1. Format Context
        context_str = ContextBuilder.build(state.get("retrieved_context", []))
        
        # 2. Add Memory
        history = []
        db = SessionLocal()
        try:
            memory_manager = MemoryManager(db)
            history = memory_manager.get_recent_history(state["session_id"], limit=5)
        finally:
            db.close()
            
        # 3. Assemble Prompt
        prompt = PromptBuilder.build_prompt(state["query"], context_str, history)
        state["final_prompt"] = prompt
        
        return state
        
    def _node_generator(self, state: AgentState) -> AgentState:
        router = LLMRouter()
        category = state.get("decision", {}).get("category", "fast_qa")
        
        # 1. Generate Output
        raw_output = router.generate(state["final_prompt"], category=category)
        
        # 2. Apply Guardrails
        guard_result = GenerationGuard.validate(raw_output)
        if not guard_result["valid"]:
            state["final_answer"] = f"Error: Output blocked. Reason: {guard_result['reason']}"
        else:
            state["final_answer"] = raw_output
            
        return state
        
    def run(self, session_id: int, query: str, has_images: bool = False) -> str:
        initial_state = {
            "session_id": session_id,
            "query": query,
            "has_images": has_images,
            "decision": {},
            "retrieved_context": [],
            "final_prompt": "",
            "final_answer": ""
        }
        result = self.runner.invoke(initial_state)
        return result["final_answer"]

agent_workflow = AgentWorkflow()
