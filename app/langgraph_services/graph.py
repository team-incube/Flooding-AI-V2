from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from app.langgraph_services.nodes import (
    GraphState,
    booking_stub,
    general_chat,
    generate,
    grade_answer,
    grade_documents,
    retrieve,
    route_question,
    transform_query,
    web_search,
)
from app.langgraph_services.routing import (
    route_after_grade_answer,
    route_after_grade_documents,
    route_after_question,
)


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("route_question", route_question)
    graph.add_node("booking_stub", booking_stub)
    graph.add_node("general_chat", general_chat)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("transform_query", transform_query)
    graph.add_node("web_search", web_search)
    graph.add_node("generate", generate)
    graph.add_node("grade_answer", grade_answer)

    graph.add_edge(START, "route_question")
    graph.add_conditional_edges(
        "route_question",
        route_after_question,
        {
            "booking_stub": "booking_stub",
            "general_chat": "general_chat",
            "web_search": "web_search",
            "retrieve": "retrieve",
        },
    )
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        route_after_grade_documents,
        {
            "generate": "generate",
            "transform_query": "transform_query",
        },
    )
    graph.add_edge("transform_query", "retrieve")
    graph.add_edge("web_search", "generate")

    # generate/general_chat/booking_stub 모두 여기서 만나 grade_answer로 검증받는다.
    graph.add_edge("generate", "grade_answer")
    graph.add_edge("general_chat", "grade_answer")
    graph.add_edge("booking_stub", "grade_answer")
    graph.add_conditional_edges(
        "grade_answer",
        route_after_grade_answer,
        {
            "generate": "generate",
            "general_chat": "general_chat",
            "booking_stub": "booking_stub",
            END: END,
        },
    )

    return graph.compile()


app_graph = build_graph()


async def ask(user_input: str) -> str:
    result = await app_graph.ainvoke({
        "messages": [HumanMessage(content=user_input)],
        "mode": "vectorstore",
        "context": [],
    })
    return result["messages"][-1].content
