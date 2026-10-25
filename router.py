
import json
import re

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You route questions for an assistant that answers ONLY from one company's annual report "
     "(financial results, segments, people, risks, strategy, outlook, governance, ESG).\n"
     "Classify the user's question:\n"
     '- "simple": one information need.\n'
     '- "multi_part": two or more distinct needs (e.g. two metrics, "X and Y", a comparison, '
     '"how did A change and why"). Split it into 2-4 standalone sub-questions, each naming its topic.\n'
     '- "out_of_scope": not answerable from a company report (investment advice, share price '
     "predictions, general knowledge, chit-chat).\n"
     'Reply with JSON only: {{"route": "...", "sub_questions": ["...", "..."]}}'),
    ("human", "{question}"),
])


class QueryPlan(BaseModel):
    route: str = "simple"
    sub_questions: list[str] = []


def _text(msg) -> str:
    return msg.text() if callable(getattr(msg, "text", None)) else str(msg.content)


def plan_query(llm, question: str) -> QueryPlan:
    """Ask Gemini how to handle the question. Falls back to 'simple' if anything goes wrong."""
    try:
        reply = _text((ROUTER_PROMPT | llm).invoke({"question": question}))
        match = re.search(r"\{.*\}", reply, re.DOTALL)          # pull the JSON out of the reply
        plan = QueryPlan.model_validate(json.loads(match.group(0)))
    except Exception:
        return QueryPlan(route="simple")
    if plan.route not in ("simple", "multi_part", "out_of_scope"):
        return QueryPlan(route="simple")
    if plan.route == "multi_part" and len(plan.sub_questions) < 2:
        return QueryPlan(route="simple")                         # a "split" into one part isn't a split
    return plan