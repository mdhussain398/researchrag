"""
Academic Prompt Templates for Grounded Research Report Generation.
"""

from typing import List
from app.models.schemas import RetrievedChunk, ResearchConfig


ACADEMIC_SYSTEM_PROMPT = """You are an elite Senior AI Research Scientist, ML Literature Reviewer, and Academic Technical Writer.
Your mission is to generate a comprehensive, objective, and rigorously cited Academic Research Synthesis Report based SOLELY on the provided retrieved evidence chunks.

STRICT CITATION & GROUNDING RULES:
1. Every factual statement, claim, methodology description, benchmark metric, or finding MUST be explicitly cited using the bracketed chunk index, e.g. [1], [2], [1, 3].
2. NEVER invent, fabricate, or hallucinate citations or facts. ONLY use the document names, page numbers, and evidence text provided in the CONTEXT.
3. If the provided evidence is insufficient or silent on a specific aspect, explicitly state: "Insufficient evidence in the provided sources." Do NOT guess or extrapolate beyond the provided text.
4. Maintain a formal, rigorous academic tone suitable for publication in IEEE/ACM/Nature Machine Intelligence.
5. Structure your output with clear Markdown headers (# for Title, ## for Sections).
"""


def build_research_prompt(
    config: ResearchConfig,
    retrieved_chunks: List[RetrievedChunk],
) -> str:
    """Builds the comprehensive user prompt containing numbered context snippets and section instructions."""
    
    context_lines = []
    for i, item in enumerate(retrieved_chunks, 1):
        c = item.chunk
        context_lines.append(
            f"[{i}] ({c.filename}, p. {c.page_number}, sec. {c.section}):\n{c.text}\n"
        )
    
    context_text = "\n".join(context_lines)
    
    objectives_text = ""
    if config.sub_objectives:
        objectives_text = "RESEARCH SUB-OBJECTIVES:\n" + "\n".join([f"- {obj}" for obj in config.sub_objectives]) + "\n\n"

    prompt = f"""EVIDENCE CONTEXT CHUNKS:
{context_text}

--------------------------------------------------------------------------------
RESEARCH QUESTION:
{config.research_question}

{objectives_text}
--------------------------------------------------------------------------------
INSTRUCTIONS:
Generate a complete, structured Research Synthesis Report using the exact section structure below.
Ensure EVERY factual claim has bracketed citations pointing to the numbered evidence chunks above (e.g. [1], [2]).

REQUIRED REPORT STRUCTURE:
# Research Report: [Generate a Descriptive, Scholarly Title]

## 1. Executive Summary
- Provide a high-level executive summary synthesizing the core findings, methodology highlights, and key conclusions from the evidence. Include citations.

## 2. Background & Context
- Synthesize the motivation, foundational problems, and theoretical framing outlined across the ingested papers.

## 3. Key Findings
- Bulleted list of the most important empirical and theoretical findings with exact citations.

## 4. Methodology Comparison
- Detailed comparison of proposed approaches, architectures, loss functions, or experimental setups described in the evidence.

## 5. Evidence Synthesis & Cross-Paper Analysis
- Synthesize how the studies interact, complement, or build upon each other. Connect specific metrics and benchmark performances across papers.

## 6. Agreements Between Studies
- Summarize points of consensus and confirmed empirical observations supported by multiple sources.

## 7. Contradictions & Disagreements
- Detail any divergent findings, conflicting claims, or contrasting methodology trade-offs between papers. If none exist in the context, explicitly state that.

## 8. Limitations
- Detail the limitations and constraints explicitly acknowledged by the authors in the sources.

## 9. Research Gaps & Future Directions
- Identify unaddressed questions, domain generalization issues, or promising future research trajectories.

## 10. Conclusion
- Final scholarly conclusion summarizing the current state of knowledge for the research question.

## 11. References
- List all cited sources with their chunk IDs, document names, and page numbers:
  [1] <Filename>, p. <Page> (Section: <Section>)
"""
    return prompt
