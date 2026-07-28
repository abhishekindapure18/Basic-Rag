"""Step 5: wrap the retriever as a tool the agent can call."""

from langchain.tools import tool

def create_search_tool(retriever):
    """Return a @tool function that searches the HR policy document."""

    @tool
    def search_hr_policy(questions: str) -> str:
        """Search the HR policy document for information about leave, work from home,
        probation, notice period, reimbursement, code of conduct, holidays, or exit process."""
        matching_chunks = retriever.invoke(questions)
        return "\n\n".join(chunks.page_content for chunks in matching_chunks)

    return search_hr_policy