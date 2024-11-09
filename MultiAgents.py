import gradio as gr
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
import os
from datetime import datetime
import logging

# Load environment variables
load_dotenv()

# Set API Keys from .env file
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# LLM
llm = ChatOpenAI(model="gpt-4o")

# Configure logging
logging.basicConfig(level=logging.INFO)

# Define the search tool
tool = TavilySearchResults(max_results=3)

# Define the AgentState data structure
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

# Base Agent class
class Agent:
    def __init__(self, model, tools, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges("llm", self.exists_action, {True: "action", False: END})
        graph.add_edge("action", "llm")   # Loop Back
        graph.set_entry_point("llm")
        self.graph = graph.compile()
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    # Processes inputs using the model.
    def call_openai(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}   # Updates the state with the model's response

    # Checks if tools need to be invoked
    def exists_action(self, state: AgentState):
        result = state['messages'][-1]     # Checks the latest message in the state
        return len(result.tool_calls) > 0        # If the model’s output includes tool calls, return True

    # Executes tool calls and retrieves results
    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            logging.info(f"Calling tool: {t['name']} with query: {t['args']['query']}")
            result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        logging.info("Returning results to the model...")
        return {'messages': results}

# Research the Industry or the Company Agent
research_prompt = """
You are a market research expert specializing in industry analysis and competitive insights. Use available tools, including web-based research, to conduct a thorough and in-depth analysis of the {query_context}.

Key goals:
1. **Understand the Industry and Segment**: Provide a detailed overview of the industry or segment the company is operating in (e.g., Automotive, Manufacturing, Finance, Retail, Healthcare, etc.). Include insights on market trends, segmentation, and challenges.

2. **Analyze the Company's Offerings and Focus Areas**:
   - Identify the company’s key offerings and their strategic focus areas (e.g., operations, supply chain, customer experience, etc.).
   - Summarize the company’s vision, mission, and product/service portfolio.

3. **Competitor Analysis**:
   - Provide a detailed list of competitors in the same industry or segment.
   - Highlight examples of their AI, GenAI, and ML applications or innovations.
   - Assess their strategies and identify differentiators that the company can leverage.

4. **Quantitative Data**: Include relevant metrics, such as market size, growth rates, technology adoption percentages, and revenue impacts, to support your findings.

5. **Actionable Insights**: Summarize clear and actionable insights that:
   - Highlight opportunities for AI/ML/automation in the company’s operations.
   - Guide further exploration and decision-making for generating impactful AI use cases.

6. **Depth of Analysis**: Ensure the research demonstrates both breadth and depth, providing a comprehensive understanding of the market landscape, the competitive environment, and technological adoption trends.

Output: Deliver a detailed yet concise analysis with insights that help generate tailored AI use cases.
"""

research_agent = Agent(llm, [tool], system=research_prompt)

# Market Standards & Use Case Generation Agent
use_case_prompt = """
You are an expert in AI use case generation, specializing in industry-specific innovation. Based on the following industry research, generate actionable and impactful AI/GenAI use cases strictly related to AI, Machine Learning (ML), and Automation, tailored to the industry’s pain points and opportunities.

Key goals:
1. Focus exclusively on use cases that leverage:
   - Artificial Intelligence (AI)
   - Machine Learning (ML)
   - Generative AI (GenAI)
   - Automation technologies
   - Large Language Models (LLMs)

2. Ensure all use cases are:
   - **Relevant**: Directly address the key challenges, opportunities, and goals specific to the industry or company.
   - **Creative**: Introduce innovative and forward-thinking applications of AI/ML/automation that go beyond conventional solutions.

3. Use a clear and structured format for each use case:
   - **Objective/Use Case**: Describe the primary goal and the specific application area.
   - **AI/ML/Automation Application**: Clearly explain how AI, ML, or automation is utilized to address the use case.
   - **Cross-Functional Benefits**: Highlight the benefits across multiple departments or functions, using bullets or subheadings.

4. Provide a list of at least five impactful use cases, formatted as follows:
   ### AI, ML & Automation Use Cases for [Industry or Company Name]

   As a leading player in [Industry/Company context], [Company/Industry Name] can leverage Artificial Intelligence (AI), Machine Learning (ML), Generative AI (GenAI), and Automation to [overall benefit statement]. The following use cases can be realized:

   **Use Case 1: [Use Case Title]**
   - **Objective/Use Case**: [Description]
   - **AI/ML/Automation Application**: [Description of how AI, ML, or automation is applied.]
   - **Cross-Functional Benefit**:
     - [Department 1]: [Benefit]
     - [Department 2]: [Benefit]

   **Use Case 2: [Use Case Title]**
   - **Objective/Use Case**: [Description]
   - **AI/ML/Automation Application**: [Description of how AI, ML, or automation is applied.]
   - **Cross-Functional Benefit**:
     - [Department 1]: [Benefit]
     - [Department 2]: [Benefit]

   Repeat the above format for all proposed use cases.

5. Summarize actionable insights clearly to guide stakeholders in prioritization and implementation.

**Remember**: Relevance and creativity are critical in generating use cases that demonstrate the unique value AI/ML/automation can bring to the industry or company.
"""

use_case_agent = Agent(llm, [tool], system=use_case_prompt)

# Resource Asset Collection Agent
resource_collection_prompt = """
You are an expert in resource asset collection, tasked with identifying datasets, tools, frameworks, and proposing actionable Generative AI (GenAI) solutions.

Key goals:
1. Search for relevant datasets on platforms like Kaggle, HuggingFace, GitHub, and others to support the proposed use cases.
2. Identify pre-trained models, APIs, or open-source tools that align with each use case.
3. Propose Generative AI (GenAI) solutions, such as:
   - **Document Search**: AI-powered tools for semantic search of internal or external documents.
   - **Automated Report Generation**: Tools or frameworks for generating tailored reports based on provided inputs.
   - **AI-Powered Chat Systems**: Virtual assistants for customer support or operational tasks.
4. Ensure that all resources and solutions are practical, accessible, and include clickable links or examples for easy implementation.

Output: Deliver a detailed list of datasets, tools, resources, and GenAI solutions organized by use case. Provide clear links and descriptions for each.
"""

resource_agent = Agent(llm, [tool], system=resource_collection_prompt)

# Function to save resources to a file
def save_resources_to_file(content: str, directory: str = "output"):
    """
    Save the resource content into a uniquely named file based on the current timestamp.

    Args:
        content (str): The resource content to save.
        directory (str): The directory where the file will be saved.

    Returns:
        str: The path of the saved file.
    """
    try:
        # Ensure the directory exists
        os.makedirs(directory, exist_ok=True)

        # Generate a unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"resources_{timestamp}.md"
        file_path = os.path.join(directory, filename)

        # Write content to the file
        with open(file_path, "w") as file:
            file.write(content)

        logging.info(f"Resources saved successfully to {file_path}")
        return file_path
    except Exception as e:
        logging.error(f"Failed to save file: {e}")
        raise


# Multi-Agent Workflow
def multi_agent_workflow(question):
    # Step 1: Research the Industry or the Company
    research_messages = [HumanMessage(content=question)]
    research_state = {"messages": research_messages}
    research_result = research_agent.graph.invoke(research_state)
    research_content = research_result["messages"][-1].content

    # Step 2: Generate Use Cases
    use_case_messages = [
        HumanMessage(content=f"Input Research: {research_content}")
    ]
    use_case_state = {"messages": use_case_messages}
    use_case_result = use_case_agent.graph.invoke(use_case_state)
    use_case_content = use_case_result["messages"][-1].content

    # Step 3: Collect Resources
    resource_messages = [
        HumanMessage(content=f"Use Case Input: {use_case_content}")
    ]
    resource_state = {"messages": resource_messages}
    resource_result = resource_agent.graph.invoke(resource_state)
    resource_content = resource_result["messages"][-1].content

    # Save resource content with a unique file name
    file_path = save_resources_to_file(resource_content)
    logging.info(f"Resource file saved at: {file_path}")

    return research_content, use_case_content, resource_content


# Define Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# Multi-Agent System for AI-Powered Market Research, Use Case Generation, and Resource Collection")
    query = gr.Textbox(label="Enter Industry or Company Query")
    output = gr.Markdown()
    submit_button = gr.Button("Generate Insights")

    def gradio_interface_with_progress(query):
        import time
        progress_log = ""  # Accumulate all outputs
        try:
            # Step 1: Research the Industry or the Company
            yield "Step 1: Researching the Industry or the Company..."  # Update progress bar
            research_messages = [HumanMessage(content=query)]
            research_state = {"messages": research_messages}
            research_result = research_agent.graph.invoke(research_state)
            research_content = research_result["messages"][-1].content
            progress_log += f"### Research Output:\n{research_content}\n\n"
            time.sleep(1)  # Simulate progress for visibility

            # Step 2: Generate Use Cases
            yield "Step 2: Generating Use Cases..."  # Update progress bar
            use_case_messages = [
                HumanMessage(content=f"Input Research: {research_content}")
            ]
            use_case_state = {"messages": use_case_messages}
            use_case_result = use_case_agent.graph.invoke(use_case_state)
            use_case_content = use_case_result["messages"][-1].content
            progress_log += f"### Use Case Output:\n{use_case_content}\n\n"
            time.sleep(1)  # Simulate progress for visibility

            # Step 3: Collect Resources
            yield "Step 3: Collecting Resources..."  # Update progress bar
            resource_messages = [
                HumanMessage(content=f"Use Case Input: {use_case_content}")
            ]
            resource_state = {"messages": resource_messages}
            resource_result = resource_agent.graph.invoke(resource_state)
            resource_content = resource_result["messages"][-1].content
            progress_log += f"### Resource Output:\n{resource_content}\n\n"
            file_path = save_resources_to_file(resource_content)
            progress_log += f"**Resources saved at:** {file_path}\n\n"
            time.sleep(1)  # Simulate progress for visibility


            # Final Output
            progress_log += "### Process Completed Successfully!"
            yield progress_log  # Display final accumulated output
        except Exception as e:
            progress_log += f"Error occurred: {e}"
            yield progress_log  # Display error message


    # Connect buttons
    submit_button.click(gradio_interface_with_progress, inputs=[query], outputs=[output])

# Launch
demo.launch()
