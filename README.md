# **Multi-Agent System for AI-Powered Market Research, Use Case Generation, and Resource Collection**

This repository implements a **multi-agent system** to automate **market research**, **AI/ML use case generation**, and **resource collection**. It uses **GPT-4**, **Gradio**, **Langchain**, **LangGraph**, and **Tavily API** to generate actionable AI solutions tailored to industries or companies.

## **Features**
- **Market Research**: Analyze industries or companies, providing insights into trends, competitors, and technology adoption.
- **Use Case Generation**: Create AI, ML, and automation use cases based on research findings.
- **Resource Collection**: Gather datasets, tools, and APIs for implementing AI solutions.
- **Gradio Interface**: User-friendly interface for input, progress tracking, and output retrieval.

## **Technologies Used**
- **GPT-4**: For generating insights and use cases.
- **Langchain**: Framework for language model-powered applications.
- **LangGraph**: Graph-based tool for modeling agent workflows.
- **Tavily API**: Web scraping for research data.
- **Gradio**: For building the interactive UI.

## **System Workflow**

The diagram below illustrates the flow of actions taken by the multi-agent system. 
![codetoflow](https://github.com/user-attachments/assets/996c54a7-0208-4f56-a43d-ed87508ed410)


### Input Query:
- Users enter a query into the **Gradio** interface.
  - **Example**: "Analyze the e-commerce industry."

#### Step 1: Market Research:
- The **research_agent**:
  - Gathers industry/company insights using **GPT-4** and the **TavilySearchResults** tool.
  - Synthesizes data into a structured output including:
    - Market trends
    - Competitor analysis
    - Quantitative data (e.g., market size, growth rates)

#### Step 2: AI Use Case Generation:
- The **use_case_agent**:
  - Takes the research findings as input.
  - Generates five or more structured use cases, including:
    - Use case objectives
    - AI/ML/automation applications
    - Cross-functional benefits

#### Step 3: Resource Collection:
- The **resource_agent**:
  - Maps use cases to actionable resources.
  - Outputs a detailed list of:
    - Relevant datasets
    - APIs, tools, and frameworks
    - Generative AI solutions (e.g., semantic search tools, chatbots)

#### File Output:
- Resource outputs are saved as a timestamped **Markdown** file in the **output** directory for easy access and sharing.

