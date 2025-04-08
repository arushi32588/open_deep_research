

from typing import Literal, Dict, List, Any, Optional

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
import json

from langgraph.constants import Send
from langgraph.graph import START, END, StateGraph
from langgraph.types import interrupt, Command

from open_deep_research.state import (
    ReportStateInput,
    ReportStateOutput,
    Sections,
    ReportState,
    SectionState,
    SectionOutputState,
    Queries,
    Feedback
)

from open_deep_research.prompts import (
    report_planner_query_writer_instructions,
    report_planner_instructions,
    query_writer_instructions, 
    section_writer_instructions,
    final_section_writer_instructions,
    section_grader_instructions,
    section_writer_inputs
)

from open_deep_research.configuration import Configuration
from open_deep_research.utils import (
    format_sections, 
    get_config_value, 
    get_search_params, 
    select_and_execute_search
)

## Nodes -- 

async def generate_report_plan(state: ReportState, config: RunnableConfig):
    """Generate the initial report plan with sections.
    
    This node:
    1. Gets configuration for the report structure and search parameters
    2. Generates search queries to gather context for planning
    3. Performs web searches using those queries
    4. Uses an LLM to generate a structured plan with sections
    
    Args:
        state: Current graph state containing the report topic
        config: Configuration for models, search APIs, etc.
        
    Returns:
        Dict containing the generated sections
    """

    # Inputs
    topic = state["topic"]
    feedback = state.get("feedback_on_report_plan", None)

    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    report_structure = configurable.report_structure
    number_of_queries = configurable.number_of_queries
    search_api = get_config_value(configurable.search_api)
    search_api_config = configurable.search_api_config or {}  # Get the config dict, default to empty
    params_to_pass = get_search_params(search_api, search_api_config)  # Filter parameters

    # Convert JSON object to string if necessary
    if isinstance(report_structure, dict):
        report_structure = str(report_structure)

    # Set writer model (model used for query writing)
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider) 
    structured_llm = writer_model.with_structured_output(Queries)

    # Format system instructions
    system_instructions_query = report_planner_query_writer_instructions.format(topic=topic, report_organization=report_structure, number_of_queries=number_of_queries)

    # Generate queries  
    results = structured_llm.invoke([SystemMessage(content=system_instructions_query),
                                     HumanMessage(content="Generate search queries that will help with planning the sections of the report.")])

    # Web search
    query_list = [query.search_query for query in results.queries]

    # Search the web with parameters
    source_str = await select_and_execute_search(search_api, query_list, params_to_pass)

    # Format system instructions
    system_instructions_sections = report_planner_instructions.format(topic=topic, report_organization=report_structure, context=source_str, feedback=feedback)

    # Set the planner
    planner_provider = get_config_value(configurable.planner_provider)
    planner_model = get_config_value(configurable.planner_model)

    # Report planner instructions
    planner_message = """Generate the sections of the report. Your response must include a 'sections' field containing a list of sections. 
                        Each section must have: name, description, plan, research, and content fields."""

    # Run the planner
    if planner_model == "claude-3-7-sonnet-latest":
        # Allocate a thinking budget for claude-3-7-sonnet-latest as the planner model
        planner_llm = init_chat_model(model=planner_model, 
                                      model_provider=planner_provider, 
                                      max_tokens=20_000, 
                                      thinking={"type": "enabled", "budget_tokens": 16_000})

    else:
        # With other models, thinking tokens are not specifically allocated
        planner_llm = init_chat_model(model=planner_model, 
                                      model_provider=planner_provider)
    
    # Generate the report sections
    structured_llm = planner_llm.with_structured_output(Sections)
    report_sections = structured_llm.invoke([SystemMessage(content=system_instructions_sections),
                                             HumanMessage(content=planner_message)])

    # Get sections
    sections = report_sections.sections

    return {"sections": sections}

def human_feedback(state: ReportState, config: RunnableConfig) -> Command[Literal["generate_report_plan","build_section_with_web_research"]]:
    """Get human feedback on the report plan and route to next steps.
    
    This node:
    1. Formats the current report plan for human review
    2. Gets feedback via an interrupt
    3. Routes to either:
       - Section writing if plan is approved
       - Plan regeneration if feedback is provided
    
    Args:
        state: Current graph state with sections to review
        config: Configuration for the workflow
        
    Returns:
        Command to either regenerate plan or start section writing
    """

    # Get sections
    topic = state["topic"]
    sections = state['sections']
    sections_str = "\n\n".join(
        f"Section: {section.name}\n"
        f"Description: {section.description}\n"
        f"Research needed: {'Yes' if section.research else 'No'}\n"
        for section in sections
    )

    # Get feedback on the report plan from interrupt
    interrupt_message = f"""Please provide feedback on the following report plan. 
                        \n\n{sections_str}\n
                        \nDoes the report plan meet your needs?\nPass 'true' to approve the report plan.\nOr, provide feedback to regenerate the report plan:"""
    
    feedback = interrupt(interrupt_message)

    # If the user approves the report plan, kick off section writing
    if isinstance(feedback, bool) and feedback is True:
        # Treat this as approve and kick off section writing
        return Command(goto=[
            Send("build_section_with_web_research", {"topic": topic, "section": s, "search_iterations": 0}) 
            for s in sections 
            if s.research
        ])
    
    # If the user provides feedback, regenerate the report plan 
    elif isinstance(feedback, str):
        # Treat this as feedback
        return Command(goto="generate_report_plan", 
                       update={"feedback_on_report_plan": feedback})
    else:
        raise TypeError(f"Interrupt value of type {type(feedback)} is not supported.")
    
def generate_queries(state: SectionState, config: RunnableConfig):
    """Generate search queries for researching a specific section.
    
    This node uses an LLM to generate targeted search queries based on the 
    section topic and description.
    
    Args:
        state: Current state containing section details
        config: Configuration including number of queries to generate
        
    Returns:
        Dict containing the generated search queries
    """

    # Get state 
    topic = state["topic"]
    section = state["section"]

    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    number_of_queries = configurable.number_of_queries

    # Generate queries 
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider) 
    structured_llm = writer_model.with_structured_output(Queries)

    # Format system instructions
    system_instructions = query_writer_instructions.format(topic=topic, 
                                                           section_topic=section.description, 
                                                           number_of_queries=number_of_queries)

    # Generate queries  
    queries = structured_llm.invoke([SystemMessage(content=system_instructions),
                                     HumanMessage(content="Generate search queries on the provided topic.")])

    return {"search_queries": queries.queries}

async def search_web(state: SectionState, config: RunnableConfig):
    """Execute web searches for the section queries.
    
    This node:
    1. Takes the generated queries
    2. Executes searches using configured search API
    3. Formats results into usable context
    
    Args:
        state: Current state with search queries
        config: Search API configuration
        
    Returns:
        Dict with search results and updated iteration count
    """

    # Get state
    search_queries = state["search_queries"]

    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    search_api = get_config_value(configurable.search_api)
    search_api_config = configurable.search_api_config or {}  # Get the config dict, default to empty
    params_to_pass = get_search_params(search_api, search_api_config)  # Filter parameters
    
    # Get local document configuration if available
    local_documents = None
    if hasattr(configurable, "local_documents") and configurable.local_documents:
        local_documents = configurable.local_documents

    # Web search
    query_list = [query.search_query for query in search_queries]

    # Search the web with parameters
    source_str = await select_and_execute_search(search_api, query_list, params_to_pass, local_documents)

    return {"source_str": source_str, "search_iterations": state["search_iterations"] + 1}

def write_section(state: SectionState, config: RunnableConfig) -> Command[Literal[END, "search_web"]]:
    """Write a section of the report and evaluate if more research is needed.
    
    This node:
    1. Writes section content using search results
    2. Evaluates the quality of the section
    3. Either:
       - Completes the section if quality passes
       - Triggers more research if quality fails
    
    Args:
        state: Current state with search results and section info
        config: Configuration for writing and evaluation
        
    Returns:
        Command to either complete section or do more research
    """

    # Get state 
    topic = state["topic"]
    section = state["section"]
    source_str = state["source_str"]

    # Get configuration
    configurable = Configuration.from_runnable_config(config)

    # Format system instructions
    section_writer_inputs_formatted = section_writer_inputs.format(topic=topic, 
                                                             section_name=section.name, 
                                                             section_topic=section.description, 
                                                             context=source_str, 
                                                             section_content=section.content)

    # Generate section  
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider) 

    section_content = writer_model.invoke([SystemMessage(content=section_writer_instructions),
                                           HumanMessage(content=section_writer_inputs_formatted)])
    
    # Write content to the section object  
    section.content = section_content.content

    # Grade prompt 
    section_grader_message = ("Grade the report and consider follow-up questions for missing information. "
                              "If the grade is 'pass', return empty strings for all follow-up queries. "
                              "If the grade is 'fail', provide specific search queries to gather missing information.")
    
    section_grader_instructions_formatted = section_grader_instructions.format(topic=topic, 
                                                                               section_topic=section.description,
                                                                               section=section.content, 
                                                                               number_of_follow_up_queries=configurable.number_of_queries)

    # Use planner model for reflection
    planner_provider = get_config_value(configurable.planner_provider)
    planner_model = get_config_value(configurable.planner_model)

    if planner_model == "claude-3-7-sonnet-latest":
        # Allocate a thinking budget for claude-3-7-sonnet-latest as the planner model
        reflection_model = init_chat_model(model=planner_model, 
                                           model_provider=planner_provider, 
                                           max_tokens=20_000, 
                                           thinking={"type": "enabled", "budget_tokens": 16_000}).with_structured_output(Feedback)
    else:
        reflection_model = init_chat_model(model=planner_model, 
                                           model_provider=planner_provider).with_structured_output(Feedback)
    # Generate feedback
    feedback = reflection_model.invoke([SystemMessage(content=section_grader_instructions_formatted),
                                        HumanMessage(content=section_grader_message)])

    # If the section is passing or the max search depth is reached, publish the section to completed sections 
    if feedback.grade == "pass" or state["search_iterations"] >= configurable.max_search_depth:
        # Publish the section to completed sections 
        return  Command(
        update={"completed_sections": [section]},
        goto=END
    )

    # Update the existing section with new content and update search queries
    else:
        return  Command(
        update={"search_queries": feedback.follow_up_queries, "section": section},
        goto="search_web"
        )
    
def write_final_sections(state: SectionState, config: RunnableConfig):
    """Write final sections of the report using web research results."""
    # ... existing code ...

def sanitize_text(text):
    """Sanitize text by removing special characters and normalizing whitespace."""
    # ... existing code ...

async def search_web(topic: str, configurable: dict):
    """Perform web search for a topic using configured search API."""
    # ... existing code ...

async def plan_report(topic: str, search_results: list, configurable: dict):
    """Plan report sections based on search results."""
    # ... existing code ...

async def write_report(topic: str, plan: str, search_results: list, configurable: dict):
    """Write report content based on plan and search results."""
    # ... existing code ...

def build_section_with_web_research(
    section_name: str,
    section_description: str,
    search_results: List[Dict[str, Any]],
    model: str = "gpt-4"
) -> Dict[str, Any]:
    """Build a section of the report using web research results."""
    # ... existing code ...

def write_final_sections(
    sections: List[Dict[str, Any]],
    search_results: List[Dict[str, Any]],
    model: str = "gpt-4"
) -> List[Dict[str, Any]]:
    """Write final sections of the report using web research results."""
    # ... existing code ...

# Report section sub-graph -- 

# Add nodes 
section_builder = StateGraph(SectionState, output=SectionOutputState)
section_builder.add_node("generate_queries", generate_queries)
section_builder.add_node("search_web", search_web)
section_builder.add_node("write_section", write_section)

# Add edges
section_builder.add_edge(START, "generate_queries")
section_builder.add_edge("generate_queries", "search_web")
section_builder.add_edge("search_web", "write_section")

# Outer graph for initial report plan compiling results from each section -- 

# Add nodes
builder = StateGraph(ReportState, input=ReportStateInput, output=ReportStateOutput, config_schema=Configuration)
builder.add_node("generate_report_plan", generate_report_plan)
builder.add_node("human_feedback", human_feedback)
builder.add_node("build_section_with_web_research", section_builder.compile())
builder.add_node("write_final_sections", write_final_sections)

# Add edges
builder.add_edge(START, "generate_report_plan")
builder.add_edge("generate_report_plan", "human_feedback")
builder.add_edge("build_section_with_web_research", "write_final_sections")
builder.add_edge("write_final_sections", END)

graph = builder.compile()
def sanitize_text(text):
    """Sanitize text by replacing problematic Unicode characters with ASCII equivalents"""
    if not isinstance(text, str):
        return text
    
    # Replace smart quotes with regular quotes
    text = text.replace('\u201c', '"')  # Opening smart quote
    text = text.replace('\u201d', '"')  # Closing smart quote
    text = text.replace('\u2018', "'")  # Opening smart single quote
    text = text.replace('\u2019', "'")  # Closing smart single quote
    
    return text

async def search_web(topic: str, configurable: dict):
    """Search the web for information on a topic"""
    try:
        # ... existing code ...
        
        # Sanitize the search results
        for result in results:
            if isinstance(result, dict):
                for key in result:
                    if isinstance(result[key], str):
                        result[key] = sanitize_text(result[key])
        
        return results
    except Exception as e:
        print(f"Error in search_web: {str(e)}")
        return []

async def plan_report(topic: str, search_results: list, configurable: dict):
    """Plan the structure of the report"""
    try:
        # ... existing code ...
        
        # Sanitize the plan before returning
        if isinstance(plan, str):
            plan = sanitize_text(plan)
        elif isinstance(plan, dict):
            for key in plan:
                if isinstance(plan[key], str):
                    plan[key] = sanitize_text(plan[key])
        
        return plan
    except Exception as e:
        print(f"Error in plan_report: {str(e)}")
        return ""

async def write_report(topic: str, plan: str, search_results: list, configurable: dict):
    """Write the report based on the plan and search results"""
    try:
        # ... existing code ...
        
        # Sanitize the report before returning
        if isinstance(report, str):
            report = sanitize_text(report)
        elif isinstance(report, dict):
            for key in report:
                if isinstance(report[key], str):
                    report[key] = sanitize_text(report[key])
        
        return report
    except Exception as e:
        print(f"Error in write_report: {str(e)}")
        return ""

def build_section_with_web_research(
    section_name: str,
    section_description: str,
    search_results: List[Dict[str, Any]],
    model: str = "gpt-4"
) -> Dict[str, Any]:
    """
    Build a section of the report using web research results.
    
    Args:
        section_name: Name of the section to build
        section_description: Description of what should be in the section
        search_results: List of search results to use
        model: Model to use for generation
        
    Returns:
        Dict containing the section content and metadata
    """
    # Format search results for the prompt
    formatted_results = []
    for result in search_results:
        formatted_results.append({
            "title": result.get("title", "No title"),
            "content": result.get("content", "No content"),
            "url": result.get("url", "No URL")
        })
    
    # Create the prompt
    prompt = f"""Write a detailed section about {section_name} for a research report.
    
Section Description:
{section_description}

Use the following search results to write the section. Focus on factual information and cite sources where possible:

{json.dumps(formatted_results, indent=2)}

Write a comprehensive section that:
1. Focuses on the most relevant and recent information
2. Synthesizes information from multiple sources
3. Includes specific data points and statistics where available
4. Maintains a professional and objective tone
5. Cites sources using [Source: title] format

Section Content:"""

    # Generate the section content
    llm = ChatOpenAI(model=model, temperature=0)
    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content

    return {
        "name": section_name,
        "content": content,
        "status": "completed",
        "sources": [result.get("url") for result in search_results if result.get("url")]
    }

def write_final_sections(
    sections: List[Dict[str, Any]],
    search_results: List[Dict[str, Any]],
    model: str = "gpt-4"
) -> List[Dict[str, Any]]:
    """
    Write final sections of the report using web research results.
    
    Args:
        sections: List of sections to write
        search_results: List of search results to use
        model: Model to use for generation
        
    Returns:
        List of completed sections
    """
    completed_sections = []
    
    for section in sections:
        if section.get("status") != "completed":
            # Build the section using web research
            completed_section = build_section_with_web_research(
                section_name=section["name"],
                section_description=section.get("description", ""),
                search_results=search_results,
                model=model
            )
            completed_sections.append(completed_section)
        else:
            completed_sections.append(section)
    
    return completed_sections



