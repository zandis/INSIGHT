"""
Agent implementations for INSIGHT.

This module provides the boss and worker agents that orchestrate
the research workflow using LLM-powered task planning and execution.
"""

import os
from ast import literal_eval
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

from colorama import Fore

import openai

from config import OPENAI_API_KEY
from utils import (
    generate_tool_prompt,
    get_gpt_chat_completion,
    get_gpt_completion,
    query_knowledge_base,
)

# Import enhanced modules if available
try:
    from logging_config import get_logger
    from metrics import get_metrics
    ENHANCED_MODE = True
    logger = get_logger("agents")
except ImportError:
    import logging
    ENHANCED_MODE = False
    logger = logging.getLogger(__name__)

# Initialize API key
openai.api_key = OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY", "")


def boss_agent(
    objective: str,
    tool_description: str,
    task_list: Union[List[str], Deque[str]],
    summaries: List[str],
    completed_tasks: List[str],
    previous_task: str = "",
    previous_result: Optional[Any] = None
) -> Deque[str]:
    """
    Boss agent that plans and prioritizes research tasks.

    The boss agent breaks down the high-level objective into manageable
    tasks for worker agents, considering completed work and available tools.

    Args:
        objective: The high-level research objective
        tool_description: Description of available tools
        task_list: Current pending tasks
        summaries: Executive summaries of completed work
        completed_tasks: List of completed task descriptions
        previous_task: Most recently completed task
        previous_result: Result from the most recent task

    Returns:
        Deque of prioritized tasks to execute
    """
    
    no_result_notification = ""
    if not previous_result and previous_task:
        no_result_notification = f"Note: Task '{previous_task}' completed but returned no results. Please decide if you should retry. If so, please change something so that you will get a result."
       
    
    if summaries:
        executive_summary = '\n'.join(summaries)
    else:
        executive_summary = "No information gathered yet."

    
    print(Fore.CYAN + "\033[1m\n*****EXECUTIVE SUMMARY*****\n\033[0m")
    print(Fore.CYAN + executive_summary)

    
    system_prompt = f"""You are BossGPT, a responsible and organized agent that is responsible for completing a high level and difficult objective.
As the boss, your goal is to break the high level objective down into small and managable tasks for your workers. These tasks will be picked up by your worker agents and completed.
You will also get an executive summary of what your workers have accomplished so far. Use the summary to make decisions about what tasks to do next, what tasks to get rid of, and to reprioritize tasks.
The highest priority task will be at the top of the task list.

You also have access to some tools. You can create a task for your workers to use any of your tools. You cannot use more than one tool per task. You only have the tools specified in the tools section. Do not make up any tools.
Your worker agents update the executive summary so that you can use new information from the completed tasks to make informed decisions about what to do next. 
It is ok to create tasks that do not directly help achieve your objective but rather just serve to add useful information.

Tasks should be a simple python array with strings as elements. Priority is only determined by the order of the elements in the array.

After you have finished generating your task array, cease output.

TOOLS
{tool_description}

===

Your responses should in this format:

THOUGHTS
(Reason about what tasks to add, change, delete, or reprioritize given your objective and all the information you have)

TASKS
(Python list of tasks)

Note: To be sure that TASKS is a valid python list, it should always start with [ and always end with ]

""".strip()

    user_prompt = f"""
Here is your objective: {objective}
Here is the current task list: {task_list}
Here are the only tools you have access to: {tool_description}
Here are the tasks that have been complete thus far: {completed_tasks}
Here is an executive summary of the information gathered so far: {executive_summary}
{no_result_notification}
Note: If a task has already been completed, do not write that same task again in the task list. If you would like a worker to continue or redo a task, be sure to word it a little differently so you don't get the same result.

===
    
    """.strip()

    content = get_gpt_chat_completion(system_prompt, user_prompt, temp=0.0)

    thoughts = content[
        content.find("THOUGHTS") + len("THOUGHTS") : content.find("TASKS")
    ].strip()
    tasks = content[content.find("TASKS") + len("TASKS") :].strip()

    # Safely parse the task list with error handling
    try:
        new_task_list = literal_eval(tasks)
        if not isinstance(new_task_list, list):
            logger.error(f"Boss agent returned non-list type: {type(new_task_list)}")
            new_task_list = []
    except (ValueError, SyntaxError) as e:
        logger.error(f"Failed to parse task list from boss agent: {e}")
        logger.debug(f"Raw tasks string: {tasks[:500]}")
        # Attempt to extract tasks manually
        new_task_list = _extract_tasks_fallback(tasks)

    print(Fore.CYAN + "\033[1m\n*****BOSS THOUGHTS*****\n\033[0m")
    print(Fore.CYAN + thoughts)

    return deque(new_task_list)


def _extract_tasks_fallback(tasks_str: str) -> List[str]:
    """
    Fallback parser for malformed task lists.

    Args:
        tasks_str: Raw string containing tasks

    Returns:
        List of extracted tasks
    """
    import re

    # Try to find array-like structure
    match = re.search(r'\[([^\]]+)\]', tasks_str, re.DOTALL)
    if match:
        content = match.group(1)
        # Extract quoted strings
        tasks = re.findall(r'["\']([^"\']+)["\']', content)
        if tasks:
            logger.info(f"Fallback parser extracted {len(tasks)} tasks")
            return tasks

    # Last resort: split by newlines and clean
    lines = [line.strip().strip('-').strip('*').strip()
             for line in tasks_str.split('\n')
             if line.strip() and not line.strip().startswith(('[', ']'))]

    logger.warning(f"Using line-based fallback, extracted {len(lines)} potential tasks")
    return lines[:10]  # Limit to prevent runaway lists


def worker_agent(
    objective: str,
    task: str,
    index: Any,
    cache: Dict[str, List[str]],
    TOOLS: List[str],
) -> Tuple[str, bool]:
    """
    Worker agent that executes individual research tasks.

    The worker agent takes a task from the boss and executes it using
    the appropriate tool (API wrapper) or general knowledge.

    Args:
        objective: The high-level research objective for context
        task: The specific task to execute
        index: The LlamaIndex knowledge base for context retrieval
        cache: Cache of previous API call parameters
        TOOLS: List of available tool names

    Returns:
        Tuple of (result_string, is_python_code) where is_python_code
        indicates if the result is API wrapper code to execute
    """
    result_is_python = False
    context = ""
    previous_params = ""

    if index.docstore.docs:
        context = query_knowledge_base(
            index,
            query=f"Provide as much useful context as possible for this task: {task}",
        )

    
    if any(tool in task for tool in TOOLS):
        result_is_python = True
    
    for tool in TOOLS:
        if tool in task and tool in cache:
            previous_params = cache[tool]
            break

    if result_is_python:
        prompt = f"""You are an AI who performs one task based on the following objective: {objective}. You will be writing code for your task. Here are the parameters used in the code from previous tasks {previous_params}. Do not use the same parameters and query again; instead use tweak the parameters or query so that you get a slightly different result."""
        prompt += generate_tool_prompt(task)
    else:
        prompt = f"""You are an AI who performs one task based on the following objective: {objective}. Here is the result of some similar previous tasks: {context}. Try to not produce the same result. Be creative so that we can get new information."""

    prompt += f"\nYour task: {task}\nResponse:"

    # Use modern model with fallback
    try:
        response = get_gpt_chat_completion(
            system_prompt="You are an AI research assistant that completes tasks accurately and efficiently.",
            user_prompt=prompt,
            temp=0.0
        )
    except Exception as e:
        logger.warning(f"Chat completion failed, falling back to completion API: {e}")
        response = get_gpt_completion(prompt, engine="gpt-3.5-turbo-instruct", temp=0.0)

    return response, result_is_python


def data_cleaning_agent(result: str, objective: str) -> str:
    """
    Clean and organize raw data results.

    Args:
        result: Raw data to clean
        objective: Research objective for context

    Returns:
        Cleaned and organized data string
    """
    system_prompt = "You are an AI who summarizes, cleans, and organizes data. Do not delete any potentially useful information. Be concise but comprehensive."
    user_prompt = f"""Clean and organize the following data:

Data: {result}

Cleaned Data:"""

    try:
        response = get_gpt_chat_completion(system_prompt, user_prompt, temp=0.1)
    except Exception as e:
        logger.warning(f"Data cleaning failed: {e}")
        response = result  # Return original if cleaning fails

    return response
