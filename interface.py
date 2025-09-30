from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import inspect
import json
import random
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple, Type
from pydantic import BaseModel, create_model, Field
import sys
import inspect
import sys, os
sys.path.insert(0, os.path.abspath("OpenManus"))
print("sys.path:", sys.path)
from OpenManus.app.tool.python_execute import PythonExecute
from OpenManus.app.tool.note import NoteTool
import os
import sys
import sys
import os
import asyncio
from OpenManus.app.agent.manus import Manus
from OpenManus.app.tool import ToolCollection
from OpenManus.app.prompt.manus import NEXT_STEP_PROMPT

class EmptyParams(BaseModel):
    pass

class BaseTool(ABC, BaseModel):
    name: str
    description: str
    parameters: Optional[dict] = None

    class Config:
        arbitrary_types_allowed = True

    async def __call__(self, **kwargs) -> Any:
        """Execute the tool with given parameters."""
        return await self.execute(**kwargs)

    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """Execute the tool with given parameters."""

    def to_param(self) -> Dict:
        """Convert tool to function call format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

def create_tools_from_environment(env: Any) -> List[BaseTool]:
    """
    Automatically scan environment instance and create tools for its public methods marked with '[agent tool]'.
    """
    tools: List[BaseTool] = []
    AGENT_TOOL_TAG = "agent tool"

    for method_name, method in inspect.getmembers(env, inspect.ismethod):
        if method_name.startswith('_'):
            continue

        docstring = inspect.getdoc(method)
        if not docstring or AGENT_TOOL_TAG not in docstring.lower():
            continue

        clean_description = docstring # Keep description part unchanged

        sig = inspect.signature(method)
        pydantic_fields: Dict[str, Tuple[Type, Any]] = {}
        required_params = [] 
        for param in sig.parameters.values():
            if param.name == 'self':
                continue
            
            param_type = param.annotation if param.annotation != inspect.Parameter.empty else Any
            if param.default == inspect.Parameter.empty:
                default_value = ... # Required field
                required_params.append(param.name)
            else:
                default_value = param.default
            pydantic_fields[param.name] = (param_type, default_value)

        # Dynamically create Pydantic model for parameter validation
        # If no parameters, create an empty BaseModel to avoid create_model error
        if not pydantic_fields:
            # ParamsModel = BaseModel
            # tool_parameters = {} # Empty dict if no parameters
            ParamsModel = EmptyParams
            tool_parameters = {
                "type": "object",
                "properties": {},
                "required": []
            }
        else:
            ParamsModel = create_model(f"{method_name.capitalize()}Params", **pydantic_fields)
            param_schema = ParamsModel.schema()
            tool_parameters = {
                "type": "object",
                "properties": param_schema.get("properties", {}),
                "required": required_params, 
            }
        
        # Dynamically create Tool class, execute method remains unchanged
        def create_execute_func(bound_method, params_model):
            async def execute(self, **kwargs) -> Any:
                try:
                    if params_model == BaseModel and not kwargs:
                        validated_args = {}
                    else:
                        validated_args = params_model(**kwargs).dict()

                    result = bound_method(**validated_args)

                    # If it's a coroutine, await it; otherwise return directly
                    if inspect.isawaitable(result):
                        result = await result

                    return result
                except Exception as e:
                    return f"Error executing tool {self.name}: {e}"
            return execute


        # Use type() to dynamically build tool class
        NewToolClass = type(
            f"{method_name.capitalize()}Tool",
            (BaseTool,),
            {
                "execute": create_execute_func(method, ParamsModel),
                # Pydantic model needs a parameterless __init__, we use default_factory to initialize
                # Actually, we can pass parameters more directly during instantiation
                "__pydantic_init__": lambda **kwargs: kwargs,
            }
        )
        
        # Instantiate dynamically created tool class and pass parameters during creation
        tool_instance = NewToolClass(name=method_name, description=clean_description, parameters=tool_parameters)
        tools.append(tool_instance)
        print(f"✅ Successfully created tool: '{method_name}' with parameters: {json.dumps(tool_parameters)}")

    return tools

from envs.grid_env.env import MysteryGridEnvironment
from envs.seq_env.env import SequenceExploreEnvironment
from envs.bio_env.env import GeneticsLabEnvironment
from envs.common import Difficulty


# select env here



async def main():
    SELECTED_ENV = MysteryGridEnvironment
    REQUIRED_ENV_STEPS = 50
    env_name = SELECTED_ENV.__name__
    print(f"🚀 Starting Single Experiment on Env: {env_name}")

    env = SELECTED_ENV(difficulty=Difficulty.HARD, required_steps=REQUIRED_ENV_STEPS)
    
    tools = create_tools_from_environment(env)
    other_tools = [PythonExecute(), NoteTool()]

    tools.extend(other_tools)
    tool_collection = ToolCollection(*tools)
    print("[All tools]:", tools)

    agent = Manus()
    agent.available_tools = tool_collection
    prompt = env.env_prompt
    async def run_agent():
        await agent.run(prompt)  # Run indefinitely, agent internally loops and calls tools until cancelled

    async def monitor_env():
        interval = 0.5

        while (not env.committed) and (agent.current_step < agent.max_steps): #[TODO]
            await asyncio.sleep(interval)

        if env.committed:
            print("✅ Agent has committed the result.")
        else:
            output_log = "❌ Agent didn't commit any result."
            if agent.current_step >= agent.max_steps:
                output_log += " Max steps reached."
            print(output_log)

        # End agent regardless of commit status
        agent_task.cancel()


    # Use asyncio to run agent and environment monitor concurrently
    agent_task = asyncio.create_task(run_agent())
    monitor_task = asyncio.create_task(monitor_env())

    try:
        await asyncio.gather(agent_task, monitor_task)
    except asyncio.CancelledError:
        print("�� Agent task was cancelled due to environment termination.")
    finally:
        await agent.cleanup()
        # save notes
        
        notes = await agent.available_tools.get_tool("note_tool").execute(action="check_notes")
        notes = "\n".join(notes['notes'])
        with open("notes.txt", "w") as f:
            f.write(notes)

async def run_single_experiment(ENV, REQUIRED_ENV_STEPS, experiment_id: int, output_dir: str, max_run_steps: int, free: bool):
    env_name = ENV.__name__
    print(f"🚀 Starting Experiment {experiment_id} on Env: {env_name}")
    env = ENV(difficulty=Difficulty.HARD, required_steps=REQUIRED_ENV_STEPS, free=free)
    tools = create_tools_from_environment(env)
    other_tools = [PythonExecute(), NoteTool()]
    tools.extend(other_tools)
    tool_collection = ToolCollection(*tools)

    agent = Manus()
    agent.set_max_run_steps(max_run_steps)
    agent.available_tools = tool_collection
    prompt = env.env_prompt
    agent.set_system_prompt(prompt)

    async def run_agent():
        await agent.run(NEXT_STEP_PROMPT)  # Run indefinitely, agent internally loops and calls tools until cancelled

    async def monitor_env():
        interval = 0.5

        while (not env.committed) and (agent.current_step < agent.max_steps): #[TODO]
            await asyncio.sleep(interval)

        if env.committed:
            print("✅ Agent has committed the result.")
        else:
            output_log = "❌ Agent didn't commit any result."
            if agent.current_step >= agent.max_steps:
                output_log += " Max steps reached."
            print(output_log)

        # End agent regardless of commit status
        agent_task.cancel()


    # Use asyncio to run agent and environment monitor concurrently
    agent_task = asyncio.create_task(run_agent())
    monitor_task = asyncio.create_task(monitor_env())

    # new
    try:
        await asyncio.gather(agent_task, monitor_task)
    except asyncio.CancelledError:
        print(f"🛑 Experiment {experiment_id} finished.")
    finally:
        await agent.cleanup()

        llm = agent.llm
        total_completion_tokens = llm.total_completion_tokens
        total_input_tokens = llm.total_input_tokens
        token_statistics = {
            "total_completion_tokens": total_completion_tokens,
            "total_input_tokens": total_input_tokens,
            "total_tokens": total_completion_tokens + total_input_tokens
        }

        notes = await agent.available_tools.get_tool("note_tool").execute(action="check_notes")
        notes = "\n".join(notes['notes'])
        notes_path = os.path.join(output_dir, "notes.txt")
        with open(notes_path, "w") as f:
            f.write(notes)
        
        output = env.final_result
        output["token_statistics"] = token_statistics
        output_path = os.path.join(output_dir, "eval_output.json")
        with open(output_path, "w") as f:
            json.dump(output, f, indent=4)

        print(f"🛑 Experiment {experiment_id} completed. Results saved to {output_path} and notes saved to {notes_path}.")

if __name__ == "__main__":
    asyncio.run(main())