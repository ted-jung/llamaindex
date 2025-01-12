from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Event,
)
from llama_index.core.workflow import draw_all_possible_flows
import random

class FirstEvent(Event):
    first_output: str

class SecondEvent(Event):
    second_output: str

class LoopEvent(Event):
    loop_output: str

class MyWorkflow(Workflow):
    @step
    async def step_one(self, ev: StartEvent | LoopEvent) -> FirstEvent | LoopEvent:
        # print(ev.first_input)
        if random.randint(0, 1) == 0:
            print("0")
            return LoopEvent(loop_output="Back to step one.")
        else:
            print("1")
            return FirstEvent(first_output="First step complete.")

    @step
    async def step_two(self, ev: FirstEvent) -> SecondEvent:
        print(ev.first_output)
        return SecondEvent(second_output="Second step complete.")

    @step
    async def step_three(self, ev: SecondEvent) -> StopEvent:
        print(ev.second_output)
        return StopEvent(result="Workflow complete.")


async def ted_event():
    w = MyWorkflow(timeout=20, verbose=False)
    result = await w.run(first_input="Start the workflow.")
    print("=============")
    print(result)
    
    
if __name__ == "__main__":
    import asyncio
    
    asyncio.run(ted_event())
    draw_all_possible_flows(MyWorkflow, filename="basic_workflow.html")
    