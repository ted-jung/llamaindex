# Run steps concurrently in workflows
# Date: 14, Jan 2025
# Writer: Ted, Jung
# When: Resuming interrupted processes, 
#       Efficient development and debugging
#       Large-scale indexing, 
#       Sharing and Collaboration

import os
import asyncio

from llama_index.core.workflow import (
    Workflow,
    step,
    StartEvent,
    StopEvent,
    Event,
    Context,
)

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.core.workflow.checkpointer import WorkflowCheckpointer

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
llm = Ollama(model="llama3.2", request_timeout=720.0)


class JokeEvent(Event):
    joke: str


class JokeFlow(Workflow):

    @step
    async def generate_joke(self, ev: StartEvent) -> JokeEvent:
        topic = ev.topic

        prompt = f"Write your best joke about {topic}."
        response = await llm.acomplete(prompt)
        return JokeEvent(joke=str(response))

    @step
    async def critique_joke(self, ev: JokeEvent) -> StopEvent:
        joke = ev.joke

        prompt = f"Give a thorough analysis and critique of the following joke: {joke}"
        response = await llm.acomplete(prompt)
        return StopEvent(result=str(response))


async def main():
    # instantiate Jokeflow
    workflow = JokeFlow(timeout=360, verbose=False)
    wflow_ckptr = WorkflowCheckpointer(workflow=workflow)

    handler = wflow_ckptr.run(
        topic="chemistry",
    )
    await handler
    print(handler)

    print(wflow_ckptr.checkpoints)
    
    for run_id, ckpts in wflow_ckptr.checkpoints.items():
        print(f"Run: {run_id} has {len(ckpts)} stored checkpoints")
        
    additional_topics = ["biology", "history"]

    for topic in additional_topics:
        handler = wflow_ckptr.run(topic=topic)
        await handler
        

    # Filter by the name of last completed step
    checkpoints_right_after_generate_joke_step = wflow_ckptr.filter_checkpoints(
        last_completed_step="generate_joke",
    )

    # checkpoint ids
    [ckpt for ckpt in checkpoints_right_after_generate_joke_step]


    # Re-run workflow from a specific checkpoint
    # can work with a new instance
    new_workflow_instance = JokeFlow(timeout=360, verbose=False)
    wflow_ckptr.workflow = new_workflow_instance

    ckpt = checkpoints_right_after_generate_joke_step[0]

    handler = wflow_ckptr.run_from(checkpoint=ckpt)
    await handler

    print(handler)
    
    for run_id, ckpts in wflow_ckptr.checkpoints.items():
        print(f"Run: {run_id} has {len(ckpts)} stored checkpoints")
    
    ted_rslt = wflow_ckptr.enabled_checkpoints
    print(ted_rslt)
    
    wflow_ckptr.disable_checkpoint(step="critique_joke")

    ted_rslt = wflow_ckptr.enabled_checkpoints
    print(ted_rslt)
    
    handler = wflow_ckptr.run(topic="cars")
    await handler

    for run_id, ckpts in wflow_ckptr.checkpoints.items():
        print(
            f"Run: {run_id} has stored checkpoints for steps {[c.last_completed_step for c in ckpts]}"
        )

    wflow_ckptr.enable_checkpoint(step="critique_joke")
    
    handler = wflow_ckptr.run(topic="cars")
    await handler

    for run_id, ckpts in wflow_ckptr.checkpoints.items():
        print(
            f"Run: {run_id} has stored checkpoints for steps {[c.last_completed_step for c in ckpts]}"
        )


if __name__ == "__main__":
    asyncio.run(main())