from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task, before_kickoff, after_kickoff
from crewai_tools import SerperDevTool, WikipediaTools, RagTool
from dotenv import load_dotenv
import os
from typing import Tuple, Union, Dict, Any
from crewai import TaskOutput
from pydantic import BaseModel

class Blog(BaseModel):
    title: str
    content: str





tasks_config = "config/tasks.yaml"

load_dotenv()

# os.environ('openai_api_key') = os.getenv('openai_api_key')
# os.environ('gemini_api_key') = os.getenv('gemini_api_key')
gemini_api_secret = os.getenv('gemini_api_key')

#for guardrails
def validate_blog_content(result: TaskOutput) -> Tuple[bool, Any]:
    """Validate blog content meets requirements."""
    try:
        # Check word count
        word_count = len(result.split())
        if word_count > 200:
            return (False, "Blog content exceeds 200 words")

        # Additional validation logic here
        return (True, result.strip())
    except Exception as e:
        return (False, "Unexpected error during validation")
    


@CrewBase
class LatestAiDevelopmentCrew():
    """LatestAiDevelopment crew"""

    agents_config = "config/agents.yaml"

    @before_kickoff
    def prepare_inputs(self, inputs):
        # Modify inputs before the crew starts
        inputs['additional_data'] = "Some extra information"
        return inputs

    @after_kickoff
    def process_output(self, output):
        # Modify output after the crew finishes
        output.raw += "\nProcessed after kickoff."
        return output

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'],
            verbose=True,
            tools=[SerperDevTool()],
            memory = True,
            respect_context_window=False, # Prefer failure over incomplete analysis
            max_retry_limit=1
        )
    

    @agent
    def reporting_analyst(self) -> Agent:
         return Agent(
      config=self.agents_config['reporting_analyst'], # type: ignore[index]
      verbose=True,
      memory=True,
      respect_context_window=True ,#auto handle context window
        
    )

    @agent
    def rag_agent(self) -> Agent:
        return Agent(
            config = self.agents_config['rag_agent'],
            tools=[RagTool()],
            respect_context_window=True,
            max_retry_limit=1,
            max_iter=50 ,#allow more iterations for complex analysis
            inject_date=True #agent to current date awareness for time sensitive data
        )
    
    @task
    def research_task(self)-> Task:
        return Task(
            config= self.tasks_config['research_task'],
            output_json=Blog
        )

    @task
    def reporting_task(self) ->Task:
        return Task(
            config=self.tasks_config['reporting_task'],
            markdown=True,
            guardrail=validate_blog_content,
            output_pydantic=Blog,

        )
    
    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[
                self.researcher(),
                self.reporting_analyst()
            ],
            tasks=[
                self.research_task(),
                self.reporting_task()
            ],
            process=Process.sequential
        )
    
result = crew.kickoff()
    

