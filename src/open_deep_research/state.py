from typing import Annotated, List, TypedDict, Literal, Dict, Any, Optional
from pydantic import BaseModel, Field
import operator
import json

class Section(BaseModel):
    name: str = Field(
        description="Name for this section of the report.",
    )
    description: str = Field(
        description="Brief overview of the main topics and concepts to be covered in this section.",
    )
    research: bool = Field(
        description="Whether to perform web research for this section of the report."
    )
    content: str = Field(
        description="The content of the section."
    )   

class Sections(BaseModel):
    sections: List[Section] = Field(
        description="Sections of the report.",
    )

class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Query for web search.")

class Queries(BaseModel):
    queries: List[SearchQuery] = Field(
        description="List of search queries.",
    )

class Feedback(BaseModel):
    grade: Literal["pass","fail"] = Field(
        description="Evaluation result indicating whether the response meets requirements ('pass') or needs revision ('fail')."
    )
    follow_up_queries: List[SearchQuery] = Field(
        description="List of follow-up search queries.",
    )

class ReportStateInput(TypedDict):
    topic: str # Report topic
    
class ReportStateOutput(TypedDict):
    final_report: str # Final report

class ReportState(TypedDict):
    topic: str # Report topic    
    feedback_on_report_plan: str # Feedback on the report plan
    sections: list[Section] # List of report sections 
    completed_sections: Annotated[list, operator.add] # Send() API key
    report_sections_from_research: str # String of any completed sections from research to write final sections
    final_report: str # Final report

class SectionState(TypedDict):
    topic: str # Report topic
    section: Section # Report section  
    search_iterations: int # Number of search iterations done
    search_queries: list[SearchQuery] # List of search queries
    source_str: str # String of formatted source content from web search
    report_sections_from_research: str # String of any completed sections from research to write final sections
    completed_sections: list[Section] # Final key we duplicate in outer state for Send() API

class SectionOutputState(TypedDict):
    completed_sections: list[Section] # Final key we duplicate in outer state for Send() API

class State:
    def __init__(self):
        self.topic: str = ""
        self.search_results: List[Dict[str, Any]] = []
        self.report_plan: str = ""
        self.report: str = ""
        self.error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary with proper encoding"""
        try:
            return {
                "topic": self.topic,
                "search_results": self.search_results,
                "report_plan": self.report_plan,
                "report": self.report,
                "error": self.error
            }
        except Exception as e:
            print(f"Error converting state to dict: {str(e)}")
            return {}

    def to_json(self) -> str:
        """Convert state to JSON string with proper encoding"""
        try:
            return json.dumps(self.to_dict(), ensure_ascii=False)
        except Exception as e:
            print(f"Error converting state to JSON: {str(e)}")
            return "{}"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'State':
        """Create state from dictionary with proper encoding"""
        state = cls()
        try:
            state.topic = str(data.get("topic", ""))
            state.search_results = data.get("search_results", [])
            state.report_plan = str(data.get("report_plan", ""))
            state.report = str(data.get("report", ""))
            state.error = str(data.get("error")) if data.get("error") else None
        except Exception as e:
            print(f"Error creating state from dict: {str(e)}")
        return state

    @classmethod
    def from_json(cls, json_str: str) -> 'State':
        """Create state from JSON string with proper encoding"""
        try:
            data = json.loads(json_str)
            return cls.from_dict(data)
        except Exception as e:
            print(f"Error creating state from JSON: {str(e)}")
            return cls()
