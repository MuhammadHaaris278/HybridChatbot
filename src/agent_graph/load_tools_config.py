import os
import yaml
from dotenv import load_dotenv
from pyprojroot import here

load_dotenv()


class LoadToolsConfig:
    def __init__(self) -> None:
        with open(here("configs/tools_config.yml")) as cfg:
            app_config = yaml.load(cfg, Loader=yaml.FullLoader)

        # Set environment variables
        os.environ['OPENAI_API_KEY'] = os.getenv("OPEN_AI_API_KEY")
        os.environ['TAVILY_API_KEY'] = os.getenv("TAVILY_API_KEY")

        # Primary agent
        self.primary_agent_llm = app_config["primary_agent"]["llm"]
        self.primary_agent_llm_temperature = app_config["primary_agent"]["llm_temperature"]

        # Internet Search config
        self.tavily_search_max_results = int(
            app_config["tavily_search_api"]["tavily_search_max_results"])

        # Clinical Notes RAG configs
        self.clinical_notes_rag_llm = app_config["clinical_notes_rag"]["llm"]
        self.clinical_notes_rag_llm_temperature = float(
            app_config["clinical_notes_rag"]["llm_temperature"])
        self.clinical_notes_rag_embedding_model = app_config["clinical_notes_rag"]["embedding_model"]
        self.clinical_notes_rag_vectordb_directory = str(here(
            app_config["clinical_notes_rag"]["vectordb"]))
        self.clinical_notes_rag_unstructured_docs_directory = str(here(
            app_config["clinical_notes_rag"]["unstructured_docs"]))
        self.clinical_notes_rag_k = app_config["clinical_notes_rag"]["k"]
        self.clinical_notes_rag_chunk_size = app_config["clinical_notes_rag"]["chunk_size"]
        self.clinical_notes_rag_chunk_overlap = app_config["clinical_notes_rag"]["chunk_overlap"]
        self.clinical_notes_rag_collection_name = app_config["clinical_notes_rag"]["collection_name"]
        self.clinical_notes_rag_qdrant_url = app_config["clinical_notes_rag"]["qdrant_url"]

        # Hospital SQL agent configs
        self.hospital_sqlagent_llm = app_config["hospital_sqlagent_configs"]["llm"]
        self.hospital_sqlagent_llm_temperature = float(
            app_config["hospital_sqlagent_configs"]["llm_temperature"])

        # Graph configs
        self.thread_id = str(
            app_config["graph_configs"]["thread_id"])