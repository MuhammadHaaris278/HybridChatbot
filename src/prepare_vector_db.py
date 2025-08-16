import os
import yaml
from pyprojroot import here
from langchain_qdrant import QdrantVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from dotenv import load_dotenv


class PrepareVectorDB:
    """
    A class to prepare and manage a Qdrant Vector Database using documents from a specified directory.
    The class performs the following tasks:
    - Loads and splits documents (PDFs).
    - Splits the text into chunks based on the specified chunk size and overlap.
    - Embeds the document chunks using a specified embedding model.
    - Stores the embedded vectors in a Qdrant collection.

    Attributes:
        doc_dir (str): Path to the directory containing documents (PDFs) to be processed.
        chunk_size (int): The maximum size of each chunk (in characters) into which the document text will be split.
        chunk_overlap (int): The number of overlapping characters between consecutive chunks.
        embedding_model (str): The name of the embedding model to be used for generating vector representations of text.
        vectordb_dir (str): Directory where the Qdrant database will be stored (for local setup).
        collection_name (str): The name of the collection to be used within the Qdrant database.
        qdrant_url (str, optional): URL for a remote Qdrant server, or None for local storage.

    Methods:
        path_maker(file_name: str, doc_dir: str) -> str:
            Creates a full file path by joining the given directory and file name.

        run() -> None:
            Executes the process of reading documents, splitting text, embedding them into vectors, and 
            saving the resulting Qdrant database.
    """

    def __init__(self,
                 doc_dir: str,
                 chunk_size: int,
                 chunk_overlap: int,
                 embedding_model: str,
                 vectordb_dir: str,
                 collection_name: str,
                 qdrant_url: str = None
                 ) -> None:
        self.doc_dir = doc_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.vectordb_dir = vectordb_dir
        self.collection_name = collection_name
        self.qdrant_url = qdrant_url

    def path_maker(self, file_name: str, doc_dir: str) -> str:
        """
        Creates a full file path by joining the given directory and file name.

        Args:
            file_name (str): The name of the file.
            doc_dir (str): The directory containing the documents.

        Returns:
            str: The full file path.
        """
        return str(here(os.path.join(doc_dir, file_name)))

    def run(self) -> None:
        """
        Executes the process of reading documents, splitting text, embedding them into vectors,
        and saving the resulting Qdrant database.
        """
        # Check if the doc directory exists
        if not os.path.exists(here(self.doc_dir)):
            raise FileNotFoundError(f"Directory not found: {here(self.doc_dir)}. Please create the directory and add PDF files.")

        # Create vector database directory if it doesn't exist
        if not self.qdrant_url and not os.path.exists(here(self.vectordb_dir)):
            os.makedirs(here(self.vectordb_dir))
            print(f"Directory '{self.vectordb_dir}' was created.")

        # Load and split documents
        file_list = os.listdir(here(self.doc_dir))
        docs = [PyPDFLoader(self.path_maker(fn, self.doc_dir)).load_and_split() for fn in file_list]
        docs_list = [item for sublist in docs for item in sublist]

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        doc_splits = text_splitter.split_documents(docs_list)

        # Create Qdrant vector store
        QdrantVectorStore.from_documents(
            documents=doc_splits,
            embedding=OpenAIEmbeddings(
                model=self.embedding_model,
                base_url="https://models.github.ai/inference",
                api_key=os.getenv("GITHUB_TOKEN")
            ),
            url=self.qdrant_url,
            path=here(self.vectordb_dir) if not self.qdrant_url else None,
            collection_name=self.collection_name,
            force_recreate=True
        )
        print("Qdrant vector store is created and saved.")
        client = QdrantClient(
            url=self.qdrant_url,
            path=here(self.vectordb_dir) if not self.qdrant_url else None
        )
        print("Number of vectors in Qdrant collection:",
              client.get_collection(self.collection_name).points_count, "\n\n")


if __name__ == "__main__":
    load_dotenv()
    os.environ['OPENAI_API_KEY'] = os.getenv("GITHUB_TOKEN")

    with open(here("configs/tools_config.yml")) as cfg:
        app_config = yaml.load(cfg, Loader=yaml.FullLoader)

    # Configs for clinical notes
    chunk_size = app_config["clinical_notes_rag"]["chunk_size"]
    chunk_overlap = app_config["clinical_notes_rag"]["chunk_overlap"]
    embedding_model = app_config["clinical_notes_rag"]["embedding_model"]
    vectordb_dir = app_config["clinical_notes_rag"]["vectordb"]
    collection_name = app_config["clinical_notes_rag"]["collection_name"]
    doc_dir = app_config["clinical_notes_rag"]["unstructured_docs"]
    qdrant_url = app_config["clinical_notes_rag"]["qdrant_url"]

    prepare_db_instance = PrepareVectorDB(
        doc_dir=doc_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=embedding_model,
        vectordb_dir=vectordb_dir,
        collection_name=collection_name,
        qdrant_url=qdrant_url
    )

    prepare_db_instance.run()