
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import TokenTextSplitter, SentenceSplitter
from llama_index.core import Settings, SimpleDirectoryReader
from llama_index.core.llms import ChatMessage
from pathlib import Path
import os
import logging
import prompt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration parameters

# Configuration parameters
BASE_INPUT_PATH = "data-test/processed/ukr_laws/"  # Base path for input files
BASE_OUTPUT_PATH = "data-test/processed/summarization"  # Base path for output files
CONTEXT_WINDOW = 500  # Context window size in tokens
CHUNK_OVERLAP = 30  # Overlap between chunks

#LLM_MODEL = "qwq:32b"
#LLM_MODEL = "phi4:14b-q8_0"
LLM_MODEL= "qwen2.5:7b-instruct-fp16"
#LLM_MODEL = "deepseek-r1:32b"

# Configure LlamaIndex settings
Settings.llm = Ollama(
    model=LLM_MODEL,
    base_url="http://localhost:11434",
    temperature=0.4,
    request_timeout=60000
)

Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

Settings.text_splitter = SentenceSplitter(
    chunk_size=CONTEXT_WINDOW,
    chunk_overlap=CHUNK_OVERLAP,
    paragraph_separator='Стаття'
)

def get_output_path(input_file_path: str) -> Path:
    """Generate output path maintaining the same structure as input."""
    relative_path = os.path.relpath(input_file_path, BASE_INPUT_PATH)
    output_path = Path(BASE_OUTPUT_PATH) / relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path

def summarize_text(text: str) -> str:
    """Summarize text using Ollama."""
    response = Settings.llm.chat([
        ChatMessage(role="system", content=prompt.prompt_ukr_lawer), 
                                 ChatMessage(role="user", content=text)
                                             ])
    return response.message.content

def process_document(doc) -> str:
    """Process document by chunks if needed and summarize."""
    if len(doc.text) <= CONTEXT_WINDOW:
        return summarize_text(doc.text)
    
    # Split into chunks if document is too large
    chunks = Settings.text_splitter.split_text(doc.text)
    summaries = []
    
    for chunk in chunks:
        print("=================================")
        print(f"Original CHUNK: " + chunk)
        print("=================================")
        summary = summarize_text(chunk)
        print(summary)
        summaries.append(summary)
    
    # Combine summaries if needed
    final_text = "\n\n".join(summaries)    
    return final_text

import tiktoken
def estimate_tokens(text):
    encoding = tiktoken.get_encoding("cl100k_base")
    toneks_count = len(encoding.encode(text))
    print(f"Document rough tokens estimate: {toneks_count}")
    return toneks_count

def main():
    """Main function to process all files."""
    # Walk through all files in input directory
    for root, _, files in os.walk(BASE_INPUT_PATH):
        for file in files:
            if not file.endswith('.txt'):  # Adjust file extension as needed
                continue
                
            input_file_path = os.path.join(root, file)
            logger.info(f"Processing file: {input_file_path}")
            
            try:
                # Load document
                documents = SimpleDirectoryReader(
                    input_files=[input_file_path]
                ).load_data()
                
                if not documents:
                    logger.warning(f"No content found in {input_file_path}")
                    continue
                
                estimate_tokens(documents[0].text)

                # Process and summarize
                summarized_text = process_document(documents[0])
                
                # Save to output maintaining directory structure
                output_path = get_output_path(input_file_path)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(summarized_text)
                
                logger.info(f"Saved summary to: {output_path}")
                
            except Exception as e:
                logger.error(f"Error processing {input_file_path}: {str(e)}")

if __name__ == "__main__":
    main()