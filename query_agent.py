import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_classic.chains.query_constructor.base import AttributeInfo
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
from rich.console import Console
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text

# Load environment variables from .env file
load_dotenv()

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

def main():
    console = Console()

    # 1. Define the exact same embedding model Chroma used by default when we loaded the data
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 2. Connect to the existing Chroma database
    vectorstore = Chroma(
        collection_name="beats_collection",
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    
    # 3. Tell the LLM what metadata attributes are available to filter by
    metadata_field_info = [
        AttributeInfo(name="bpm", description="The tempo of the beat in BPM", type="integer"),
        AttributeInfo(name="key", description="The musical key of the beat (e.g. 'C Minor', 'G Major')", type="string"),
        AttributeInfo(name="mood", description="The descriptive mood of the beat", type="string"),
        AttributeInfo(name="overall_energy", description="The overall energy / RMS of the beat", type="float"),
        AttributeInfo(name="bounciness", description="The bounciness or onset density", type="float"),
        AttributeInfo(name="filename", description="The file name of the beat", type="string"),
        AttributeInfo(name="filepath", description="The full path to the beat", type="string"),
    ]
    
    document_content_description = "A description of a musical beat's vibe, tempo, key, energy, and mood."
    
    # 4. Initialize the LLM (this acts as the brain to construct the query)
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    
    # 5. Create the Self-Query Retriever
    retriever = SelfQueryRetriever.from_llm(
        llm,
        vectorstore,
        document_content_description,
        metadata_field_info,
        enable_limit=True, # allows the user to say "find me 2 beats"
    )
    
    # 6. Interactive search loop
    console.print(Panel(Text("BeatRAG AI Agent loaded!", justify="center"), title="Welcome", border_style="green"))
    while True:
        user_input = console.input("\n[bold cyan]What kind of beat are you looking for?[/] ('exit' to quit)\n> ")
        if user_input.lower() in ['exit', 'quit']:
            break
        
        with console.status("[bold yellow]Analyzing request and searching database...[/]") as status:
            results = retriever.invoke(user_input)
        
        if not results:
            console.print("[bold red]No beats found matching that description.[/]")
            continue
            
        console.print(f"\n[bold green]Found {len(results)} matches:[/]\n")
        for i, doc in enumerate(results, 1):
            mood = doc.metadata.get('mood', 'N/A')
            key = doc.metadata.get('key', 'N/A')
            bpm = doc.metadata.get('bpm', 'N/A')
            filename = doc.metadata.get('filename', 'N/A')
            filepath = doc.metadata.get('filepath', 'N/A')


            panel_content = Text()
            panel_content.append(f"[bold cyan]File:[/bold] {filename}\n")
            panel_content.append(f"[bold cyan]BPM:[/bold] {bpm} | [bold cyan]Key:[/bold] {key} | [bold cyan]Mood:[/bold] {mood}\n")
            panel_content.append(f"[bold cyan]Path:[/bold] {filepath}\n")

            console.print(Panel(panel_content, title=f"Beat {i}", border_style="cyan"))


if __name__ == "__main__":
    main()