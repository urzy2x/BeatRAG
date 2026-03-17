import json
import os
import chromadb
from rich.console import Console
from rich.panel import Panel

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(BASE_DIR, "sample_database.json")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

def main(file_paths=None):
    console = Console()
    console.print(Panel("[bold green]Loading beats into ChromaDB[/bold green]"))

    if not file_paths:
        if not os.path.exists(JSON_PATH):
            console.print(f"[bold red]JSON file not found at {JSON_PATH}. Run extractor.py first![/bold red]")
            return
        with open(JSON_PATH, "r") as f:
            beats = json.load(f)
    else:
        beats = []
        for file_path in file_paths:
            with open(file_path, "r") as f:
                beats.extend(json.load(f))

    # 1. Initialize ChromaDB persistent client
    console.print(f"Initializing ChromaDB at: {CHROMA_PATH}")
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # 2. Get or create a collection
    collection = client.get_or_create_collection(name="beats_collection")

    console.print(f"Loaded {len(beats)} beats. Preparing to load into ChromaDB...")

    documents = []
    metadatas = []
    ids = []
    
    existing_ids = collection.get(ids=[beat["filename"] for beat in beats])["ids"]

    # 4. Format the data for the vector database
    for beat in beats:
        if beat["filename"] in existing_ids:
            console.print(f"[yellow]Skipping {beat['filename']}, already in database.[/yellow]")
            continue
        # Create a descriptive text document for the embedding model to understand
        document = (
            f"A {beat['mood'].lower()} beat in the key of {beat['key']} "
            f"with a tempo of {beat['bpm']} BPM. It has an overall energy "
            f"of {beat['overall_energy']} and a bounciness of {beat['bounciness']}."
        )
        documents.append(document)
        
        # Store the raw JSON data as metadata
        metadatas.append(beat)
        
        # Use the filename as a unique ID
        ids.append(beat["filename"])

    # 5. Add the data to the collection
    if ids:
        console.print(f"Upserting {len(ids)} new beats into ChromaDB (this may take a moment to generate embeddings)...")
        collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
        console.print("[bold green]Successfully loaded new beats into ChromaDB![/bold green]")
    else:
        console.print("[bold yellow]No new beats to add.[/bold yellow]")

if __name__ == "__main__":
    main()