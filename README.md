# BeatRAG
BeatRAG 🎵🤖
BeatRAG is an intelligent pipeline designed to analyze, embed, and semantically search a local library of audio samples or beats. By combining traditional audio signal processing with a Retrieval-Augmented Generation (RAG) architecture, it allows producers and musicians to find the exact beat they are looking for using natural language queries.

How It Works
The project is broken down into a three-step pipeline:

1. Feature Extraction (extractor.py)
This script crawls a local samples/ directory for audio files (.wav, .mp3) and uses Librosa to extract detailed musical and acoustic features using parallel processing.

Extracted Data: Tempo (BPM), musical key (via Chroma CENS or STFT), spectral centroid, variance, flatness, overall energy (RMS), and bounciness (onset density).

Mood Classification: Automatically categorizes the beat's mood (e.g., "Aggressive, Intense, Dark", "Chill, Relaxed, Peaceful") based on a combination of musical key (major/minor), energy levels, and tempo.

Output: Saves the extracted metadata for all samples into sample_database.json.

2. Database Loading (load_to_chroma.py)
This script takes the extracted JSON data and loads it into a local ChromaDB vector database.

It generates a descriptive text string for each beat (e.g., "A aggressive, intense, dark beat in the key of G# Minor with a tempo of 144 BPM...").

It embeds these descriptions alongside the raw metadata attributes, ensuring the database understands both the semantic vibe and the exact numerical stats of the audio.

3. Semantic Query Agent (query_agent.py)
This is the interactive terminal interface for searching your library.

Powered by LangChain, HuggingFace Embeddings (all-MiniLM-L6-v2), and an OpenAI LLM (gpt-4o-mini).

It utilizes a SelfQueryRetriever that translates natural language prompts (e.g., "Find me a dark, high-energy trap beat around 140 BPM") into exact metadata filters and vector similarity searches to return the most relevant audio files.

Tech Stack
Audio Processing: Librosa, NumPy

Vector Database: ChromaDB

AI / RAG Pipeline: LangChain, OpenAI API, HuggingFace Sentence Transformers

CLI/UI: Rich (for beautiful terminal formatting)

Setup & Usage
1. Install Dependencies

Bash

pip install -r requirements.txt
2. Add Environment Variables
Create a .env file in the root directory and add your OpenAI API key:

Code snippet

OPENAI_API_KEY=your_api_key_here
3. Run the Pipeline

Place your audio files into a samples/ folder.

Extract Features: Run python extractor.py

Build Database: Run python load_to_chroma.py

Search Library: Run python query_agent.py and start typing your queries!
