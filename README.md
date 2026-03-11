# 🤖 AI Technical Interviewer

An interactive, multi-modal AI mock interview application built with **Streamlit**, **ChromaDB**, **Hugging Face**, and **Google Gemini**. 

This application simulates a highly realistic interview environment. It pulls role-specific questions from a local vector database (RAG pipeline), dynamically scales the difficulty as you progress, and allows you to answer using text or your voice. At the end of the session, it provides a comprehensive coaching report based on your technical accuracy and detected vocal tone.

## ✨ Key Features
* **Dual Interview Modes:** Choose between a traditional "Text Only" interview or a realistic "Audio Based" interview using your microphone.
* **Tone & Emotion Detection:** Analyzes your vocal responses using a Hugging Face audio classification model (`whisper-large-v3` architecture) to provide feedback on your confidence and emotional tone.
* **Real-Time Speech-to-Text:** Instantly transcribes your spoken answers so the AI can grade your technical accuracy.
* **Dynamic Difficulty Scaling:** Automatically progresses questions from Easy ➔ Medium ➔ Hard.
* **Retrieval-Augmented Generation (RAG):** Questions and ideal grading rubrics are retrieved locally from a ChromaDB vector database, ensuring accurate and grounded interview scenarios.
* **Comprehensive Coaching Report:** Generates a final Markdown report highlighting your strengths, areas for technical improvement, and communication advice.

---

## ⚠️ Important: Database Setup

To keep this repository lightweight, the heavy database files and raw JSON datasets are **not** included in this repo. You must download the pre-built database to run the app.

1. **Download the Database:** [https://drive.google.com/drive/folders/1E_9oFyIbh6GQE2ayY5Fwnud9-D5Ndm-O?usp=sharing]
2. Extract the downloaded folder.
3. Place the folder named `chroma_interview_db_v4` directly into the root directory of this project.

Your project structure should look exactly like this before running:
```
your-repo-name/
│
├── chroma_interview_db_v4/    <-- (Extracted from the download link)
├── app.py                     
├── .env                       <-- (You will create this)
└── README.md
```                 
🛠️ Prerequisites & Installation
Before you begin, ensure you have Python 3.9+ installed and a free Google Gemini API Key from Google AI Studio.

1. Clone the repository
Bash
```
git clone https://github.com/ovindumandith/chatbot_interview/
cd YOUR_REPOSITORY_NAME
```
3. Create a virtual environment (Highly Recommended)
Windows:
Bash
```
python -m venv venv
venv\Scripts\activate
```
macOS/Linux:
Bash
```
python3 -m venv venv
source venv/bin/activate
```
3. Install the dependencies
Install the required libraries for the frontend, database, AI evaluation, and audio processing:
Bash
```
pip install streamlit chromadb google-genai python-dotenv torch torchvision torchaudio transformers librosa SpeechRecognition soundfile numpy
(Note: If you run into an error regarding AutoModelForAudioClassification, run pip install --upgrade transformers)
```
5. Configure your API Key
Create a new file in the root directory named .env. Open it and add your Gemini API key:

Code snippet
GEMINI_API_KEY="your_api_key_here"
🚀 Running the Application
Once your database folder is in place and dependencies are installed, start the Streamlit server:
Bash
```
python -m streamlit run app.py
The application will automatically open in your default web browser.
```
