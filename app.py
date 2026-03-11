import streamlit as st
import chromadb
import json
import os
import torch
import librosa
import numpy as np
import speech_recognition as sr
import random
from dotenv import load_dotenv
from google import genai
from google.genai import types
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

# --- 1. SETUP & CONFIGURATION ---
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("Missing GEMINI_API_KEY. Please check your .env file.")
    st.stop()

@st.cache_resource
def load_clients():
    gemini_client = genai.Client(api_key=api_key)
    chroma_client = chromadb.PersistentClient(path="./chroma_interview_db_v4")
    collection = chroma_client.get_collection(name="interview_bank")
    return gemini_client, collection

@st.cache_resource
def load_emotion_model():
    model_id = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"
    model = AutoModelForAudioClassification.from_pretrained(model_id)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, do_normalize=True)
    id2label = model.config.id2label
    return model, feature_extractor, id2label

client, collection = load_clients()
emotion_model, feature_extractor, id2label = load_emotion_model()

evaluator_system_prompt = """You are a strict but fair senior technical interviewer. 
Compare the candidate's answer to the ideal answer. 
You MUST return your response in the following JSON format ONLY:
{
  "score": <integer from 1 to 10>,
  "feedback": "<concise, constructive feedback>"
}"""


# --- 2. SESSION STATE MANAGEMENT ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [] 
if "asked_ids" not in st.session_state:
    st.session_state.asked_ids = []
if "current_state" not in st.session_state:
    st.session_state.current_state = {"q_id": None, "q": None, "ideal": None, "difficulty": None}
if "total_score" not in st.session_state:
    st.session_state.total_score = 0
if "interview_active" not in st.session_state:
    st.session_state.interview_active = False
if "current_q_num" not in st.session_state:
    st.session_state.current_q_num = 0
if "max_questions" not in st.session_state:
    st.session_state.max_questions = 5
if "interview_mode" not in st.session_state:
    st.session_state.interview_mode = "Text Only"
if "interview_data" not in st.session_state:
    st.session_state.interview_data = []


# --- 3. BACKEND LOGIC ---
def get_target_difficulty(role, current_q_num, max_qs):
    if role == "Tester": return None 
    progress_ratio = current_q_num / max_qs
    if progress_ratio <= 0.34: return "Easy"
    elif progress_ratio <= 0.67: return "Medium"
    else: return "Hard"

def get_unasked_question(role, asked_ids, target_difficulty=None):
    def fetch_with_filter(where_filter):
        results = collection.get(where=where_filter)
        num_results = len(results['ids'])
        
        if num_results == 0:
            return None, None, None
            
        # Shuffle the indices so the first question is always random!
        indices = list(range(num_results))
        random.shuffle(indices)
        
        for i in indices:
            q_id = results['ids'][i]
            if q_id not in asked_ids:
                text_block = results['documents'][i]
                parts = text_block.split("Ideal Answer:")
                if len(parts) == 2:
                    question = parts[0].replace("Interview Question:", "").strip()
                    raw_ideal = parts[1]
                    ideal_answer = raw_ideal.split("Category:")[0].strip() if "Category:" in raw_ideal else raw_ideal.strip()
                    return q_id, question, ideal_answer
        return None, None, None

    # Try strict difficulty filter first
    if target_difficulty:
        strict_filter = {"$and": [{"role": {"$eq": role}}, {"difficulty": {"$eq": target_difficulty}}]}
        q_id, q, ideal = fetch_with_filter(strict_filter)
        if q: return q_id, q, ideal, target_difficulty

    # Fallback to any question for that role
    q_id, q, ideal = fetch_with_filter({"role": {"$eq": role}})
    return q_id, q, ideal, "Random/Fallback"

def evaluate_answer(question, ideal_answer, user_answer):
    try:
        user_prompt = f"Question: {question}\nIdeal Answer: {ideal_answer}\nCandidate Answer: {user_answer}"
        config = types.GenerateContentConfig(
            system_instruction=evaluator_system_prompt,
            response_mime_type="application/json"
        )
        response = client.models.generate_content(
            model='gemini-3-flash-preview',
            contents=user_prompt,
            config=config
        )
        return json.loads(response.text)
    except Exception as e:
        if "429" in str(e):
            return {"score": 0, "feedback": "⚠️ **System busy!** We are receiving too many requests. Please wait 60 seconds and try again."}
        return {"score": 0, "feedback": f"Error during evaluation: {str(e)}"}

def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        return ""
    except Exception as e:
        return f"[Transcription Error: {str(e)}]"

def predict_emotion(audio_path, max_duration=30.0):
    audio_array, sampling_rate = librosa.load(audio_path, sr=None)
    max_length = int(feature_extractor.sampling_rate * max_duration)
    
    if len(audio_array) > max_length:
        audio_array = audio_array[:max_length]
    else:
        audio_array = np.pad(audio_array, (0, max_length - len(audio_array)))

    inputs = feature_extractor(
        audio_array, sampling_rate=feature_extractor.sampling_rate,
        max_length=max_length, truncation=True, return_tensors="pt",
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_to_use = emotion_model.to(device)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model_to_use(**inputs)

    predicted_id = torch.argmax(outputs.logits, dim=-1).item()
    return id2label[predicted_id]

def generate_final_feedback(interview_data, interview_mode):
    prompt = f"You are an expert Career Coach and Senior Executive Recruiter. Review the candidate's performance from a {interview_mode} interview.\n\n"
    
    for i, data in enumerate(interview_data):
        prompt += f"Q{i+1}: {data['question']}\n"
        prompt += f"Candidate Answer: {data['answer']}\n"
        prompt += f"Score: {data['score']}/10\n"
        if data.get('emotion'):
            prompt += f"Detected Tone: {data['emotion']}\n"
        prompt += "\n"
    
    prompt += """Please provide a beautifully formatted Markdown report containing:
    1. 🌟 Overall Performance Summary
    2. 💪 Key Strengths
    3. 📈 Areas for Technical Improvement (reference specific questions)
    4. 🗣️ Communication & Tone Feedback (If audio was used, provide specific advice on managing emotions, projecting confidence, and controlling tone based on the detected emotions. If text only, skip the tone advice).
    Keep your tone encouraging, professional, and highly constructive."""

    try:
        response = client.models.generate_content(
            model='gemini-3-flash-preview',
            contents=prompt
        )
        return response.text
    except Exception as e:
        if "429" in str(e):
            return "⚠️ **Final Report Delayed:** We are experiencing high traffic (API limit reached). Please wait a moment and try submitting your last answer again."
        return f"Error generating final feedback: {str(e)}"


# --- 4. STREAMLIT UI LAYOUT ---
st.set_page_config(page_title="AI Interviewer", page_icon="🤖", layout="wide")
st.title("🤖 AI Technical Interviewer")

# --- SIDEBAR CONTROLS & METRICS ---
with st.sidebar:
    st.header("⚙️ Interview Settings")
    
    selected_mode = st.radio(
        "Interview Mode", 
        ["Text Only", "Audio Based"], 
        disabled=st.session_state.interview_active,
        horizontal=True
    )
    
    selected_role = st.selectbox("Select Job Role", ["Software Engineering", "Human Resources", "Tester"], disabled=st.session_state.interview_active)
    selected_max_qs = st.slider("Number of Questions", min_value=3, max_value=10, value=5, disabled=st.session_state.interview_active)
    st.divider()
    
    if st.session_state.interview_active:
        st.header("📊 Live Metrics")
        col1, col2 = st.columns(2)
        col1.metric("Current Score", f"{st.session_state.total_score}")
        col2.metric("Question", f"{st.session_state.current_q_num} / {st.session_state.max_questions}")
        st.progress(st.session_state.current_q_num / st.session_state.max_questions, text="Interview Progress")
        st.divider()
        if st.button("End Interview Early", type="secondary"):
            st.session_state.interview_active = False
            st.session_state.chat_history.append({"role": "assistant", "content": f"🛑 Interview ended early. Final Score: {st.session_state.total_score} / {st.session_state.current_q_num * 10}"})
            st.rerun()
            
    elif not st.session_state.interview_active:
        if st.button("Start Interview", type="primary", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.asked_ids = []
            st.session_state.total_score = 0
            st.session_state.current_q_num = 1
            st.session_state.max_questions = selected_max_qs
            st.session_state.interview_mode = selected_mode 
            st.session_state.interview_data = [] 
            st.session_state.interview_active = True
            
            target_diff = get_target_difficulty(selected_role, 1, selected_max_qs)
            q_id, question, ideal, actual_diff = get_unasked_question(selected_role, [], target_diff)
            
            if not question:
                st.error(f"Not enough questions found for '{selected_role}'.")
                st.session_state.interview_active = False
            else:
                st.session_state.current_state = {"q_id": q_id, "q": question, "ideal": ideal, "difficulty": actual_diff}
                st.session_state.asked_ids = [q_id]
                welcome_msg = f"Welcome! Let's start your **{selected_mode}** interview for **{selected_role}**.\n\n**Question 1 [{actual_diff}]:** {question}"
                st.session_state.chat_history.append({"role": "assistant", "content": welcome_msg})
                st.rerun()


# --- MAIN CHAT INTERFACE ---
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if st.session_state.interview_active and st.session_state.current_state["q"]:
    
    final_answer_text = None
    detected_emotion = None
    user_input_triggered = False

    # --- PATH 1: TEXT ONLY ---
    if st.session_state.interview_mode == "Text Only":
        user_text = st.chat_input("Type your answer here...")
        if user_text:
            final_answer_text = user_text
            st.session_state.chat_history.append({"role": "user", "content": user_text})
            with st.chat_message("user"):
                st.markdown(user_text)
            user_input_triggered = True

    # --- PATH 2: AUDIO BASED ---
    elif st.session_state.interview_mode == "Audio Based":
        dynamic_key = f"audio_q_{st.session_state.current_q_num}"
        user_audio = st.audio_input("Record your answer 🎤", key=dynamic_key)
        
        if user_audio:
            with st.chat_message("user"):
                with st.spinner("Processing your voice..."):
                    temp_audio_path = "temp_answer.wav"
                    with open(temp_audio_path, "wb") as f:
                        f.write(user_audio.getbuffer())
                    
                    final_answer_text = transcribe_audio(temp_audio_path)
                    detected_emotion = predict_emotion(temp_audio_path)
                    
                    if final_answer_text:
                        display_text = f"🎙️ **Transcribed:** {final_answer_text}\n\n🎭 **Detected Tone:** {detected_emotion}"
                        st.markdown(display_text)
                        st.session_state.chat_history.append({"role": "user", "content": display_text})
                        user_input_triggered = True
                    else:
                        st.error("I couldn't hear you clearly. Please try recording again.")

    # --- SHARED EVALUATION LOGIC ---
    if user_input_triggered and final_answer_text:
        with st.chat_message("assistant"):
            with st.spinner("Evaluating your answer..."):
                question = st.session_state.current_state["q"]
                ideal = st.session_state.current_state["ideal"]
                
                # Evaluate individual answer
                eval_result = evaluate_answer(question, ideal, final_answer_text)
                score = eval_result.get("score", 0)
                feedback = eval_result.get("feedback", "No feedback.")
                st.session_state.total_score += score
                
                # Save to history for the final report
                st.session_state.interview_data.append({
                    "question": question,
                    "answer": final_answer_text,
                    "score": score,
                    "feedback": feedback,
                    "emotion": detected_emotion
                })
                
                ai_response = f"**Feedback (Score: {score}/10):**\n{feedback}\n\n"
                
                # Check for next question
                if st.session_state.current_q_num < st.session_state.max_questions:
                    st.session_state.current_q_num += 1
                    target_diff = get_target_difficulty(selected_role, st.session_state.current_q_num, st.session_state.max_questions)
                    next_q_id, next_q, next_ideal, actual_diff = get_unasked_question(selected_role, st.session_state.asked_ids, target_diff)
                    
                    if next_q:
                        st.session_state.asked_ids.append(next_q_id)
                        st.session_state.current_state = {"q_id": next_q_id, "q": next_q, "ideal": next_ideal, "difficulty": actual_diff}
                        ai_response += f"**Question {st.session_state.current_q_num} [{actual_diff}]:** {next_q}"
                    else:
                        ai_response += "⚠️ We ran out of questions in the database!\n\n"
                        st.session_state.interview_active = False
                else:
                    # End of Interview - trigger final report
                    st.session_state.interview_active = False
                    ai_response += f"🎉 **Interview Complete!** 🎉\nYour final score is {st.session_state.total_score} out of {st.session_state.max_questions * 10}.\n\n"
                    ai_response += "---\n### 📊 Generating Your Final Coaching Report...\n\n"
                    
                    with st.spinner("Analyzing your overall performance and tone..."):
                        final_report = generate_final_feedback(st.session_state.interview_data, st.session_state.interview_mode)
                        ai_response += final_report
                        st.session_state.current_state = {"q_id": None, "q": None, "ideal": None, "difficulty": None}
                    
            st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
            st.rerun()