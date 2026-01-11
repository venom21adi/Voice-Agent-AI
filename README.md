# Voice Agent Lab

> A local-first RAG-powered voice agent built with small language models (sLMs), Chatterbox TTS, and a Streamlit interface.

---

## Overview

**Voice Agent Lab** is a focused implementation of a **retrieval-augmented voice assistant** that combines:

- **sLM-based reasoning**
- **RAG for grounded responses**
- **Chatterbox TTS for natural speech output**
- **Streamlit** as the interactive UI layer

The goal is to build a **practical, locally runnable voice agent** that can answer questions, reason over retrieved context, and respond with expressive synthesized speech.

---

## Core Components

### 🧠 Small Language Model (SLM)
- Lightweight LLMs optimized for local inference
- Used for intent understanding, reasoning, and response generation
- Designed to run efficiently on consumer GPUs

### 📚 Retrieval-Augmented Generation (RAG)
- Vector-based document retrieval
- Context injection into the sLM prompt
- Enables domain-specific, grounded responses

### 🗣 Chatterbox TTS
- Neural text-to-speech engine
- Focus on natural prosody and clarity
- Integrated directly into the response pipeline

### 🖥 Streamlit Application
- Simple, fast UI for interacting with the voice agent
- Supports text input and spoken output
- Acts as the orchestration layer for RAG → sLM → TTS

---

## End-to-End Flow

```text
User Input
   ↓
Retriever (RAG)
   ↓
Small Language Model (sLM)
   ↓
Chatterbox TTS
   ↓
Spoken Response (Streamlit)
