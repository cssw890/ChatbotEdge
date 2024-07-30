# ChatbotEdge

## Introduction
This project shows a chatbot implementation on an embedded Linux Device

## Prerequisites
Ubuntu 20.04
Horizon Sunrise X3 Pi

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/cssw890/ChatbotEdge.git
    ```
2. Install the required dependencies:
    ```bash
    # Update package list 
    sudo apt-get update
    
    # Install Python packages
    pip install sounddevice
    pip install faster-whisper
    pip install languagemodels
    pip install sherpa-onnx
    pip install numpy
    ```
3. Download the TTS model on the same project folder:
    ```bash
    wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-amy-low.tar.bz2
    tar xvf vits-piper-en_US-amy-low.tar.bz2
    rm vits-piper-en_US-amy-low.tar.bz2
    ```
