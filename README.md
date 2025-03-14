# Gemma3-1b-gradio-chatbot
A gradio Chatbot with Gemma3-1b-it GGUF

A full locally running chat-bot powered by llama.cpp server and Gemma3-1b-it-GGUF


<img src='https://github.com/fabiomatricardi/Gemma3-1b-gradio-chatbot/raw/main/Gemma3-interface.gif' width=1000>


### Instructions
- Clone the repo
- create a `venv` (optional) and install the dependencies
```
# create a virtual environment
python -m venv venv
# activate the virtual environment
venv\Scripts\activate
# install the dependencies
pip install openai tiktoken gradio
```
- download the GGUF file in the same directory from the Bartowski repository [google_gemma-3-1b-it-Q8_0.gguf](https://huggingface.co/bartowski/google_gemma-3-1b-it-GGUF/resolve/main/google_gemma-3-1b-it-Q8_0.gguf?download=true)

### How to run
from the terminal run
```
python gr_Gemma3.py
```

