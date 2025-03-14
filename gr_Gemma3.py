import gradio as gr
from openai import OpenAI
# when using llamacpp-server, you need to check if the stream chunk is present
# usually the first and the last chunk are empty and will throw an error
# https://www.gradio.app/guides/creating-a-custom-chatbot-with-blocks
example = """
#### Example for Image Generation help
"""
mycode ="""
```
I want to create an image with Flux but I need assistance for a good prompt. 
The image should be about '''[userinput]'''. Comic art style.
```
"""
note = """#### 🔹 Gemma 3 1B Instruct
> [Gemma 3](https://ai.google.dev/gemma/docs/core), a collection of lightweight, state-of-the-art open models built from the same research and technology that powers our Gemini 2.0 models. 
<br>

[Gemma 3](https://developers.googleblog.com/en/introducing-gemma3/) comes in a range of sizes (1B, 4B, 12B and 27B)
These are the Google most advanced, portable and responsibly developed open models yet. 
<br><br>

Starting settings: `Temperature=0.45` `Max_Length=1100`
"""
# RUN the llamaCPP server binaries as a subrocess
import subprocess
#start cmd.exe /k "llama-server.exe -m google_gemma-3-1b-it-Q8_0.gguf -c 8192 -ngl 999"
modelname = 'google_gemma-3-1b-it'
NCTX = 8192
print(f"Starting llamacpp server for {modelname} Context length={NCTX} tokens...")
mc = ['start',
    'cmd.exe',
    '/k',
    'llama-server.exe',
    '-m',
    'google_gemma-3-1b-it-Q8_0.gguf',
    '-c',         
    '8192',
    '-ngl',
    '999'   
]
res = subprocess.call(mc,shell=True)
# STARTING THE INTERFACE
with gr.Blocks(theme=gr.themes.Ocean()) as demo: #gr.themes.Ocean() Citrus() #https://www.gradio.app/guides/theming-guide
    gr.Markdown("# Chat with Gemma 3 1b Instruct 🔷 running Locally with [llama.cpp](https://github.com/ggml-org/llama.cpp)")
    with gr.Row():
        with gr.Column(scale=1):
            maxlen = gr.Slider(minimum=250, maximum=4096, value=1100, step=1, label="Max new tokens")
            temperature = gr.Slider(minimum=0.1, maximum=4.0, value=0.45, step=0.1, label="Temperature")          
            gr.Markdown(note)
            with gr.Accordion("See suggestions",open=False):
                gr.Markdown(example)
                gr.Code(mycode,language='markdown',wrap_lines=True)
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(type="messages",show_copy_button = True,
                    avatar_images=['https://i.ibb.co/m588VrQ6/fabio-Matricardi.png','https://clipartcraft.com/images/transparent-background-google-logo-brand-2.png'],
                    height=550, layout='bubble')
            msg = gr.Textbox(lines=3,placeholder='Shift+Enter to send your message')
            # Button the clear the conversation history
            clear = gr.ClearButton([msg, chatbot],variant='primary')
    # Handle the User Messages
    def user(user_message, history: list):
        return "", history + [{"role": "user", "content": user_message}]    
    # HANDLE the inference with the API server
    def respond(chat_history,t,m):
        STOPS = ['<eos>']
        client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed", organization='Gemma3')
        stream = client.chat.completions.create(     
            messages=chat_history,
            model='Gemma 3 1B Instruct',
            max_tokens=m,
            stream=True,
            temperature=t,
            stop=STOPS)
        chat_history.append({"role": "assistant", "content": ""})
        for chunk in stream:
            # this is used with llama-server
            if chunk.choices[0].delta.content:
                chat_history[-1]['content'] += chunk.choices[0].delta.content
            yield chat_history
    msg.submit(user, [msg, chatbot], [msg, chatbot]).then(respond, [chatbot,temperature,maxlen], [chatbot])
# LAUNCH THE GRADIO APP with Opening automatically the default browser
demo.launch(inbrowser=True)
