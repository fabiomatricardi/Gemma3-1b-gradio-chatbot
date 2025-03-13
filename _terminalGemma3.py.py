# Chat with an intelligent assistant in your terminal  with google_gemma-3-1b-it-Q8_0.gguf
# model served in another terminal window with llama-server
from openai import OpenAI
import sys
from time import sleep
import warnings
warnings.filterwarnings(action='ignore')
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

STOPS = ['<eos>']
COUNTERLIMITS = 10  #an even number


# ASCII ART FROM https://asciiart.club/  or https://texteditor.com/ascii-art/
print("\033[94m")
t = """                                                                                                                                                           
▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
██░▄▄░██░▄▄▄██░▄▀▄░██░▄▀▄░█░▄▄▀███░▄▄░
██░█▀▀██░▄▄▄██░█░█░██░█░█░█░▀▀░█████▄▀
██░▀▀▄██░▀▀▀██░███░██░███░█░██░███░▀▀░
▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
             
"""
print(t)
modelname = 'Gemma3 1b Instruct'
NCTX = '32k'
# Point to the local server
client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed", organization=modelname)
print(f"✅ Ready to Chat with {modelname} Context length={NCTX} tokens...")
print("\033[0m")  #reset all

history = [
]
print("\033[92;1m")
counter = 1
while True:
    if counter > COUNTERLIMITS:
        history = [
        ]        
    userinput = ""
    print("\033[1;30m")  #dark grey
    print("Enter your text (end input with Ctrl+D on Unix or Ctrl+Z on Windows) - type quit! to exit the chatroom:")
    print("\033[91;1m")  #red
    lines = sys.stdin.readlines()
    for line in lines:
        userinput += line + "\n"
    if "quit!" in lines[0].lower():
        print("\033[0mBYE BYE!")
        break
    history.append({"role": "user", "content": userinput})
    print("\033[92;1m")

    completion = client.chat.completions.create(
        model=modelname, # this field is currently unused
        messages=history,
        temperature=0.3,
        frequency_penalty  = 1.6,
        max_tokens = 1000,
        stream=True,
        stop=STOPS
    )

    new_message = {"role": "assistant", "content": ""}
    
    for chunk in completion:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
            new_message["content"] += chunk.choices[0].delta.content
    history.append(new_message)  
    counter += 1  


##############MODEL CARD##########################################
"""
.\llama-server.exe -m google_gemma-3-1b-it-Q8_0.gguf -c 8192 -ngl 99
ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = Intel(R) UHD Graphics (Intel Corporation) | uma: 1 | fp16: 1 | warp size: 32 | shared memory: 32768 | matrix cores: none
build: 4879 (f08f4b31) with MSVC 19.43.34808.0 for x64
system info: n_threads = 10, n_threads_batch = 10, total_threads = 12

system_info: n_threads = 10 (n_threads_batch = 10) / 12 | CPU : SSE3 = 1 | SSSE3 = 1 | AVX = 1 | AVX2 = 1 | F16C = 1 | FMA = 1 | BMI2 = 1 | LLAMAFILE = 1 | OPENMP = 1 | AARCH64_REPACK = 1 |

main: HTTP server is listening, hostname: 127.0.0.1, port: 8080, http threads: 11
main: loading model
srv    load_model: loading model 'google_gemma-3-1b-it-Q8_0.gguf'
llama_model_load_from_file_impl: using device Vulkan0 (Intel(R) UHD Graphics) - 8028 MiB free
llama_model_loader: loaded meta data with 42 key-value pairs and 340 tensors from google_gemma-3-1b-it-Q8_0.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = gemma3
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Gemma 3 1b It
llama_model_loader: - kv   3:                           general.finetune str              = it
llama_model_loader: - kv   4:                           general.basename str              = gemma-3
llama_model_loader: - kv   5:                         general.size_label str              = 1B
llama_model_loader: - kv   6:                            general.license str              = gemma
llama_model_loader: - kv   7:                   general.base_model.count u32              = 1
llama_model_loader: - kv   8:                  general.base_model.0.name str              = Gemma 3 1b Pt
llama_model_loader: - kv   9:          general.base_model.0.organization str              = Google
llama_model_loader: - kv  10:              general.base_model.0.repo_url str              = https://huggingface.co/google/gemma-3...
llama_model_loader: - kv  11:                               general.tags arr[str,1]       = ["text-generation"]
llama_model_loader: - kv  12:                      gemma3.context_length u32              = 32768
llama_model_loader: - kv  13:                    gemma3.embedding_length u32              = 1152
llama_model_loader: - kv  14:                         gemma3.block_count u32              = 26
llama_model_loader: - kv  15:                 gemma3.feed_forward_length u32              = 6912
llama_model_loader: - kv  16:                gemma3.attention.head_count u32              = 4
llama_model_loader: - kv  17:    gemma3.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  18:                gemma3.attention.key_length u32              = 256
llama_model_loader: - kv  19:              gemma3.attention.value_length u32              = 256
llama_model_loader: - kv  20:                      gemma3.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv  21:            gemma3.attention.sliding_window u32              = 512
llama_model_loader: - kv  22:             gemma3.attention.head_count_kv u32              = 1
llama_model_loader: - kv  23:                       tokenizer.ggml.model str              = llama
llama_model_loader: - kv  24:                         tokenizer.ggml.pre str              = default
llama_model_loader: - kv  25:                      tokenizer.ggml.tokens arr[str,262144]  = ["<pad>", "<eos>", "<bos>", "<unk>", ...
llama_model_loader: - kv  26:                      tokenizer.ggml.scores arr[f32,262144]  = [-1000.000000, -1000.000000, -1000.00...
llama_model_loader: - kv  27:                  tokenizer.ggml.token_type arr[i32,262144]  = [3, 3, 3, 3, 3, 4, 3, 3, 3, 3, 3, 3, ...
llama_model_loader: - kv  28:                tokenizer.ggml.bos_token_id u32              = 2
llama_model_loader: - kv  29:                tokenizer.ggml.eos_token_id u32              = 1
llama_model_loader: - kv  30:            tokenizer.ggml.unknown_token_id u32              = 3
llama_model_loader: - kv  31:            tokenizer.ggml.padding_token_id u32              = 0
llama_model_loader: - kv  32:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  33:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  34:                    tokenizer.chat_template str              = {{ bos_token }}\n{%- if messages[0]['r...
llama_model_loader: - kv  35:            tokenizer.ggml.add_space_prefix bool             = false
llama_model_loader: - kv  36:               general.quantization_version u32              = 2
llama_model_loader: - kv  37:                          general.file_type u32              = 7
llama_model_loader: - kv  38:                      quantize.imatrix.file str              = /models_out/gemma-3-1b-it-GGUF/google...
llama_model_loader: - kv  39:                   quantize.imatrix.dataset str              = /training_dir/calibration_datav3.txt
llama_model_loader: - kv  40:             quantize.imatrix.entries_count i32              = 182
llama_model_loader: - kv  41:              quantize.imatrix.chunks_count i32              = 129
llama_model_loader: - type  f32:  157 tensors
llama_model_loader: - type q8_0:  183 tensors
print_info: file format = GGUF V3 (latest)
print_info: file type   = Q8_0
print_info: file size   = 1013.54 MiB (8.50 BPW)
load: special_eos_id is not in special_eog_ids - the tokenizer config may be incorrect
load: special tokens cache size = 6414
load: token to piece cache size = 1.9446 MB
print_info: arch             = gemma3
print_info: vocab_only       = 0
print_info: n_ctx_train      = 32768
print_info: n_embd           = 1152
print_info: n_layer          = 26
print_info: n_head           = 4
print_info: n_head_kv        = 1
print_info: n_rot            = 256
print_info: n_swa            = 512
print_info: n_embd_head_k    = 256
print_info: n_embd_head_v    = 256
print_info: n_gqa            = 4
print_info: n_embd_k_gqa     = 256
print_info: n_embd_v_gqa     = 256
print_info: f_norm_eps       = 0.0e+00
print_info: f_norm_rms_eps   = 1.0e-06
print_info: f_clamp_kqv      = 0.0e+00
print_info: f_max_alibi_bias = 0.0e+00
print_info: f_logit_scale    = 0.0e+00
print_info: f_attn_scale     = 6.2e-02
print_info: n_ff             = 6912
print_info: n_expert         = 0
print_info: n_expert_used    = 0
print_info: causal attn      = 1
print_info: pooling type     = 0
print_info: rope type        = 2
print_info: rope scaling     = linear
print_info: freq_base_train  = 1000000.0
print_info: freq_scale_train = 1
print_info: n_ctx_orig_yarn  = 32768
print_info: rope_finetuned   = unknown
print_info: ssm_d_conv       = 0
print_info: ssm_d_inner      = 0
print_info: ssm_d_state      = 0
print_info: ssm_dt_rank      = 0
print_info: ssm_dt_b_c_rms   = 0
print_info: model type       = 1B
print_info: model params     = 999.89 M
print_info: general.name     = Gemma 3 1b It
print_info: vocab type       = SPM
print_info: n_vocab          = 262144
print_info: n_merges         = 0
print_info: BOS token        = 2 '<bos>'
print_info: EOS token        = 1 '<eos>'
print_info: EOT token        = 106 '<end_of_turn>'
print_info: UNK token        = 3 '<unk>'
print_info: PAD token        = 0 '<pad>'
print_info: LF token         = 248 '<0x0A>'
print_info: EOG token        = 1 '<eos>'
print_info: EOG token        = 106 '<end_of_turn>'
print_info: max token length = 48
load_tensors: loading model tensors, this can take a while... (mmap = true)
load_tensors: offloading 26 repeating layers to GPU
load_tensors: offloading output layer to GPU
load_tensors: offloaded 27/27 layers to GPU
load_tensors:      Vulkan0 model buffer size =  1013.54 MiB
load_tensors:   CPU_Mapped model buffer size =   306.00 MiB
.......................................................
llama_init_from_model: n_seq_max     = 1
llama_init_from_model: n_ctx         = 8192
llama_init_from_model: n_ctx_per_seq = 8192
llama_init_from_model: n_batch       = 2048
llama_init_from_model: n_ubatch      = 512
llama_init_from_model: flash_attn    = 0
llama_init_from_model: freq_base     = 1000000.0
llama_init_from_model: freq_scale    = 1
llama_init_from_model: n_ctx_per_seq (8192) < n_ctx_train (32768) -- the full capacity of the model will not be utilized
llama_kv_cache_init: kv_size = 8192, offload = 1, type_k = 'f16', type_v = 'f16', n_layer = 26, can_shift = 1
llama_kv_cache_init:    Vulkan0 KV buffer size =   208.00 MiB
llama_init_from_model: KV self size  =  208.00 MiB, K (f16):  104.00 MiB, V (f16):  104.00 MiB
llama_init_from_model: Vulkan_Host  output buffer size =     1.00 MiB
llama_init_from_model:    Vulkan0 compute buffer size =   514.25 MiB
llama_init_from_model: Vulkan_Host compute buffer size =    34.26 MiB
llama_init_from_model: graph nodes  = 1047
llama_init_from_model: graph splits = 2
common_init_from_params: setting dry_penalty_last_n to ctx_size = 8192
common_init_from_params: warming up the model with an empty run - please wait ... (--no-warmup to disable)
srv          init: initializing slots, n_slots = 1
slot         init: id  0 | task -1 | new slot n_ctx_slot = 8192
main: model loaded
main: chat template, chat_template: {{ bos_token }}
{%- if messages[0]['role'] == 'system' -%}
    {%- if messages[0]['content'] is string -%}
        {%- set first_user_prefix = messages[0]['content'] + '

' -%}
    {%- else -%}
        {%- set first_user_prefix = messages[0]['content'][0]['text'] + '

' -%}
    {%- endif -%}
    {%- set loop_messages = messages[1:] -%}
{%- else -%}
    {%- set first_user_prefix = "" -%}
    {%- set loop_messages = messages -%}
{%- endif -%}
{%- for message in loop_messages -%}
    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}
        {{ raise_exception("Conversation roles must alternate user/assistant/user/assistant/...") }}
    {%- endif -%}
    {%- if (message['role'] == 'assistant') -%}
        {%- set role = "model" -%}
    {%- else -%}
        {%- set role = message['role'] -%}
    {%- endif -%}
    {{ '<start_of_turn>' + role + '
' + (first_user_prefix if loop.first else "") }}
    {%- if message['content'] is string -%}
        {{ message['content'] | trim }}
    {%- elif message['content'] is iterable -%}
        {%- for item in message['content'] -%}
            {%- if item['type'] == 'image' -%}
                {{ '<start_of_image>' }}
            {%- elif item['type'] == 'text' -%}
                {{ item['text'] | trim }}
            {%- endif -%}
        {%- endfor -%}
    {%- else -%}
        {{ raise_exception("Invalid content type") }}
    {%- endif -%}
    {{ '<end_of_turn>
' }}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{'<start_of_turn>model
'}}
{%- endif -%}
, example_format: '<start_of_turn>user
You are a helpful assistant

Hello<end_of_turn>
<start_of_turn>model
Hi there<end_of_turn>
<start_of_turn>user
How are you?<end_of_turn>
<start_of_turn>model
'
main: server is listening on http://127.0.0.1:8080 - starting the main loop
srv  update_slots: all slots are idle



"""    