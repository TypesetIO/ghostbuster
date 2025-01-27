import openai
import json
import numpy as np
import tiktoken
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from pdb import set_trace


tokenizer = tiktoken.encoding_for_model("davinci")

llama_tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab_map = {}
vocab = llama_tokenizer.vocab
for token in vocab:
    idx = vocab[token]
    vocab_map[idx] = token


def write_logprobs(text, file, model):
    """
    Run text under model and write logprobs to file, separated by newline.
    """
    tokens = tokenizer.encode(text)
    doc = tokenizer.decode(tokens[:2047])

    response = openai.Completion.create(
        model=model,
        prompt="<|endoftext|>" + doc,
        max_tokens=0,
        echo=True,
        logprobs=1,
    )

    subwords = response["choices"][0]["logprobs"]["tokens"][1:]
    subprobs = response["choices"][0]["logprobs"]["token_logprobs"][1:]

    gpt2_map = {"\n": "Ċ", "\t": "ĉ", " ": "Ġ"}

    for i in range(len(subwords)):
        for k, v in gpt2_map.items():
            subwords[i] = subwords[i].replace(k, v)

    to_write = ""
    for _, (w, p) in enumerate(zip(subwords, subprobs)):
        to_write += f"{w} {-p}\n"

    with open(file, "w") as f:
        f.write(to_write)


def write_llama_logprobs(text, file, model):
    with torch.no_grad():
        encodings = llama_tokenizer(text, return_tensors="pt").to(device)
        logits = F.softmax(model(encodings["input_ids"]).logits, dim=2)

        tokens = encodings["input_ids"]
        indices = torch.tensor([[[i] for i in tokens[0]]])[:, 1:, :].to(device)
        subwords = [vocab_map[int(idx)]
                    for idx in encodings["input_ids"][0][1:]]
        subprobs = (
            torch.gather(logits[:, :-1, :], dim=2, index=indices)
            .flatten()
            .cpu()
            .detach()
            .numpy()
        )

    to_write = ""
    for _, (w, p) in enumerate(zip(subwords, subprobs)):
        to_write += f"{w} {-np.log(p)}\n"

    with open(file, "w") as f:
        f.write(to_write)

import nltk
from copy import deepcopy
def write_llama_logprobs_sentence(text, file, model, human_counter, gpt_counter):
    with torch.no_grad():
        encodings = llama_tokenizer(text, return_tensors="pt").to(device)
        logits = F.softmax(model(encodings["input_ids"]).logits, dim=2)

        tokens = encodings["input_ids"]
        indices = torch.tensor([[[i] for i in tokens[0]]])[:, 1:, :].to(device)
        tokens = tokens[0][1:]
        sentence_level_indices = []
        start_idx = 0
        sentence_meta = []
        sentence_encodings = []
        temp_encodings = []
        for idx_, token in enumerate(tokens.tolist()):
            temp_encodings.append(token)
            if token == 13:
                sentence_encodings.append({'encodings':temp_encodings, 'indices':indices[0][start_idx:start_idx+len(temp_encodings)]})
                start_idx += len(temp_encodings)
                temp_encodings = []
        if temp_encodings:
            sentence_encodings.append({'encodings':temp_encodings, 'indices':indices[0][start_idx:start_idx+len(temp_encodings)]})
        for s_idx, sent in enumerate(sentence_encodings):
            encodings = sent['encodings']
            sentence_text = llama_tokenizer.decode(encodings)
            subwords = [vocab_map[int(idx)] for idx in encodings]
            subprobs = torch.gather(logits, dim=2, index=sent['indices'].reshape(1, -1, 1)).flatten().cpu().detach().numpy()
            to_write = ""
            assert len(subwords) == len(subprobs) == len(encodings)
            for _, (w, p, e) in enumerate(zip(subwords, subprobs, encodings)):
                to_write += f"{w} {-np.log(p)} {e}\n"
            with open(f'{"/".join(file.split("/")[:-1])}/{human_counter if "human" in file else gpt_counter}-llama-13b.txt', "w") as f:
                f.write(to_write)
            if 'human' in file:
                with open(f'data/essay/sentence_text/human/{human_counter}.txt', "w") as f:
                    f.write(sentence_text)
                human_counter += 1
            else:
                with open(f'data/essay/sentence_text/gpt/{gpt_counter}.txt', "w") as f:
                    f.write(sentence_text)
                gpt_counter += 1
        # tokens = tokens[0][1:]
        # sentence_level_indices = []
        # start_idx = 0
        # sentence_meta = []
        # for s_idx, sentence in enumerate(sentences):
        #     sentence_text = deepcopy(sentence)
        #     if s_idx > 0:
        #         sentence_text = " " + sentence_text
        #     sentence_encodings = llama_tokenizer(sentence_text, return_tensors="pt").to(device)
        #     sentence_tokens = sentence_encodings["input_ids"][0][1:]
        #     sentence_meta_item = dict()
        #     sentence_meta_item["text"] = sentence_text
        #     sentence_meta_item["tokens"] = sentence_tokens
        #     sentence_meta_item["indices"] = indices[0][start_idx:start_idx + len(sentence_tokens)]
        #     sentence_meta_item["encodings"] = encodings
        #     sentence_meta.append(sentence_meta_item)
        #     start_idx += len(sentence_tokens)
        
        # for s_idx, sent in enumerate(sentence_meta):
        #     if s_idx >0:
        #         sentence_indices = sent["indices"].reshape(1, -1, 1)#[:,1:,:]
        #         sentence_text = sent["text"][:]
        #     else:
        #         sentence_indices = sent["indices"].reshape(1, -1, 1)
        #         sentence_text = sent["text"]
        #     if len(sentence_text.split()) <6:
        #         continue
        #     subwords = [vocab_map[int(idx)] for idx in sent["tokens"]]
        #     subprobs = torch.gather(logits, dim=2, index=sentence_indices).flatten().cpu().detach().numpy()
        #     set_trace()
        #     to_write = ""
        #     for _, (w, p) in enumerate(zip(subwords, subprobs)):
        #         to_write += f"{w} {-np.log(p)}\n"
        #     with open(f'{"/".join(file.split("/")[:-1])}/{counter}-llama-13b.txt', "w") as f:
        #         f.write(to_write)
        #     if 'human' in file:
        #         with open(f'data/essay/sentence_text/human/{counter}.txt', "w") as f:
        #             f.write(sentence_text)
        #     else:
        #         with open(f'data/essay/sentence_text/gpt/{counter}.txt', "w") as f:
        #             f.write(sentence_text)
        #     counter += 1
        # subwords = [vocab_map[int(idx)]
        #             for idx in encodings["input_ids"][0][1:]]
        # subprobs = (
        #     torch.gather(logits[:, :-1, :], dim=2, index=indices)
        #     .flatten()
        #     .cpu()
        #     .detach()
        #     .numpy()
        # )
    # set_trace()

    # to_write = ""
    # for _, (w, p) in enumerate(zip(subwords, subprobs)):
    #     to_write += f"{w} {-np.log(p)}\n"

    # with open(file, "w") as f:
    #     f.write(to_write)
    return human_counter, gpt_counter