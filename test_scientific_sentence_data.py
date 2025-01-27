import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import dill as pickle
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.featurize import get_logprobs, normalize
from utils.n_gram import TrigramBackoff
from nltk.util import ngrams
from nltk.corpus import brown
from utils.symbolic import vec_functions, scalar_functions
from collections import defaultdict
from utils.featurize import convert_file_to_logprob_file, get_logprobs
import tqdm
import torch
import torch.nn.functional as F
from pdb import set_trace
import json
from glob import glob
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import stanza
nlp = stanza.Pipeline('en')


# Initialize tokenizer and trigram model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct").to(device)
# model.eval()
# Initialize with an empty list or pre-trained data


sentences = brown.sents()
tokenized_corpus = []
for sentence in tqdm.tqdm(sentences):
    tokens = tokenizer(" ".join(sentence))["input_ids"]
    tokenized_corpus += tokens

trigram = TrigramBackoff(tokenized_corpus)

# Define vectors and functions
vectors = ["llama-logprobs", "unigram-logprobs", "trigram-logprobs"]


vec_combinations = defaultdict(list)
for vec1 in range(len(vectors)):
    for vec2 in range(vec1):
        for func in vec_functions:
            if func != "v-div":
                vec_combinations[vectors[vec1]].append(
                    f"{func} {vectors[vec2]}")

for vec1 in vectors:
    for vec2 in vectors:
        if vec1 != vec2:
            vec_combinations[vec1].append(f"v-div {vec2}")


def get_words(exp):
    """
    Splits up expression into words, to be individually processed
    """
    return exp.split(" ")


def backtrack_functions(
    max_depth=2,
):
    """
    Backtrack all possible features.
    """

    def helper(prev, depth):
        if depth >= max_depth:
            return []

        all_funcs = []
        prev_word = get_words(prev)[-1]

        for func in scalar_functions:
            all_funcs.append(f"{prev} {func}")

        for comb in vec_combinations[prev_word]:
            all_funcs += helper(f"{prev} {comb}", depth + 1)

        return all_funcs

    ret = []
    for vec in vectors:
        ret += helper(vec, 0)
    return ret


def score_ngram(doc, model, tokenizer, n=3):
    """
    Returns vector of ngram probabilities given document, model and tokenizer
    """
    scores = []
    tokens = (
        tokenizer(doc.strip())[1:] if n == 1 else (
            n - 2) * [2] + tokenizer(doc.strip())
    )

    for i in ngrams(tokens, n):
        scores.append(model.n_gram_probability(i))

    return np.array(scores)


def get_all_logprobs(
    generate_dataset,
    preprocess=lambda x: x.strip(),
    verbose=True,
    trigram=None,
    tokenizer=None,
    num_tokens=2047,
):
    llama_logprobs = {}
    trigram_logprobs, unigram_logprobs = {}, {}

    if verbose:
        print("Loading logprobs into memory")

    file_names = generate_dataset(lambda file: file, verbose=False)
    to_iter = tqdm.tqdm(file_names) if verbose else file_names

    for file in to_iter:
        if "logprobs" in file:
            continue
        with open(file, "r") as f:
            doc = preprocess(f.read())
        llama_logprobs[file] = get_logprobs(
            convert_file_to_logprob_file(file, "llama-13b")
        )[:num_tokens]
        trigram_logprobs[file] = score_ngram(
            doc, trigram, tokenizer, n=3)[:num_tokens]
        unigram_logprobs[file] = score_ngram(doc, trigram.base, tokenizer, n=1)[
            :num_tokens
        ]

    return llama_logprobs, trigram_logprobs, unigram_logprobs


all_funcs = backtrack_functions(max_depth=3)
np.random.seed(0)

# Define best features
best_features = [
    "trigram-logprobs v-add unigram-logprobs v-sub llama-logprobs s-avg",
    "trigram-logprobs v-add unigram-logprobs v-> llama-logprobs s-avg-top-25",
    "unigram-logprobs v-mul llama-logprobs s-avg",
    "trigram-logprobs v-sub llama-logprobs s-avg-top-25",
    "trigram-logprobs v-< unigram-logprobs v-div trigram-logprobs s-avg-top-25",
    "trigram-logprobs v-sub unigram-logprobs v-> llama-logprobs s-l2",
    "trigram-logprobs v-mul unigram-logprobs v-div trigram-logprobs s-avg-top-25",
    "trigram-logprobs v-mul llama-logprobs v-div trigram-logprobs s-var",
    "trigram-logprobs v-div unigram-logprobs s-avg-top-25",
    "unigram-logprobs v-add llama-logprobs v-div unigram-logprobs s-avg-top-25",
    "trigram-logprobs v-add llama-logprobs v-div trigram-logprobs s-l2",
    "trigram-logprobs v-sub unigram-logprobs v-add llama-logprobs s-var",
    "unigram-logprobs s-min",
    "llama-logprobs v-div unigram-logprobs v-sub llama-logprobs s-var",
    "trigram-logprobs v-> unigram-logprobs v-< llama-logprobs s-var",
    "trigram-logprobs v-< unigram-logprobs v-> llama-logprobs s-avg-top-25",
    "trigram-logprobs v-add llama-logprobs s-var",
]

# Function to calculate n-gram scores


def score_ngram(doc, model, tokenizer, n=3):
    tokens = tokenizer(doc.strip())[1:] if n == 1 else (
        n - 2) * [2] + tokenizer(doc.strip())
    scores = [model.n_gram_probability(i) for i in ngrams(tokens, n)]
    return np.array(scores)


# Function to predict text
vocab_map = {}
vocab = tokenizer.vocab
for token in vocab:
    idx = vocab[token]
    vocab_map[idx] = token


def get_logprobs_direct(text):
    """
    Returns the calculated feature from the logprobs of a given text input
    based on the provided expression.
    """
    with torch.no_grad():
        encodings = tokenizer(text, return_tensors="pt").to(device)
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
        subprobs = [-np.log(line) for line in subprobs]
        logprobs = np.array([np.exp(-line) for line in subprobs])

        # Clean up CUDA memory
        del encodings
        del logits
        del tokens
        del indices
        torch.cuda.empty_cache()

    return logprobs


def get_logprobs_for_text(
    text,
    preprocess=lambda x: x.strip(),
    verbose=True,
    trigram=None,
    tokenizer=None,
    num_tokens=2047,
):
    llama_logprobs = {}
    trigram_logprobs, unigram_logprobs = {}, {}

    if verbose:
        print("Calculating logprobs for the provided text")

    # Preprocess the text
    doc = preprocess(text)

    # Calculate log probabilities
    llama_logprobs = get_logprobs_direct(doc)[:num_tokens]
    trigram_logprobs = score_ngram(doc, trigram, tokenizer, n=3)[:num_tokens]
    unigram_logprobs = score_ngram(
        doc, trigram.base, tokenizer, n=1)[:num_tokens]

    return llama_logprobs, trigram_logprobs, unigram_logprobs


def get_exp_featurize_for_text(best_features, vector_map):
    def calc_features(text, exp):
        exp_tokens = get_words(exp)
        curr = vector_map[exp_tokens[0]](text)

        for i in range(1, len(exp_tokens)):
            if exp_tokens[i] in vec_functions:
                next_vec = vector_map[exp_tokens[i + 1]](text)
                curr = vec_functions[exp_tokens[i]](curr, next_vec)
            elif exp_tokens[i] in scalar_functions:
                return scalar_functions[exp_tokens[i]](curr)

    def exp_featurize(text):
        return np.array([calc_features(text, exp) for exp in best_features])

    return exp_featurize


def predict_text(text):
    # Tokenize and score n-grams
    llama_logprobs, trigram_logprobs, unigram_logprobs = get_logprobs_for_text(
        text,
        verbose=True,
        tokenizer=lambda x: tokenizer(x)["input_ids"],
        trigram=trigram,
        num_tokens=2047,
    )

    # Map vectors to their respective logprobs
    vector_map = {
        "llama-logprobs": lambda file: llama_logprobs,
        "trigram-logprobs": lambda file: trigram_logprobs,
        "unigram-logprobs": lambda file: unigram_logprobs,
    }
    model_directory = "sentence_level_model/"
    model_name = "llama_model.pkl"
    with open(model_directory+model_name, "rb") as f:
        logistic_model = pickle.load(f)
    mu = pickle.load(open(model_directory+"mu", "rb"))
    sigma = pickle.load(open(model_directory+"sigma", "rb"))

    data = get_exp_featurize_for_text(best_features, vector_map)(text)
    data = normalize(data.reshape(1, -1), mu, sigma)

    # Predict using the logistic regression model
    prediction = logistic_model.predict(data)
    probability = logistic_model.predict_proba(data)[0][0]

    print(f"Probability: {probability:.4f}")
    result = "GPT" if prediction == 1 else "Human"
    print(f"Prediction: {result}")
    return prediction


# Example usage
# text = """
# Humanity is a tapestry woven from countless threads of culture, history, experience, and aspiration. At its core, it represents the collective existence of human beings, encompassing the shared traits and unique diversities that define our species. Within the broad spectrum of human identity, there exists a complex interplay of emotions, intellect, and spirit that propels us toward progress, yet often pulls us into conflict.

# One of the most commendable aspects of humanity is our capacity for empathy. This intrinsic ability to understand and share the feelings of others forms the bedrock of our social interactions. It has fueled efforts to build communities, foster relationships, and, on a larger scale, strive for global cooperation. Through empathy, we forge connections that transcend geographical, cultural, and ideological boundaries. This is evident in humanitarian efforts and the universal reaction to crises, where individuals unite to offer support and aid to those in need, reminding us of our inherently compassionate nature.

# Yet, the narrative of humanity is not without its darker chapters. Our history is marred by conflict, prejudice, and an insatiable appetite for power and resources. These elements have often led to the exploitation of others and the environment, posing significant challenges to our collective well-being. The residue of these actions lingers in modern issues such as inequality, systemic injustice, and environmental degradation. Nevertheless, these challenges have also driven us to question, to learn, and to evolve in our understanding of justice and equality.

# Innovation and creativity stand as testaments to the human spirit's resilience and ingenuity. From the creation of the wheel to the mapping of the human genome, humanity's pursuit of knowledge and improvement has been relentless. This drive has propelled technological advancements that revolutionize our lives and expand our capabilities. Moreover, the arts—encompassing literature, music, and visual media—reflect and critique our human experience, offering insights into our condition and exploring existential themes that resonate across time and space.

# Throughout history, humanity has been on a quest for meaning, seeking to understand our place in the universe. Philosophies and religions have emerged as frameworks that guide our ethics and philosophies, providing solace and direction amidst the uncertainties of life. These belief systems underscore a recurring theme in the human narrative: the search for truth and enlightenment.

# In today's globalized world, the notion of humanity is increasingly vital as it encourages us to cherish diversity while recognizing our shared destiny. The intertwining of cultures and the instantaneous exchange of ideas through digital connectivity offer unprecedented opportunities for collaboration. However, this interconnectedness also demands a conscientious approach to ensuring that our actions contribute positively to the global community.

# In essence, humanity embodies both the extraordinary and the ordinary facets of life. It captures our potential to transcend individual limitations in the pursuit of collective betterment. As we navigate the complexities of the modern world, the essence of humanity lies in a delicate balance: recognizing our flaws, cultivating our strengths, and, most importantly, holding onto hope. The enduring challenge is to harness the richness of our diversity to build a future where equity, peace, and prosperity are not mere ideals but tangible realities.
# """
# text = """
# This study presents a comprehensive calculation of prompt diphoton production cross sections at the Tevatron and Large Hadron Collider (LHC) energies. The prompt diphoton process, wherein two photons are produced directly in the hard scattering process, is an essential channel for probing the properties of the Higgs boson and searching for new physics phenomena. To accurately determine the cross sections, we employ state-of-the-art theoretical frameworks incorporating next-to-leading-order (NLO) QCD corrections and parton shower effects. We systematically analyze the energy dependence of the cross sections, examining the impact of various kinematic cuts and isolation criteria commonly used in experimental analyses. By comparing the results with available experimental data, we validate our theoretical predictions and provide useful insights into the sources of theoretical uncertainties. Our calculations serve as valuable references for the design and interpretation of diphoton measurements at both the Tevatron and LHC, crucial for understanding the underlying physics processes and guiding future experimental endeavors in high-energy physics.
# """
# prediction = predict_text(text)
# print(f"Prediction: {'GPT' if prediction == 1 else 'Human'}")


path = "/home/asheesh/Documents/beast/ai-detector-survey/generate_data/data_gpt/*.json"
files = glob(path)[:]
# files = files[:640]+files[650:]
predictions = []
ground_truth = []
for file in tqdm.tqdm(files):
    print('file', file)
    with open(file, "r") as f:
        data = json.load(f)

    if not data["original_paragraph"] or not data["generated_paragraph"]:
        continue

    original_paragraph = data["original_paragraph"]
    doc = nlp(original_paragraph)
    sentences = [sentence.text for sentence in doc.sentences]
    for sentence in sentences:
        prediction = predict_text(sentence)
        predictions.append(prediction)
        ground_truth.append(0)
    generated_paragraph = data["generated_paragraph"]
    doc = nlp(generated_paragraph)
    sentences = [sentence.text for sentence in doc.sentences]
    for sentence in sentences:
        prediction = predict_text(sentence)
        predictions.append(prediction)
        ground_truth.append(1)


# Calculate metrics
conf_matrix = confusion_matrix(ground_truth, predictions)
conf_matrix_normalized = conf_matrix.astype(
    'float') / conf_matrix.sum(axis=1)[:, np.newaxis]

# Create heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_normalized, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=['Human', 'GPT'], yticklabels=['Human', 'GPT'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Normalized Confusion Matrix')
plt.savefig('normalized_confusion_matrix_gpt_sentence_level.png')
plt.close()

print("\nConfusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(classification_report(ground_truth,
      predictions, target_names=['Human', 'GPT']))

accuracy = accuracy_score(ground_truth, predictions)
print(f"\nAccuracy: {accuracy:.4f}")

set_trace()
