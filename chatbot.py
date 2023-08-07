import csv
import typing

import openai  # for calling the OpenAI API
import pandas as pd  # for storing text and embeddings data
import tiktoken  # for counting tokens
from scipy import spatial  # for calculating vector similarities for search
from openai_keys import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

# models
EMBEDDING_MODEL = "text-embedding-ada-002"  # OpenAI's best embeddings as of Apr 2023
GPT_MODEL = "gpt-3.5-turbo"
CSV_PATH = "EatsDB - Dishes.csv"

BATCH_SIZE = 2048  # you can submit up to 2048 embedding inputs per request


eatsdb_strings = []
with open(CSV_PATH) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        # Drop all rows where the below columns have the same value
        if row['name_en'] == row['summary_en']:
            # print(row)
            continue

        # omit_keys = ['locale_id', 'wikipedia_url_en', 'image_url']
        include_keys = ['name_native', 'name_en', 'Geo Region', 'summary_en']
        text = '\n'.join([row[k] for k in include_keys])
        eatsdb_strings.append(text)

# print(eatsdb_strings)
# exit()

embeddings = []
for batch_start in range(0, len(eatsdb_strings), BATCH_SIZE):
    batch_end = batch_start + BATCH_SIZE
    batch = eatsdb_strings[batch_start:batch_end]
    # print(f"Batch {batch_start} to {batch_end-1}")

    response = openai.Embedding.create(model=EMBEDDING_MODEL, input=batch)
    response = typing.cast(dict, response)  # satisfy type checker    
    for i, be in enumerate(response["data"]):
        assert i == be["index"]  # double check embeddings are in same order as input
    batch_embeddings = [e["embedding"] for e in response["data"]]
    embeddings.extend(batch_embeddings)

df = pd.DataFrame({"text": eatsdb_strings, "embedding": embeddings})

# convert embeddings from CSV str type back to list type
# df['embedding'] = df['embedding'].apply(ast.literal_eval)


# search function
def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 100
) -> tuple[list[str], list[float]]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    query_embedding_response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = query_embedding_response["data"][0]["embedding"] # type: ignore
    strings_and_relatednesses = [
        (row["text"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n] # type: ignore


# # examples
# strings, relatednesses = strings_ranked_by_relatedness("stir-fry", df, top_n=5)
# for string, relatedness in zip(strings, relatednesses):
#     print(f"{relatedness=:.3f}")
#     print(string)


def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def query_message(
    query: str,
    df: pd.DataFrame,
    model: str,
    token_budget: int
) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    strings, relatednesses = strings_ranked_by_relatedness(query, df)
    # print(query, strings, relatednesses)

    introduction = 'Use the below articles on food to answer the subsequent question. If the answer cannot be found in the articles, write "I could not find an answer."'
    # introduction = 'Use the below articles on food to answer the subsequent question.'
    question = f"\n\nQuestion: {query}"
    message = introduction
    for string in strings:
        next_article = f'\n\nEatsDB article section:\n"""\n{string}\n"""'
        if (
            num_tokens(message + next_article + question, model=model)
            > token_budget
        ):
            break
        else:
            message += next_article
    return message + question


def ask(
    query: str,
    df: pd.DataFrame = df,
    model: str = GPT_MODEL,
    token_budget: int = 4096 - 500
) -> str:
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
    message = query_message(query, df, model=model, token_budget=token_budget)
    messages = [
        {"role": "system", "content": "You answer questions about food."},
        {"role": "user", "content": message},
    ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0
    )
    response_message = response["choices"][0]["message"]["content"] # type: ignore
    return response_message


# query = 'What is the difference between a torta and a tostada?'
# print(query, ask(query))
#
# query = 'What is a Japanese soup that features both sweet and savory flavors?'
# print(query, ask(query))
#
# query = 'What is a Japanese curry that features both sweet and savory flavors?'
# print(query, ask(query))
