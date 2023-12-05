## Start the server

This is a Python 3.11 FastAPI project with the necessary requirements added to the `requirements.txt`. After configuring the virtual environment, run

```
python -m pip install -r requirements.txt
```

This should install the libraries you need to start the server. After you have all of the files and the necessary Python requirements installed in your environment, run 

```
python -m uvicorn app:app
```

to start the server.

---

## Data

1. The `wikipedia_1M_dataset.jsonl.gz` dataset contains 1 million Wikipedia articles. The `wikipedia_200k_dataset.jsonl.gz` dataset is a reduced version containing 200 thousand Wikipedia articles. Both datasets are in JSONL format where each lines is a separate JSON of the following format. 

```
{
    "docid": <document id>
    "title": <document title>
    "text": <the entire text of the document>
    "categories": [<each Wikipedia category>]
}
```

2. The collected relevance scores are in 

- `relevance.dev.csv`
- `relevance.train.csv`
- `relevance.test.csv`

3. `doc2query.csv` has the queries generated for all of the 200k documents using `doc2query/msmarco-t5-base-v1`. *This should be used while indexing, not in document preprocessor.*

4. `wiki-200k-vecs.msmarco-MiniLM-L12-cos-v5.npy` has all the document embeddings meant to be used in the `VectorRanker`. Use `numpy.load` to load the numpy matrix and the embedding insertion order follows the document order in the `wikipedia_200k_dataset.jsonl`.

5. `personalization.jsonl` contains information on the two simulated users
   - This file contains the documents in the seed set.

---

## `pipeline.py`

**The name of the index folder to be generated is set in the `pipeline.py` file.**

- Sample system configurations:
  - L2R system with a BM25 ranker
  - L2R system with a BM25 ranker and pseudofeedback
  - L2R system with a VectorRanker and pseudofeedback

### `SearchEngine`

- This code will run the search engine locally.

- `__init__` : Standard initialization specification:
  * Use the `RegexTokenizer`.
  * Load in the stopwords here to pass them to the relevant class's `__init__` functions.
  * Create separate `BasicInvertedIndex` objects for the main body of the argument and the title text.
  * Use the `L2RRanker` instead of the `Ranker`.

---

## `document_preprocessor.py`

### `Tokenizer` class

- _General Comment_: All Tokenizer subclasses should follow the pattern of using the base class's `postprocess` once they have applied their class-specific logic for how to turn the string into an initial list of tokens. 

- `__init__(self, lowercase: bool= True, multiword_expressions: list[str]=None) -> None`
  * `lowercase: bool` (default `True`): Enable lowercasing as option.
  * `multiword_expressions: list[str]` (default `None`): Supply the MWE into the `Tokenizer`.
   
- `find_and_replace_mwes(self, input_tokens: list[str]) -> list[str]`
  * Function to find and replace the MWEs.

- `postprocess(self, input_tokens: list[str]) -> list[str]`
  * Function to apply MWE and lowercasing, if enabled (**NOTE** Only lowercase is enabled).
  * After tokenizing in each of the child classes, pass through this function.

### `SplitTokenizer` class
- `__init__(self, lowercase: bool= True, multiword_expressions: list[str]=None) -> None`

### `RegexTokenizer` class
- `__init__(self, token_regex: str, lowercase: bool= True, multiword_expressions: list[str]=None) -> None`
  * `token_regex: str`: The regular expression used.

### `Doc2QueryAugmenter` class

- _General Comment_: This class should be a functional piece of code which takes in a doc2query model and can generate queries from a piece of text. **DO NOT** generating queries for all 200k documents on a laptop. For downstream tasks such as index augmentation with the queries, use `doc2query.csv`.

---

## `indexing.py`

### `BasicInvertedIndex`

- `__init__(self)`

- `save(self, index_directory_name)`
  * This function takes an argument for which directory to save the index file to. 
  * Save stores _all_ state needed to rank documents. Any metadata needed is stored in some file. Multiple files are used to save the document metadata and the collection statistics.

- `load(self, index_directory_name)`
  * This function takes an argument for which directory to load the index data from.

### `Indexer`

- `def create_index(index_type: IndexType, dataset_path: str,
                     document_preprocessor: Tokenizer, stopwords: set[str],
                     minimum_word_frequency: int, text_key="text",
                     max_docs: int = -1, doc_augment_dict: dict[int, list[str]] | None = None)`
  * `stopwords: set[bool]` 
    - A set of stopwords provided or `None` or an empty `set` to not perform stopwords filtering.
  * `text_key` 
    - This argument specifies which key to use in the dataset's JSON object for getting text when creating the index. It defaults to "text" but when you create an index for the title, you'll need to change its value.
  * `max_docs`
    - This specifies the maximum number of document to process from the dataset (ignoring the rest). If the argument's value is -1, use all the documents. This argument is helpful for testing settings where an index of real data is acquired without loading in all the data.
  * **Important Note** `minimum_word_frequency` needs to be computed based on frequencies at the collection-level, _not_ at the document level. This means if some minimum is required, the entire collection needs to be read and tokenized first to count word frequencies and _then_ the collection should be re-read to filter based on these counts. With 200K documents, the reference implementation takes around 90 seconds for the first pass (token counting) and 120 seconds to index everything.
  * *Optional* argument `doc_augment_dict`. This dict should be created from the `doc2query.csv` The keys are the document id and the values are the list of queries for a particular document. The augmentation of the document should happen before all of the preprocessing steps, i.e., before stopwords removal or minimum word filtering. 

---

## `ranker.py`

### `Ranker` class

- `__init__(self, index, document_preprocessor, stopwords: set[str], scorer: 'RelevanceScorer') -> None:`
  * `stopwords: set[str]`
    * Supply the stopwords directly into the `Ranker`.
  * `scorer` **IMPORTANT** Pass an instantiated `RelevanceScorer` to this function.
  * `raw_text_dict`: A dictionary mapping a document ID to the raw string of the document.
    
- `query(self, query: str) -> list[tuple[int, float]]`
  * **IMPORTANT** During the process of retrieving the list of postings for each of the terms in the query, accumulate the word frequencies for each of the documents in the postings (from the data in the list) for use with the new `RelevanceScorer.score`  method. This means that a mapping is created from a document ID to a dictionary of the counts of the query terms in that document. Note that this is _not_ the full count of all words' frequencies for that document---just the query terms. The data collected with this code will replace the slower `bisect`-based code in the `RelevanceScorer.score` method that was doing the same accumulation in practice. 
  * Pass each document's term-frequency dictionary to the `RelevanceScorer` as input. Again, note that this is just a _subset_ of the document's terms that has been pre-filtered to just those terms in the query.
  * **IMPORTANT** Standardize the `query` output of `Ranker` to match with `L2RRanker.query` and `VectorRanker.query`. The `query` function returns a sorted list of tuples where each tuple has the first element as the document id and the second element as the score of the document after the ranking process. 
  * `pseudofeedback_num_docs`: If pseudo-feedback is requested, the number of top-ranked documents to be used in the query, default is 0 (not using pseudo-feedback).
  * `pseduofeedback_alpha`: If pseudo-feedback is used, the alpha parameter for weighting how much to include of the original query in the updated query.
  * `pseduofeedback_beta`: If pseudo-feedback is used, the beta parameter for weighting how much to include of the relevant documents in the updated query.
  * `user_id`: The integer id of the user who is issuing the query or None if the user is unknown.
  * Within `query` the pseudodocument from the specified number of pseudo-relevant results is created. 
 
### `RelevanceScorer` class
- `score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float`
  * Argument (mentioned in `Ranker.query` above) `doc_word_counts`: a dictionary containing the frequencies of all the query words in the document. **Note** Use the counts to simplify the logic and speed up relevance ranking in the process.
  * The `score` function returns _only_ a `float`: a score for how relevant the document is, where higher scores are more relevant.
  * `query_word_counts: dict[str, int]`: A dictionary containing all words in the query and their frequencies (Words that have been filtered will be None).
    * Implemented with `Counter(query_parts)`. It is more efficient to compute this query word-count counter once when calling `score` often.
  * `PersonalizedBM25` class
    * `__init__(self, index: InvertedIndex, relevant_doc_index: InvertedIndex,
                  parameters={'b': 0.75, 'k1': 1.2, 'k3': 8}) -> None:`
    * `revelant_doc_index`: The inverted index of only documents a user has rated as relevant, which is used when calcuating the personalized part of BM25

### `CrossEncoderScorer` class

- _General Comment_: This class uses a pre-trained cross-encoder model from the Sentence Transformers package to score a given query-document pair. Since the cross-encoder model can process a maximum of 512 tokens, a dictionary that maps the document ID to a text (string) with the first 500 words in the document is created. Then, pass the dictionary to the class as an argument. Note that the cross-encoder model receives raw strings as input; Neither filtering stopwords nor tokenizing a query or document text is done before feeding them into the model.

---

### `network_features.py`

#### `NetworkFeatures`

- This class is going to generate the PageRank and HITS features for all the documents using the link network. 
- This class is run _separately_ from the main part of the IR system to generate the network features _once_. Do not re-generate these every time on start.
- Most of this class is dealing with the `sknetwork` library (scikit-network) to load a network and call functions. 
- It is estimated that the entire network construction memory requirement is under 5GB based on tests with the reference implementation.
- `calculate_page_rank(self, graph, damping_factor=0.85, iterations=100, weights=None) -> list[float]:` has a `weights` argument for Personalized PageRank
- `weights`: A data structure containing the restart distribution as a vector (over the length of nodes) or a dict {node: weight}
- check https://scikit-network.readthedocs.io/en/latest/reference/ranking.html#pagerank

---

## `l2r.py`

- ` def __init__(self, document_index: InvertedIndex, title_index: InvertedIndex,
                 document_preprocessor: Tokenizer, stopwords: set[str], ranker: Ranker,
                 feature_extractor: 'L2RFeatureExtractor')`

### `L2RFeatureExtractor`

- This class is responsible for turning query-document pairs into a vector of features that could indicate how relevant the document is.
- Uses existing classes when generating features, e.g., calculating TF-IDF

### `LambdaMART`

- This class is responsible for wrapping the `LightGBM` model and calling the appropriate training and predicting functions on it. 

### `L2RRanker`

- This class is where the basic `Ranker`-like functions uses a `LambdaMART` learning to rank model under the hood. 

- `maximize_mmr`: Takes the thresholded list of results and runs the maximum marginal relevance diversification algorithm on the list. 
  - Arguments:
    - `thresholded_search_results`: The thresholded search results. 
    - `similarity_matrix`: Precomputed similarity scores for all the thresholded search results. 
    - `list_docs`: The list of documents following the indexes of the similarity matrix.
      - If document 421 is at the 5th index (row, column) of the similarity matrix,
      - it should be on the 5th index of list_docs
    - `mmr_lambda`: The hyperparameter lambda used to measure the MRR scores of each document.
  - Returns a list of the same length with the same overall documents but with different document ranks. 
- In `query`, run MRR diversification for appropriate values of lambda by calling `maximize_mmr` to rerank at the very end. 

---

## `vector_ranker.py`

### `vector_ranker.py` implements a vector-based ranker that utilizes pre-trained models from the HuggingFace Transformers library to generate document embeddings and query embeddings.

`VectorRanker` Class inherits from the Ranker class and contains the following:
- `def __init__(self, bi_encoder_model_name: str, encoded_docs: ndarray, row_to_docid: list[int]) -> None`:
    - Instantiates the Sentence Transformer model and accepts the following parameters:
      - bi_encoder_model_name (str): The name of a HuggingFace model used to initialize a Sentence Transformer model.
      - encoded_docs (ndarray): A matrix where each row represents an already-encoded document using the same encoding as specified by `bi_encoder_model_name`.
      - row_to_docid (list[int]): A list that maps row numbers to the document IDs corresponding to the embeddings.

 -  `def query(self, query: str) -> list[tuple[int, float]]`:
    - Takes a query string as input, encodes the query into a vector, and calculates the relevance scores between the query and all documents.
    - Returns a sorted list of tuples, where each tuple contains a document ID and its relevance score, with the most relevant documents ranked first.
    - Compute the dot product between the query vector and document vectors.

- Similar to the `Ranker` class, the `VectorRanker` class' `query` function has arguments for pseudo-feedback. 
  - `pseudofeedback_num_docs`: If pseudo-feedback is requested, the number of top-ranked documents to be used in the query, default is 0 (not using pseudo-feedback)
  - `pseduofeedback_alpha`: If pseudo-feedback is used, the alpha parameter for weighting how much to include of the original query in the updated query
  - `pseduofeedback_beta`: If pseudo-feedback is used, the beta parameter for weighting how much to include of the relevant documents in the updated query
 
- If using pseudo-feedback, after encoding the query using the biencoder
  -  Get the most-relevant document vectors for the initial query
  -  Compute the average vector of the most relevant docs
  -  Combine the original query doc with the feedback doc to use as the new query embedding

- `document_similarity`: Compute the `similarity_matrix` in `L2RRanker.maximize_mmr`
  - Can use the dot product here, since the vectors are normalized for the default models we use
  - Return a matrix (np.ndarray) where element [i][j] represents the similarity between list_docs[i] and list_docs[j]

---

## `relevance.py`

- `map_score(search_result_relevances: list[int], cut_off=10) -> float`

- `def ndcg_score(search_result_relevances: list[float], ideal_relevance_score_ordering: list[float], cut_off=10)`

- `def run_relevance_tests(relevance_data_filename: str, ranker) -> dict[str, float]`
  * `relevance_data_filename`: Specify which file to read the relevance scores from. 

- `nfairr_score`: Computes the normalized Fairness of Retrieval Results (NFAIRR) score for a list of omega values for the list of ranked documents.
  - If all documents are from the protected class, then the NFAIRR score is 0
  - Arguments:
    - `actual_omega_values`: The omega value for a ranked list of documents. The most relevant document is the first item in the list.
    - `cut_off`: The rank cut-off to use for calculating NFAIRR. Omega values in the list after this cut-off position are not used. The default is 200.
  - Returns the NFAIRR score.
- `run_fairness_test`:
  - Implement NFaiRR metric for a list of queries to measure fairness for those queries
  - NOTE: This has no relation to relevance scores and measures fairness of representation of classes 
  - Arguments:
    - `attributes_file_path`: The path where `person-attributes.csv` is saved
    - `protected_class`: The protected class (e.g., Race)
    - `queries`: The list of queries
    - `ranker`: a Ranker
    - `cutoff`: The rank cut-off to use for calculating NFAIRR

---

## `Interactive_Example.ipynb `

- This is a Jupyter notebook that can help you walk through the various steps you need to run everything. Feel free to use this for interactive debugging and development, in addition to the unit tests.

---

## How to use the public test cases

- To run individual test cases, in your terminal, run:
  * `python [filename] [class].[function]`
  * ex: `python test_relevance_scorers.py TestRankingMetrics.test_bm25_single_word_query`
 
- To run one class's tests from file, in your terminal, run:
  * `python [filename] [class] -vvv`
  * ex: `python test_indexing.py TestBasicInvertedIndex -vvv`

- To run all the tests from file, in your terminal, run:
  * `python [filename] -vvv`
  * ex: `python test_indexing.py -vvv`


- To add your own test cases, the basic structure is as follows:
  
```
import unittest

class TestStringMethods(unittest.TestCase):

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

if __name__ == '__main__':
    unittest.main()
  
```
