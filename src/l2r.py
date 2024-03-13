from tqdm import tqdm
import pandas as pd
import lightgbm
from indexing import InvertedIndex
import multiprocessing
from collections import defaultdict, Counter
import numpy as np
from document_preprocessor import Tokenizer
from ranker import Ranker, TF_IDF, BM25, PivotedNormalization, CrossEncoderScorer
import csv
import math
import os
import pickle

CACHE_PATH = '../__cache__/'

class L2RRanker:
    def __init__(self, document_index: InvertedIndex, title_index: InvertedIndex,
                 document_preprocessor: Tokenizer, stopwords: set[str], ranker: Ranker,
                 feature_extractor: 'L2RFeatureExtractor') -> None:
        """
        Initializes a L2RRanker system.

        Args:
            document_index: The inverted index for the contents of the document's main text body
            title_index: The inverted index for the contents of the document's title
            document_preprocessor: The DocumentPreprocessor to use for turning strings into tokens
            stopwords: The set of stopwords to use or None if no stopword filtering is to be done
            ranker: The Ranker object
            feature_extractor: The L2RFeatureExtractor object
        """
        # TODO: Save any new arguments that are needed as fields of this class
        self.document_index = document_index
        self.title_index = title_index
        self.document_preprocessor = document_preprocessor
        self.stopwords = stopwords
        self.ranker = ranker
        self.feature_extractor = feature_extractor

        # TODO: Initialize the LambdaMART model (but don't train it yet)
        self.model = LambdaMART()
        self.trained = False

    def prepare_training_data(self, query_to_document_relevance_scores: dict[str, list[tuple[int, int]]]):
        """
        Prepares the training data for the learning-to-rank algorithm.

        Args:
            query_to_document_relevance_scores: A dictionary of queries mapped to a list of
                documents and their relevance scores for that query
                The dictionary has the following structure:
                    query_1_text: [(docid_1, relance_to_query_1), (docid_2, relance_to_query_2), ...]

        Returns:
            X (list): A list of feature vectors for each query-document pair
            y (list): A list of relevance scores for each query-document pair
            qgroups (list): A list of the number of documents retrieved for each query
        """
        # NOTE: qgroups is not the same length as X or y
        #       This is for LightGBM to know how many relevance scores we have per query
        X = []
        y = []
        qgroups = []

        # TODO: For each query and the documents that have been rated for relevance to that query,
        #       process these query-document pairs into features
        for query, doc_scores in tqdm(query_to_document_relevance_scores.items()):

            # TODO: Accumulate the token counts for each document's title and content here
            query_parts = self.document_preprocessor.tokenize(query)
            doc_term_counts = self.accumulate_doc_term_counts(
                self.document_index, query_parts)
            title_term_counts = self.accumulate_doc_term_counts(
                self.title_index, query_parts)

            # TODO: For each of the documents, generate its features, then append
            #       the features and relevance score to the lists to be returned
            for doc in doc_scores:
                docid = doc[0]
                score = doc[1]
                features = self.feature_extractor.generate_features(
                    docid, doc_term_counts[docid], title_term_counts[docid], query_parts, query)
                X.append(features)
                y.append(score)

            # Keep track of how many scores we have for this query
            qgroups.append(len(doc_scores))

        return X, y, qgroups

    @staticmethod
    def accumulate_doc_term_counts(index: InvertedIndex, query_parts: list[str]) -> dict[int, dict[str, int]]:
        """
        A helper function that for a given query, retrieves all documents that have any
        of these words in the provided index and returns a dictionary mapping each document id to
        the counts of how many times each of the query words occurred in the document

        Args:
            index: An inverted index to search
            query_parts: A list of tokenized query tokens

        Returns:
            A dictionary mapping each document containing at least one of the query tokens to
            a dictionary with how many times each of the query words appears in that document
        """
        # TODO: Retrieve the set of documents that have each query word (i.e., the postings) and
        #       create a dictionary that keeps track of their counts for the query word
        doc_term_count = defaultdict(Counter)

        relevant_docs = set()
        for word in query_parts:
            if word in index.index:
                relevant_docs.update([x[0] for x in index.index[word]])

        for word in query_parts:
            if word in index.index:
                for index_doc in index.index[word]:
                    if index_doc[0] in relevant_docs:
                        doc_term_count[index_doc[0]][word] = index_doc[1]

        return doc_term_count

    def train(self, training_data_filename: str, model_name: str='') -> None:
        """
        Trains a LambdaMART pair-wise learning to rank model using the documents and relevance scores provided 
        in the training data file.

        Args:
            training_data_filename (str): a filename for a file containing documents and relevance scores
        """
        # TODO: Convert the relevance data into the right format for training data preparation
        if os.path.exists(CACHE_PATH + model_name + 'model.pkl'):
            self.model = pickle.load(open(CACHE_PATH + model_name + 'model.pkl', 'rb'))
            self.trained = True
            return
        if os.path.exists(CACHE_PATH + model_name + 'X.pkl'):
            X = pickle.load(open(CACHE_PATH + model_name + 'X.pkl', 'rb'))
            y = pickle.load(open(CACHE_PATH + model_name + 'y.pkl', 'rb'))
            qgroups = pickle.load(open(CACHE_PATH + model_name + 'qgroups.pkl', 'rb'))
        else:
            query_to_doc_rel_scores = {}
            if training_data_filename.endswith('.csv'):
                with open(training_data_filename, 'r') as f:
                    reader = csv.reader(f)
                    reader.__next__()
                    if training_data_filename.endswith('data-relevance.csv'):
                        for row in tqdm(reader):
                            query = row[0]
                            docid = int(row[1])
                            rel_score = row[2]
                            if query not in query_to_doc_rel_scores:
                                query_to_doc_rel_scores[query] = [
                                    (docid, rel_score)]
                            else:
                                query_to_doc_rel_scores[query].append(
                                    (docid, rel_score))
                    elif training_data_filename.endswith('.train.csv'):
                        for row in tqdm(reader):
                            query = row[0]
                            docid = int(row[2])
                            rel_score = row[4]
                            if query not in query_to_doc_rel_scores:
                                query_to_doc_rel_scores[query] = [
                                    (docid, rel_score)]
                            else:
                                query_to_doc_rel_scores[query].append(
                                    (docid, rel_score))
            elif training_data_filename.endswith('.jsonl'):
                query_to_doc_rel_scores = pd.read_json(
                    training_data_filename, lines=True)
            else:
                raise ValueError("Unsupported file format.")

        # TODO: Prepare the training data by featurizing the query-doc pairs and
        #       getting the necessary datastructures
        X, y, qgroups = self.prepare_training_data(query_to_doc_rel_scores)
        pickle.dump(X, open(CACHE_PATH + model_name + 'X.pkl', 'wb'))
        pickle.dump(y, open(CACHE_PATH + model_name + 'y.pkl', 'wb'))
        pickle.dump(qgroups, open(CACHE_PATH + model_name + 'qgroups.pkl', 'wb'))

        # TODO: Train the model
        print("Training model...")
        self.model.fit(X, y, qgroups)
        pickle.dump(self.model, open(CACHE_PATH + model_name + 'model.pkl', 'wb'))
        self.trained = True

    def predict(self, X):
        """
        Predicts the ranks for featurized doc-query pairs using the trained model.

        Args:
            X (array-like): Input data to be predicted
                This is already featurized doc-query pairs.

        Returns:
            array-like: The predicted rank of each document

        Raises:
            ValueError: If the model has not been trained yet.
        """
        # TODO: Return a prediction made using the LambdaMART model
        if self.trained == False:
            raise ValueError("Model has not been trained yet.")
        return self.model.predict(X)

    # TODO: Implement MMR diversification for a given list of documents and their cosine similarity scores
    @staticmethod
    def maximize_mmr(thresholded_search_results: list[tuple[int, float]], similarity_matrix: np.ndarray,
                     list_docs: list[int], mmr_lambda: int) -> list[tuple[int, float]]:
        """
        Takes the thresholded list of results and runs the maximum marginal relevance diversification algorithm
        on the list.
        It should return a list of the same length with the same overall documents but with different document ranks.

        Args:
            thresholded_search_results: The thresholded search results
            similarity_matrix: Precomputed similarity scores for all the thresholded search results
            list_docs: The list of documents following the indexes of the similarity matrix
                       If document 421 is at the 5th index (row, column) of the similarity matrix,
                       it should be on the 5th index of list_docs.
            mmr_lambda: The hyperparameter lambda used to measure the MMR scores of each document

        Returns:
            A list containing tuples of the documents and their MMR scores when the documents were added to S
        """
        # NOTE: This algorithm implementation requires some amount of planning as you need to maximize
        #       the MMR at every step.
        #       1. Create an empty list S
        #       2. Find the element with the maximum MMR in thresholded_search_results, R (but not in S)
        #       3. Move that element from R and append it to S
        #       4. Repeat 2 & 3 until there are no more remaining elements in R to be processed

        S = []
        
        while len(thresholded_search_results) > 0:
            MR_list = []
            max_id = 0
            max_sim = 0
            for i in range(len(thresholded_search_results)):
                MR = mmr_lambda * thresholded_search_results[i][1]
                if S != []:
                    max_sim = max(similarity_matrix[list_docs.index(thresholded_search_results[i][0])][list_docs.index(doc[0])] for doc in S) 
                    MR -= (1 - mmr_lambda) * max_sim
                MR_list.append((thresholded_search_results[i][0], MR))
            max_id = MR_list.index(max(MR_list, key=lambda x: x[1]))
            S.append(MR_list[max_id])
            thresholded_search_results.pop(max_id)

        return S

    def query(self, query: str, pseudofeedback_num_docs=0, pseudofeedback_alpha=0.8,
              pseudofeedback_beta=0.2, user_id=None, mmr_lambda: int = 1, mmr_threshold: int = 100) -> list[tuple[int, float]]:
        """
        Retrieves potentially-relevant documents, constructs feature vectors for each query-document pair,
        uses the L2R model to rank these documents, and returns the ranked documents.

        Args:
            query: A string representing the query to be used for ranking
            pseudofeedback_num_docs: If pseudo-feedback is requested, the number of top-ranked documents
                to be used in the query
            pseudofeedback_alpha: If pseudo-feedback is used, the alpha parameter for weighting
                how much to include of the original query in the updated query
            pseudofeedback_beta: If pseudo-feedback is used, the beta parameter for weighting
                how much to include of the relevant documents in the updated query
            user_id: the integer id of the user who is issuing the query or None if the user is unknown
            mmr_lambda: Hyperparameter for MMR diversification scoring
            mmr_threshold: Documents to rerank using MMR diversification

        Returns:
            A list containing tuples of the ranked documents and their scores, sorted by score in descending order
                The list has the following structure: [(doc_id_1, score_1), (doc_id_2, score_2), ...]
        """
        # TODO: Retrieve potentially-relevant documents
        query_parts = self.document_preprocessor.tokenize(query)
        if len(query_parts) == 0:
            return []

        # TODO: Fetch a list of possible documents from the index and create a mapping from
        #       a document ID to a dictionary of the counts of the query terms in that document.
        #       You will pass the dictionary to the RelevanceScorer as input
        #
        # NOTE: we collect these here (rather than calling a Ranker instance) because we'll
        #       pass these doc-term-counts to functions later, so we need the accumulated representations
        relevant_docs = set()
        for word in query_parts:
            if word in self.document_index.index:
                relevant_docs.update(
                    [x[0] for x in self.document_index.index[word]])

        # TODO: Accumulate the documents word frequencies for the title and the main body
        doc_term_counts = self.accumulate_doc_term_counts(
            self.document_index, query_parts)
        title_term_counts = self.accumulate_doc_term_counts(
            self.title_index, query_parts)

        # TODO: Score and sort the documents by the provided scorer for just the document's main text (not the title).
        #       This ordering determines which documents we will try to *re-rank* using our L2R model
        results = self.ranker.query(
            query, pseudofeedback_num_docs, pseudofeedback_alpha, pseudofeedback_beta)

        # TODO: Filter to just the top 100 documents for the L2R part for re-ranking
        results_top_100 = results[:100]
        results_tails = results[100:]

        # TODO: Construct the feature vectors for each query-document pair in the top 100
        X_pred = []
        for item in results_top_100:
            docid = item[0]
            X_pred.append(self.feature_extractor.generate_features(
                docid, doc_term_counts[docid], title_term_counts[docid], query_parts, query))

        # TODO: Use your L2R model to rank these top 100 documents
        scores = self.predict(X_pred)

        # TODO: Sort posting_lists based on scores
        for i in range(len(results_top_100)):
            results_top_100[i] = (results_top_100[i][0], scores[i])
        results_top_100.sort(key=lambda x: x[1], reverse=True)

        # TODO: Make sure to add back the other non-top-100 documents that weren't re-ranked
        results = results_top_100 + results_tails

        # TODO (HW5): Run MMR diversification for appropriate values of lambda

        # TODO (HW5): Get the threholded part of the search results, aka top t results and
        #      keep the rest separate
        results_tails = results[mmr_threshold:]

        # TODO (HW5): Get the document similarity matrix for the thresholded documents using vector_ranker
        #      Preserve the input list of documents to be used in the MMR function
        if self.ranker.__class__.__name__ == 'VectorRanker':
            list_docs = [item[0] for item in results[:mmr_threshold]]
            document_similarity = self.ranker.document_similarity(list_docs)

        # TODO (HW5): Run the maximize_mmr function with appropriate arguments
            results_thres = self.maximize_mmr(
                results[:mmr_threshold], document_similarity, list_docs, mmr_lambda)

        # TODO (HW5): Add the remaining search results back to the MMR diversification results
            results = results_thres + results_tails

        # TODO: Return the ranked documents
        return results

    def save_model(self) -> None:
        pickle.dump(self.model.ranker, open('../cache/l2r_model_' +
                    self.ranker.__class__.__name__ + '.pkl', 'wb'))

    def load_model(self) -> None:
        self.model.ranker = pickle.load(
            open('../cache/l2r_model_' + self.ranker.__class__.__name__ + '.pkl', 'rb'))


class L2RFeatureExtractor:
    def __init__(self, document_index: InvertedIndex, title_index: InvertedIndex,
                 doc_category_info: dict[int, list[str]],
                 document_preprocessor: Tokenizer, stopwords: set[str],
                 recognized_categories: set[str], docid_to_network_features: dict[int, dict[str, float]],
                 ce_scorer: CrossEncoderScorer = None) -> None:
        """
        Initializes a L2RFeatureExtractor object.

        Args:
            document_index: The inverted index for the contents of the document's main text body
            title_index: The inverted index for the contents of the document's title
            doc_category_info: A dictionary where the document id is mapped to a list of categories
            document_preprocessor: The DocumentPreprocessor to use for turning strings into tokens
            stopwords: The set of stopwords to use or None if no stopword filtering is to be done
            recognized_categories: The set of categories to be recognized as binary features
                (whether the document has each one)
            docid_to_network_features: A dictionary where the document id is mapped to a dictionary
                with keys for network feature names "page_rank", "hub_score", and "authority_score"
                and values with the scores for those features
            ce_scorer: The CrossEncoderScorer object
        """
        # TODO: Set the initial state using the arguments
        self.document_index = document_index
        self.title_index = title_index
        self.doc_category_info = doc_category_info
        self.document_preprocessor = document_preprocessor
        self.stopwords = stopwords
        self.recognized_categories = recognized_categories
        self.docid_to_network_features = docid_to_network_features

        # TODO: For the recognized categories (i.e,. those that are going to be features), consider
        #       how you want to store them here for faster featurizing

        # TODO: Initialize any RelevanceScorer objects needed to support the methods below.
        #             Be sure to use the right InvertedIndex object when scoring
        self.tf_idf_scorer = TF_IDF(self.document_index)
        self.bm25_scorer = BM25(self.document_index)
        self.pivoted_norm_scorer = PivotedNormalization(self.document_index)
        self.ce_scorer = ce_scorer

    # TODO: Article Length
    def get_article_length(self, docid: int) -> int:
        """
        Gets the length of a document (including stopwords).

        Args:
            docid: The id of the document

        Returns:
            The length of a document
        """
        return self.document_index.get_doc_metadata(docid)['length']

    # TODO: Title Length
    def get_title_length(self, docid: int) -> int:
        """
        Gets the length of a document's title (including stopwords).

        Args:
            docid: The id of the document

        Returns:
            The length of a document's title
        """
        return self.title_index.get_doc_metadata(docid)['length']

    # TODO: TF
    def get_tf(self, index: InvertedIndex, docid: int, word_counts: dict[str, int], query_parts: list[str]) -> float:
        """
        Calculates the TF score.

        Args:
            index: An inverted index to use for calculating the statistics
            docid: The id of the document
            word_counts: The words in some part of a document mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The TF score
        """
        score = 0
        query_index = Counter(query_parts)
        for item in query_index:
            if item not in word_counts:
                continue
            score += math.log(word_counts[item] + 1)
        return score

    # TODO: TF-IDF
    def get_tf_idf(self, index: InvertedIndex, docid: int,
                   word_counts: dict[str, int], query_parts: list[str]) -> float:
        """
        Calculates the TF-IDF score.

        Args:
            index: An inverted index to use for calculating the statistics
            docid: The id of the document
            word_counts: The words in some part of a document mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The TF-IDF score
        """
        query_word_count = Counter(query_parts)
        return self.tf_idf_scorer.score(docid, word_counts, query_word_count)

    # TODO: BM25
    def get_BM25_score(self, docid: int, doc_word_counts: dict[str, int],
                       query_parts: list[str]) -> float:
        """
        Calculates the BM25 score.

        Args:
            docid: The id of the document
            doc_word_counts: The words in the document's main text mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The BM25 score
        """
        # TODO: Calculate the BM25 score and return it
        query_word_count = Counter(query_parts)
        return self.bm25_scorer.score(docid, doc_word_counts, query_word_count)

    # TODO: Pivoted Normalization
    def get_pivoted_normalization_score(self, docid: int, doc_word_counts: dict[str, int],
                                        query_parts: list[str]) -> float:
        """
        Calculates the pivoted normalization score.

        Args:
            docid: The id of the document
            doc_word_counts: The words in the document's main text mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The pivoted normalization score
        """
        # TODO: Calculate the pivoted normalization score and return it
        query_word_count = Counter(query_parts)
        return self.pivoted_norm_scorer.score(docid, doc_word_counts, query_word_count)

    # TODO: Document Categories
    def get_document_categories(self, docid: int) -> list:
        """
        Generates a list of binary features indicating which of the recognized categories that the document has.
        Category features should be deterministically ordered so list[0] should always correspond to the same
        category. For example, if a document has one of the three categories, and that category is mapped to
        index 1, then the binary feature vector would look like [0, 1, 0].

        Args:
            docid: The id of the document

        Returns:
            A list containing binary list of which recognized categories that the given document has
        """
        cat_list = [0] * len(self.recognized_categories)
        if docid in self.doc_category_info:
            for id, category in enumerate(self.recognized_categories):
                if category in self.doc_category_info[docid]:
                    cat_list[id] = 1
        return cat_list

    # TODO: PageRank
    def get_pagerank_score(self, docid: int) -> float:
        """
        Gets the PageRank score for the given document.

        Args:
            docid: The id of the document

        Returns:
            The PageRank score
        """
        score = 0
        if docid in self.docid_to_network_features:
            score = self.docid_to_network_features[docid]['pagerank']
        return score

    # TODO: HITS Hub
    def get_hits_hub_score(self, docid: int) -> float:
        """
        Gets the HITS hub score for the given document.

        Args:
            docid: The id of the document

        Returns:
            The HITS hub score
        """
        score = 0
        if docid in self.docid_to_network_features:
            score = self.docid_to_network_features[docid]['hub_score']
        return score

    # TODO: HITS Authority
    def get_hits_authority_score(self, docid: int) -> float:
        """
        Gets the HITS authority score for the given document.

        Args:
            docid: The id of the document

        Returns:
            The HITS authority score
        """
        score = 0
        if docid in self.docid_to_network_features:
            score = self.docid_to_network_features[docid]['authority_score']
        return score

    # TODO: Cross-Encoder Score
    def get_cross_encoder_score(self, docid: int, query: str) -> float:
        """
        Gets the cross-encoder score for the given document.

        Args:
            docid: The id of the document
            query: The query in its original form (no stopword filtering/tokenization)

        Returns:
            The Cross-Encoder score
        """
        return self.ce_scorer.score(docid, query)

    # TODO: Add at least one new feature to be used with your L2R model
    def generate_features(self, docid: int, doc_word_counts: dict[str, int],
                          title_word_counts: dict[str, int], query_parts: list[str],
                          query: str) -> list:
        """
        Generates a dictionary of features for a given document and query.

        Args:
            docid: The id of the document to generate features for
            doc_word_counts: The words in the document's main text mapped to their frequencies
            title_word_counts: The words in the document's title mapped to their frequencies
            query_parts : A list of tokenized query terms to generate features for
            query: The query in its original form (no stopword filtering/tokenization)

        Returns:
            A vector (list) of the features for this document
                Feature order should be stable between calls to the function
                (the order of features in the vector should not change).
        """
        # NOTE: We can use this to get a stable ordering of features based on consistent insertion
        #       but it's probably faster to use a list to start

        feature_vector = []

        # TODO: Document Length
        feature_vector.append(self.get_article_length(docid))

        # TODO: Title Length
        feature_vector.append(self.get_title_length(docid))

        # TODO: Query Length
        feature_vector.append(len(query_parts))

        # TODO: TF (document)
        feature_vector.append(self.get_tf(
            self.document_index, docid, doc_word_counts, query_parts))

        # TODO: TF-IDF (document)
        feature_vector.append(self.get_tf_idf(
            self.document_index, docid, doc_word_counts, query_parts))

        # TODO: TF (title)
        feature_vector.append(self.get_tf(
            self.title_index, docid, title_word_counts, query_parts))

        # TODO: TF-IDF (title)
        feature_vector.append(self.get_tf_idf(
            self.title_index, docid, title_word_counts, query_parts))

        # TODO: BM25
        feature_vector.append(self.get_BM25_score(
            docid, doc_word_counts, query_parts))

        # TODO: Pivoted Normalization
        feature_vector.append(self.get_pivoted_normalization_score(
            docid, doc_word_counts, query_parts))

        # TODO: PageRank
        feature_vector.append(self.get_pagerank_score(docid))

        # TODO: HITS Hub
        feature_vector.append(self.get_hits_hub_score(docid))

        # TODO: HITS Authority
        feature_vector.append(self.get_hits_authority_score(docid))

        # TODO: Cross-Encoder Score
        feature_vector.append(self.get_cross_encoder_score(docid, query))

        # TODO: Add at least one new feature to be used with your L2R model

        # TODO: Document Categories
        #       This should be a list of binary values indicating which categories are present
        feature_vector.extend(self.get_document_categories(docid))

        return feature_vector


class LambdaMART:
    def __init__(self, params=None) -> None:
        """
        Initializes a LambdaMART (LGBRanker) model using the lightgbm library.

        Args:
            params (dict, optional): Parameters for the LGBMRanker model. Defaults to None.
        """
        default_params = {
            'objective': "lambdarank",
            'boosting_type': "gbdt",
            'n_estimators': 20,
            'importance_type': "gain",
            'metric': "ndcg",
            'num_leaves': 20,
            'learning_rate': 0.005,
            'max_depth': -1,
            "n_jobs": multiprocessing.cpu_count()-1,
            "verbosity": 1,
        }

        if params:
            default_params.update(params)

        # TODO: Initialize the LGBMRanker with the provided parameters and assign as a field of this class
        self.params = default_params
        self.ranker = lightgbm.LGBMRanker(**self.params)

    def fit(self, X_train, y_train, qgroups_train):
        """
        Trains the LGBMRanker model.

        Args:
            X_train (array-like): Training input samples
            y_train (array-like): Target values
            qgroups_train (array-like): Query group sizes for training data

        Returns:
            self: Returns the instance itself
        """
        # TODO: Fit the LGBMRanker's parameters using the provided features and labels
        X_train = np.array(X_train)
        self.ranker.fit(X_train, y_train, group=qgroups_train)

    def predict(self, featurized_docs):
        """
        Predicts the target values for the given test data.

        Args:
            featurized_docs (array-like): 
                A list of featurized documents where each document is a list of its features
                All documents should have the same length

        Returns:
            array-like: The estimated ranking for each document (unsorted)
        """
        # TODO: Generate the predicted values using the LGBMRanker
        featurized_docs = np.array(featurized_docs)
        if len(featurized_docs.shape) == 1:
            return []
        ypred = self.ranker.predict(featurized_docs)
        return ypred


if __name__ == '__main__':
    pass
