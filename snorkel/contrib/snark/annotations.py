from collections import defaultdict
from functools import partial

from models.candidate import wrap_candidate
from snorkel.annotations import load_feature_matrix, load_label_matrix
from snorkel.features import get_span_feats, get_token_count_feats
from snorkel.models.annotation import Feature, FeatureKey, Label, LabelKey
from snorkel.models.meta import snorkel_conn_string
from snorkel.models.views import create_serialized_candidate_view

import logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

class SparkLabelAnnotator:
    """
    Distributes candidates to a Spark cluster and applies labeling functions
    over them. See snorkel.annotations.LabelAnnotator.
    """
    def __init__(self, snorkel_session, spark_session, candidate_class, split_cache=None):
        """
        Constructor

        :param snorkel_session: the SnorkelSession for the Snorkel application
        :param spark_session: a PySpark SparkSession
        :param candidate_class: the subclass of Candidate to be labeled
        :param split_cache: optional split cache, which can be reused across annotators
        """
        self.snorkel_session = snorkel_session
        self.spark_session = spark_session
        self.candidate_class = candidate_class

        self.split_cache = split_cache or {}

        if split_cache is None:
            create_serialized_candidate_view(snorkel_session, candidate_class)

    def apply(self, LFs, split, use_cached=True):
        """
        Applies a collection of labeling functions to a split of candidates.

        The results are persisted in the Snorkel database.

        :param LFs: collection of labeling functions
        :param split: the split of candidates to label
        :param use_cached: If True, distributed candidate sets are cached by
            split for the annotator, so  repeated calls to apply() will operate
            over the same split of candidates as the first call for that split,
            regardless of how the underlying database changes. Intended for e.g.
            usage during iterative development of LFs given fixed Candidate set.

        :return: a csr_LabelMatrix of the resulting labels
        """
        self._clear_labels()

        if split not in self.split_cache or not use_cached:
            self._load_candidates(split)
        else:
            print("Using cached Candidate set for split %s" % split)
        print('Applying labelling functions to %d candidates' % self.split_cache[split].count())

        # Bulk insert the LF labels (output values)
        key_query = LabelKey.__table__.insert()
        label_tuples = []
        for lf in LFs:
            lf_id = self.snorkel_session.execute(key_query,
                {'name': lf.__name__, 'group': 0}).inserted_primary_key[0]

            labels = self.split_cache[split].map(lambda c: (c.id, lf(c)))
            labels.filter(lambda (_, value): value != 0 and value is not None)
            for cid, value in labels.toLocalIterator():
                label_tuples.append({
                    'candidate_id': cid, 'key_id': lf_id, 'value': value})

                # Periodically flushes the labels to disk
                if len(label_tuples) >= 100000:
                    self.snorkel_session.execute(Label.__table__.insert(), label_tuples)
                    label_tuples = []
        # Flushes the remaining labels
        self.snorkel_session.execute(Label.__table__.insert(), label_tuples)

        # Return label matrix from the Snorkel DB
        return load_label_matrix(self.snorkel_session, split=split)

    def _clear_labels(self):
        """
        Clears the database of labels
        """
        self.snorkel_session.query(Label).delete(synchronize_session='fetch')

        query = self.snorkel_session.query(LabelKey).filter(LabelKey.group == 0)
        query.delete(synchronize_session='fetch')

    def _load_candidates(self, split):
        """
        Loads a set of candidates as a Spark RDD and caches the results as
        self.split_cache[split]

        :param split: the split of candidates to load
        """
        jdbcDF = self.spark_session.read \
            .format("jdbc") \
            .option("url", "jdbc:" + snorkel_conn_string) \
            .option("dbtable",
                self.candidate_class.__tablename__ + "_serialized") \
            .load()

        rdd = jdbcDF.rdd.map(partial(
            wrap_candidate,
            class_name=self.candidate_class.__name__,
            argnames=self.candidate_class.__argnames__
        ))
        rdd = rdd.filter(lambda c: c.split==split)
        rdd = rdd.setName("Snorkel Candidates, Split " + str(split) + \
            " (" + self.candidate_class.__name__ + ")")
        self.split_cache[split] = rdd.cache()


class SparkFeatureAnnotator:
    """
    Distributes candidates to a Spark cluster and applies feature generators
    over them. See snorkel.annotations.FeatureAnnotator.
    """
    def __init__(self, snorkel_session, spark_session, candidate_class, f=get_span_feats, split_cache=None):
        """
        Constructor

        :param snorkel_session: the SnorkelSession for the Snorkel application
        :param spark_session: a PySpark SparkSession
        :param candidate_class: the subclass of Candidate to be labeled
        :param f: a feature generating function, defaults to get_span_feats
        :param split_cache: optional split cache, which can be reused across annotators
        """
        self.snorkel_session = snorkel_session
        self.spark_session = spark_session
        self.f = f
        self.candidate_class = candidate_class
        self.split_cache = split_cache or {}
        self.key_cache = {}

        if split_cache is None:
            create_serialized_candidate_view(snorkel_session, candidate_class)

    def apply(self, split, use_cached=True):
        """
        Applies a feature generating function to a split of candidates.

        The results are persisted in the Snorkel database.

        :param split: the split of candidates to label

        :return: a csr_FeatureMatrix of the resulting labels
        """
        self._clear_features()

        if split not in self.split_cache or not use_cached:
            self._load_candidates(split)
        else:
            print("Using cached Candidate set for split %s" % split)
        print('Generating features for %d candidates' % self.split_cache[split].count())

        total_features = 0

        # Bulk insert the LF labels (output values)
        temp_feature_tuples = []

        def feature_fn(candidate):
            from treedlib import compile_relation_feature_generator
            features = defaultdict(float)
            for feature, value in get_span_feats(candidate):
                features[feature] += value
            return [(candidate.id, feature, value) for feature, value in features.iteritems()]

        def insert_features(temp_feature_tuples):
            LOGGER.info('Inserting %d features', len(temp_feature_tuples))
            key_names = set([_t['key_name'] for _t in temp_feature_tuples])
            for key_name in key_names:
                if key_name not in self.key_cache:
                    key_args = {'name': key_name}
                    key_id = self.snorkel_session.execute(key_insert_query, key_args).inserted_primary_key[0]
                    self.key_cache[key_name] = key_id
            feature_tuples = [
                {
                    'candidate_id': _t['candidate_id'],
                    'key_id': self.key_cache[_t['key_name']],
                    'value': _t['value'],
                    'key_group':0
                }
                for _t in temp_feature_tuples
            ]
            self.snorkel_session.execute(Feature.__table__.insert(), feature_tuples)
            return len(feature_tuples)

        features = self.split_cache[split].flatMap(feature_fn)
        key_insert_query = FeatureKey.__table__.insert()
        for cid, feature, value in features.toLocalIterator():
            temp_feature_tuples.append({
                'candidate_id': cid, 'key_name': feature, 'value': value})

            # Periodically flushes the features to disk
            if len(temp_feature_tuples) >= 100000:
                total_features += insert_features(temp_feature_tuples)
                temp_feature_tuples = []

        # Flushes the remaining features
        if len(temp_feature_tuples) > 0:
            total_features += insert_features(temp_feature_tuples)

        LOGGER.info('Generated %d features', total_features)

        # Return feature matrix from the Snorkel DB
        return load_feature_matrix(self.snorkel_session, split=split)

    def _clear_features(self):
        """
        Clears the database of features
        """
        self.snorkel_session.query(Feature).delete(synchronize_session='fetch')

        query = self.snorkel_session.query(FeatureKey).filter(FeatureKey.group == 0)
        query.delete(synchronize_session='fetch')

    def _load_candidates(self, split):
        """
        Loads a set of candidates as a Spark RDD and caches the results as
        self.split_cache[split]

        :param split: the split of candidates to load
        """
        jdbcDF = self.spark_session.read \
            .format("jdbc") \
            .option("url", "jdbc:" + snorkel_conn_string) \
            .option("dbtable",
                self.candidate_class.__tablename__ + "_serialized") \
            .load()

        rdd = jdbcDF.rdd.map(partial(
            wrap_candidate,
            class_name=self.candidate_class.__name__,
            argnames=self.candidate_class.__argnames__
        ))
        rdd = rdd.filter(lambda c: c.split==split)
        rdd = rdd.setName("Snorkel Candidates, Split " + str(split) + \
            " (" + self.candidate_class.__name__ + ")")
        self.split_cache[split] = rdd.cache()
