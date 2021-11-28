import random
import itertools
import tensorflow as tf
from typing import List, Tuple, Dict

tf.random.set_seed(1234)
random.seed(1234)

'''
For each true training triple (s, p, o), generate a false triple by replacing o with any other entity
in the dataset, as long as the resulting triple (s, p, o') is not a true triple in the training, 
validation, or testing sets. 
'''
def generate_negative_train_triples(train_triples: List[List[bytes]],
                                    validation_triples: List[List[bytes]],
                                    test_triples: List[List[bytes]]) -> List[List[bytes]]:
    negative_triples = []
    all_triples = set()
    all_entities = set()
    for triple in itertools.chain(train_triples, validation_triples, test_triples):
        all_triples.add(tuple(triple))
        all_entities.add(triple[0])
        all_entities.add(triple[2])

    all_entities = list(all_entities)
    for subject, predicate, object_ in train_triples:
        new_object = object_
        while (subject, predicate, new_object) in all_entities:
            new_object = random.choice(all_entities)

        negative_triples.append([subject, predicate, new_object])

    return negative_triples

def load_triples(path: str) -> List[List[bytes]]:
    triples = []
    with open(path, encoding='utf-8') as file:
        for line in file:
            splits = line.split()
            triples.append([split.encode('utf-8') for split in splits])
    return triples

class TriVec(tf.keras.Model):
    def __init__(self, all_train_triples: List[List[bytes]], K: int, lamb: float):
        super(TriVec, self).__init__()
        
        # map entity/relation codenames to embeddings indices, and vice versa
        name_to_index = {}
        index_to_entity_name = {}
        index_to_relation_name = {}

        entities = set()
        relations = set()
        for entity1, relation, entity2 in all_train_triples:
            entities.add(entity1)
            entities.add(entity2)
            relations.add(relation)

        for i, entity in enumerate(entities):
            name_to_index[entity] = i
            index_to_entity_name[i] = entity
        for i, relation in enumerate(relations):
            name_to_index[relation] = i
            index_to_relation_name[i] = relation

        # construct embeddings layer
        self.embeddings = Embeddings(name_to_index, index_to_entity_name, index_to_relation_name, K, lamb)

    def call(self, triples: List[List[bytes]], dropout: tf.Tensor = tf.constant(0.0)) -> tf.Tensor:
        return self.embeddings(triples, dropout)

class Embeddings(tf.keras.layers.Layer):
    def __init__(self, name_to_index: Dict[bytes, int], index_to_entity_name: Dict[int, bytes], index_to_relation_name: Dict[int, bytes], K: int, lamb: float):
        super(Embeddings, self).__init__()
        self.embeddings = self.add_weight(shape=(len(name_to_index), 3, K),
                                          initializer=tf.keras.initializers.GlorotUniform(),
                                          trainable=True)
        self.name_to_index = name_to_index
        self.index_to_entity_name = index_to_entity_name
        self.index_to_relation_name = index_to_relation_name
        self.lamb = lamb

    def calculate_score(self, subject: tf.Tensor, predicate: tf.Tensor, object_: tf.Tensor) -> tf.Tensor:
        return tf.reduce_sum((subject[0] * predicate[0] * object_[2])
                             + (subject[1] * predicate[1] * object_[1])
                             + (subject[2] * predicate[2] * object_[0]))

    def get_score_and_loss(self, triples: List[List[bytes]], dropout:tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        scores = []
        total_loss = 0
        for subject_name, predicate_name, object_name in triples:
            print('Embeddings.get_score_and_loss: start loop')
            subject_name = subject_name.numpy()
            predicate_name = predicate_name.numpy()
            object_name = object_name.numpy()

            subject_index = self.name_to_index[subject_name]
            object_index = self.name_to_index[object_name]

            subject_tensor = tf.nn.dropout(self.embeddings[subject_index], rate=dropout)
            predicate_tensor = tf.nn.dropout(self.embeddings[self.name_to_index[predicate_name]], rate=dropout)
            object_tensor = tf.nn.dropout(self.embeddings[object_index], rate=dropout)
            score = self.calculate_score(subject_tensor, predicate_tensor, object_tensor)
            scores.append(score)
            
            # compute loss
            total_loss += -score + tf.math.log(
                                    tf.math.reduce_sum(
                                        tf.math.exp(
                                            tf.convert_to_tensor(
                                                [self.calculate_score(subject_tensor, predicate_tensor, self.embeddings[new_object_index])
                                                 for new_object_index in self.index_to_entity_name if
                                                 new_object_index != object_index], dtype=float
                                            )
                                        )
                                    )
                                ) \
                          -score + tf.math.log(
                                    tf.math.reduce_sum(
                                        tf.math.exp(
                                            tf.convert_to_tensor(
                                                [self.calculate_score(self.embeddings[new_subject_index], predicate_tensor, object_tensor)
                                                 for new_subject_index in self.index_to_entity_name if
                                                 new_subject_index != subject_index], dtype=float
                                            )
                                        )
                                    )
                                ) \
                          + (self.lamb / 3) * tf.math.reduce_sum(tf.math.pow(tf.math.abs(subject_tensor), 3)
                                                                 + tf.math.pow(tf.math.abs(predicate_tensor), 3)
                                                                 + tf.math.pow(tf.math.abs(object_tensor), 3))

        return tf.stack(scores), tf.convert_to_tensor(total_loss, dtype=float)

    def call(self, triples: List[List[bytes]], dropout: tf.Tensor) -> tf.Tensor:
        scores, total_loss = self.get_score_and_loss(triples, dropout)
        self.add_loss(total_loss)
        return scores
