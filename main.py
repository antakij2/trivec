#! /usr/bin/env python
import os
import numpy as np
from utils import *

def main(K=100, lamb=0.03, learning_rate=0.1, batch_size=6000, dropout=0.2):
    print(f'K: {K}, lambda: {lamb}, learning_rate: {learning_rate}, batch_size: {batch_size}, dropout: {dropout}')
    
    # load data, generate negative triples
    train_triples = load_triples(os.path.join('data', 'ploypharmacy_facts_train.txt'))
    validation_triples = load_triples(os.path.join('data', 'ploypharmacy_facts_valid.txt'))
    test_triples = load_triples(os.path.join('data', 'ploypharmacy_facts_test.txt'))
    negative_train_triples = generate_negative_train_triples(train_triples, validation_triples, test_triples)

    # create full train/test data/labels datasets
    true_train_labels = np.ones(len(train_triples))
    false_train_labels = np.zeros(len(negative_train_triples))
    all_train_labels = np.concatenate((true_train_labels, false_train_labels))
    all_train_triples = np.array(train_triples + negative_train_triples)

    true_test_labels = np.ones(len(test_triples))
    all_test_labels = true_test_labels
    all_test_triples = np.array(test_triples)

    train_ds = tf.data.Dataset.from_tensor_slices((all_train_triples, all_train_labels)).shuffle(len(all_train_triples)).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((all_test_triples, all_test_labels)).batch(batch_size)

    # initialize everything
    model = TriVec(train_triples, K=K, lamb=lamb)
    optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
    train_roc = tf.keras.metrics.AUC(num_thresholds=200, curve='ROC')
    train_pr = tf.keras.metrics.AUC(num_thresholds=200, curve='PR')
    train_ap_50 = 0
    test_roc = tf.keras.metrics.AUC(num_thresholds=200, curve='ROC')
    test_pr = tf.keras.metrics.AUC(num_thresholds=200, curve='PR')
    test_ap_50 = 0

    #@tf.function
    def train_step(triples, labels):
        nonlocal train_ap_50
        
        with tf.GradientTape() as tape:
            raw = model(triples, dropout=dropout)
            loss_value = model.losses

        gradients = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        scores = tf.math.sigmoid(raw)
        train_roc.update_state(y_true=labels, y_pred=scores)
        train_pr.update_state(y_true=labels, y_pred=scores)
        #TODO: train_ap_50 = tf.compat.v1.metrics.average_precision_at_k(labels=labels, predictions=scores, k=50)

    #@tf.function
    def test_step(triples, labels):
        nonlocal test_ap_50

        raw = model(triples)
        scores = tf.math.sigmoid(raw)

        test_roc.update_state(y_true=labels, y_pred=scores)
        test_pr.update_state(y_true=labels, y_pred=scores)
        #TODO: test_ap_50 = tf.compat.v1.metrics.average_precision_at_k(labels=labels, predictions=scores, k=50)

    # run training and testing
    for epoch in range(1000): # paper used a fixed 1000 epochs
        train_roc.reset_states()
        train_pr.reset_states()
        test_roc.reset_states()
        test_pr.reset_states()

        for triple_batch, label_batch in train_ds:
            train_step(triple_batch, label_batch)

        for triple_batch, label_batch in test_ds:
            test_step(triple_batch, label_batch)

        print('Epoch', epoch)
        print('\ttrain AUC-ROC: {0:.4f}'.format(train_roc.result().numpy()), 'train AUC-PR: {0:.4f}'.format(train_pr.result().numpy()))
        print('\ttest AUC-ROC: {0:.4f}'.format(test_roc.result().numpy()), 'test AUC-PR: {0:.4f}'.format(test_pr.result().numpy()))
        print()

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser('Run TriVec on polypharmacy data.')
    parser.add_argument('--K', help='size of embeddings', type=int, required=True)
    parser.add_argument('--lambda', dest='lamb', help='regularization weight', type=float, required=True)
    parser.add_argument('--learning_rate', help='learning rate for optimizer', type=float, required=True)
    parser.add_argument('--batch_size', help='size of each minibatch of training data', type=int, required=True)
    parser.add_argument('--dropout', help='fraction of embedding tensor elements to drop during training', type=float, required=True)

    args = parser.parse_args()
    main(args.K, args.lamb, args.learning_rate, args.batch_size, args.dropout)
