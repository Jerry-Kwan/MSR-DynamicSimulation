import os
import random
import argparse
import matplotlib.pyplot as plt

import numpy as np
try:
    import cupy as cp
    use_cupy = True
    print('---Using Cupy---', end='\n\n')
except ModuleNotFoundError:
    use_cupy = False
    print('---Using Numpy---', end='\n\n')

from my_transformer import Transformer, Adam, Noam, CrossEntropyLoss
from data_processor import DataProcessor


def get_args():
    parser = argparse.ArgumentParser(description='My Transformer')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')

    # model params
    parser.add_argument('--src-max-len',
                        type=int,
                        default=5000,
                        help='src max seq_len (cannot use max_len in dataset, because pred_input_len may be longer)')
    parser.add_argument('--tgt-max-len',
                        type=int,
                        default=5000,
                        help='tgt max seq_len (cannot use max_len in dataset, because pred_output_len may be longer)')
    parser.add_argument('--num-enc-heads', type=int, default=8, help='the number of enc heads')
    parser.add_argument('--num-dec-heads', type=int, default=8, help='the number of dec heads')
    parser.add_argument('--num-enc-layers', type=int, default=6, help='the number of enc layers')
    parser.add_argument('--num-dec-layers', type=int, default=6, help='the number of dec layers')
    parser.add_argument('--d-model', type=int, default=512, help='d_model')
    parser.add_argument('--d-ff', type=int, default=2048, help='d_ff')
    parser.add_argument('--enc-dropout', default=0.1, type=float, help='enc dropout rate')
    parser.add_argument('--dec-dropout', default=0.1, type=float, help='dec dropout rate')
    parser.add_argument('--data-type', default='float32', type=str, choices=['float32'], help='data type')

    # ckpt params
    parser.add_argument('--load-ckpt', type=bool, default=False, help='whether loading model from ckpt (load the best)')
    parser.add_argument('--save-ckpt', type=bool, default=False, help='whether saving ckpt')
    parser.add_argument('--save-best-ckpt', type=bool, default=False, help='whether saving best ckpt')
    parser.add_argument('--best-ckpt-suffix', default='best', type=str, choices=['best'], help='best ckpt suffix')
    parser.add_argument('--save-ckpt-every-epochs', type=int, default=3, help='save ckpt every epochs')
    parser.add_argument('--ckpt-root',
                        type=str,
                        default='../../../data/vanilla_python_impl/transformer/',
                        help='ckpt root directory')

    # optimizer params
    parser.add_argument('--adam-alpha', default=1e-4, type=float, help='adam alpha')
    parser.add_argument('--adam-beta1', default=0.9, type=float, help='adam beta1')
    parser.add_argument('--adam-beta2', default=0.98, type=float, help='adam beta2')
    parser.add_argument('--adam-epsilon', default=1e-9, type=float, help='adam epsilon')
    parser.add_argument('--noam-scale-factor', default=2.0, type=float, help='noam scale factor')
    parser.add_argument('--noam-warmup-steps', type=int, default=4000, help='noam warmup steps')

    # training params
    parser.add_argument('--train', type=bool, default=True, help='whether training model')
    parser.add_argument('--num-epoch', type=int, default=30, help='the number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size')
    parser.add_argument('--save-root', default='./', type=str, help='save root directory for graph')
    parser.add_argument('--log-every-batches', type=int, default=10, help='log every batches')

    # dataset params
    parser.add_argument('--test-ratio', default=0.1, type=float, help='test ratio')
    parser.add_argument('--min-freq', type=int, default=2, help='min freq of vocab')
    parser.add_argument('--num-examples',
                        type=int,
                        default=None,
                        help='number of examples used in dataset (default all)')
    parser.add_argument('--dataset-root',
                        default='../../../data/vanilla_python_impl/transformer/',
                        type=str,
                        help='dataset root directory')

    # other params
    parser.add_argument('--plot-attn', type=bool, default=True, help='whether plotting attn')
    parser.add_argument('--num-attn-rows', type=int, default=2, help='the number of rows of attn graph')
    parser.add_argument('--num-attn-cols', type=int, default=4, help='the number of cols of attn graph')

    return parser.parse_args()


def seed_everything(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    if use_cupy:
        cp.random.seed(args.seed)


def get_data_type(args):
    # currently, only use float32
    return cp.float32 if use_cupy else np.float32


def create_model_with_optim(args, dp, data_type):
    """Create Transformer with Noam(Adam)."""
    model = Transformer(src_vocab_size=len(dp.src_vocab),
                        tgt_vocab_size=len(dp.tgt_vocab),
                        src_max_len=args.src_max_len,
                        tgt_max_len=args.tgt_max_len,
                        num_enc_heads=args.num_enc_heads,
                        num_dec_heads=args.num_dec_heads,
                        num_enc_layers=args.num_enc_layers,
                        num_dec_layers=args.num_dec_layers,
                        d_model=args.d_model,
                        d_ff=args.d_ff,
                        enc_dropout=args.enc_dropout,
                        dec_dropout=args.dec_dropout,
                        data_type=data_type)
    optim = Noam(optimizer=Adam(alpha=args.adam_alpha,
                                beta1=args.adam_beta1,
                                beta2=args.adam_beta2,
                                epsilon=args.adam_epsilon),
                 d_model=args.d_model,
                 scale_factor=args.noam_scale_factor,
                 warmup_steps=args.noam_warmup_steps)

    model.set_optimizer(optim)
    return model


def train(args, dp, model, criterion):
    print('------training------', end='\n\n')

    train_epoch_loss, test_epoch_loss = [], []
    best_test_loss = float('inf')

    for epoch in range(args.num_epoch):
        train_epoch_loss.append(train_one_epoch(args, dp, model, criterion, epoch))
        test_epoch_loss.append(test_one_epoch(args, dp, model, criterion, epoch))
        if args.save_ckpt and (epoch + 1) % args.save_ckpt_every_epochs == 0:
            model.save_model(args.ckpt_root, epoch + 1)

        if args.save_best_ckpt and test_epoch_loss[-1] < best_test_loss:
            best_test_loss = test_epoch_loss[-1]
            model.save_model(args.ckpt_root, args.best_ckpt_suffix)

    print('--------------------', end='\n\n')
    return train_epoch_loss, test_epoch_loss


def train_one_epoch(args, dp, model, criterion, epoch):
    sum_loss = 0
    num_inputs = 0
    print(f'[Epoch {epoch + 1}/{args.num_epoch}] training...')

    for batch_num, data in enumerate(dp.train_iter()):
        s, t = data
        src, src_mask = s
        tgt, tgt_mask = t

        # used for calculationg loss per seq
        num_inputs += src.shape[0]

        if use_cupy:
            # data_type transformed is done by model
            src, tgt = cp.asarray(src), cp.asarray(tgt)
            src_mask, tgt_mask = cp.asarray(src_mask), cp.asarray(tgt_mask)

        # forward
        out, _ = model.forward(src, tgt[:, :-1], src_mask, tgt_mask[:, :-1, :-1], training=True)
        _out = out.reshape(out.shape[0] * out.shape[1], out.shape[2])

        # loss
        loss = criterion.forward(_out, tgt[:, 1:].reshape(-1)).flatten().sum()
        if use_cupy:
            loss = cp.asnumpy(loss)

        sum_loss += loss

        # backward and update
        grad = criterion.backward()
        model.backward(grad.reshape(out.shape))
        model.update_weights()

        # log per batches
        if batch_num % args.log_every_batches == 0:
            print(f'[Epoch {epoch + 1}/{args.num_epoch}] '
                  f'[Batch {batch_num + 1}/{dp.num_train_batches}] avg_loss_per_seq = {sum_loss / num_inputs}')

    print(f'[Epoch {epoch + 1}/{args.num_epoch}] end training, avg_loss_per_seq = {sum_loss / num_inputs}')
    return sum_loss / num_inputs


def test_one_epoch(args, dp, model, criterion, epoch):
    sum_loss = 0
    num_inputs = 0
    print(f'[Epoch {epoch + 1}/{args.num_epoch}] testing...')

    for batch_num, data in enumerate(dp.test_iter()):
        s, t = data
        src, src_mask = s
        tgt, tgt_mask = t

        # used for calculationg loss per seq
        num_inputs += src.shape[0]

        if use_cupy:
            # data_type transformed is done by model
            src, tgt = cp.asarray(src), cp.asarray(tgt)
            src_mask, tgt_mask = cp.asarray(src_mask), cp.asarray(tgt_mask)

        # forward
        out, _ = model.forward(src, tgt[:, :-1], src_mask, tgt_mask[:, :-1, :-1], training=False)
        _out = out.reshape(out.shape[0] * out.shape[1], out.shape[2])

        # loss
        loss = criterion.forward(_out, tgt[:, 1:].reshape(-1)).flatten().sum()
        if use_cupy:
            loss = cp.asnumpy(loss)

        sum_loss += loss

    print(f'[Epoch {epoch + 1}/{args.num_epoch}] end testing, avg_loss_per_seq = {sum_loss / num_inputs}\n')
    return sum_loss / num_inputs


def plot_graph(train_epoch_loss, test_epoch_loss):
    """Draw curve."""
    x = np.arange(len(train_epoch_loss)) + 1

    plt.subplot(121)
    plt.plot(x, train_epoch_loss, 'bo-')
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Per Sentence')

    plt.subplot(122)
    plt.plot(x, test_epoch_loss, 'ro-')
    plt.title('Testing Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Per Sencence')

    plt.suptitle('Transformer eng-fra Curve')
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_root, 'curve.png'))


def plot_attn(args, dp, src, tgt, attn, idx):
    """Plot multi-head attention of enc-dec-attn."""
    assert args.num_attn_rows * args.num_attn_cols == args.num_dec_heads, 'error: r * c != num_dec_heads'

    # src should contain bos and eos, tgt should contain bos
    src = [dp.BOS_TOKEN] + [word.lower() for word in src] + [dp.EOS_TOKEN]
    tgt = [dp.BOS_TOKEN] + tgt[:-1]

    fig = plt.figure()

    for h in range(args.num_dec_heads):
        ax = fig.add_subplot(args.num_attn_rows, args.num_attn_cols, h + 1)
        ax.set_xlabel(f'Key\nHead {h + 1}')
        ax.set_ylabel('Query')
        if use_cupy:
            ax.matshow(cp.asnumpy(attn[h]), cmap='inferno')
        else:
            ax.matshow(attn[h], cmap='inferno')

        ax.tick_params(labelsize=7)

        ax.set_xticks(range(len(src)))
        ax.set_yticks(range(len(tgt)))

        ax.set_xticklabels(src, rotation=90)
        ax.set_yticklabels(tgt)

    plt.tight_layout()
    plt.savefig(os.path.join(args.save_root, f'head_attn_{idx}.png'))


def pred_and_plot_attn(args, dp, model):
    """Predict and plot attention."""
    EXAMPLES = [
        ['i', 'love', 'study', '.'],
        ['the', 'whether', 'is', 'good', 'today', '.']
    ]

    for i, example in enumerate(EXAMPLES):
        src = dp.src_vocab[example]
        src = [dp.BOS_INDEX] + src + [dp.EOS_INDEX]

        src = np.array(src).reshape(1, -1)
        src_mask = dp.get_pad_mask([src])[0]

        enc_src = model.encoder.forward(src, src_mask, training=False)

        tgt_indices = [dp.BOS_INDEX]

        for _ in range(args.tgt_max_len):
            tgt = np.array(tgt_indices).reshape(1, -1)
            tgt_mask = dp.get_pad_mask([tgt])[0] & dp.get_sub_mask([tgt])[0]

            out, attn = model.decoder.forward(tgt, tgt_mask, enc_src, src_mask, training=False)
            tgt_index = out.argmax(axis=-1)[:, -1].item()
            tgt_indices.append(tgt_index)

            if tgt_index == dp.EOS_INDEX or len(tgt_indices) >= args.tgt_max_len:
                break

        decoded_seq = dp.tgt_vocab.to_tokens(tgt_indices)[1:]
        print(f"Input: {' '.join(example)}\nOutput: {' '.join(decoded_seq)}")
        plot_attn(args, dp, example, decoded_seq, attn[0], i)


if __name__ == '__main__':
    args = get_args()
    seed_everything(args)
    data_type = get_data_type(args)

    # dp
    dp = DataProcessor()
    dp.prepare_data(path=args.dataset_root,
                    batch_size=args.batch_size,
                    test_ratio=args.test_ratio,
                    min_freq=args.min_freq,
                    num_examples=args.num_examples)

    # model
    model = create_model_with_optim(args, dp, data_type)
    if args.load_ckpt:
        # currently, only support load the best ckpt
        model.load_model(args.ckpt_root, args.best_ckpt_suffix)

    # train
    if args.train:
        criterion = CrossEntropyLoss(ignore_index=dp.PAD_INDEX)
        train_epoch_loss, test_epoch_loss = train(args, dp, model, criterion)
        plot_graph(train_epoch_loss, test_epoch_loss)

    # plot attn
    if args.plot_attn:
        pred_and_plot_attn(args, dp, model)
