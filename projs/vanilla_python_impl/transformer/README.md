TOBEDONE: 

> we share the same weight matrix between the two embedding layers and the pre-softmax linear transformation, similar to [30].

https://stackoverflow.com/questions/70308466/why-are-weight-matrices-shared-between-embedding-layers-in-attention-is-all-you

data_processor 实现每次 load 都会更改一个 batch 内的样本, 而非只是 shuffle batches
不丢掉 batch_size 整除不了的那些样本
add bleu metric
save ckpt 时记下模型参数，便于 load ckpt 时能正确恢复模型的参数而不只是 encoder 和 decoder 对象的恢复
