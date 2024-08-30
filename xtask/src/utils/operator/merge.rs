use super::{super::Tensor, blk_tensor_name, Content, DataPromise, BLK_TENSOR_REGEX};
use ggus::{llm_block_count, DataFuture};
use memmap2::MmapMut;
use std::borrow::Cow;

impl Content<'_> {
    pub(super) fn is_linear_merged(&self) -> bool {
        self.tensors.contains_key("blk.0.attn_qkv.weight")
    }

    pub(super) fn merge_linear(&mut self, ty: bool) {
        if self.is_linear_merged() == ty {
            return;
        }

        self.assert_llama();
        let tensors = std::mem::take(&mut self.tensors);
        if ty {
            let blk = self
                .meta_kvs
                .get(&*llm_block_count("llama"))
                .map(|kv| kv.value_reader().read_llm_block_count_val().unwrap())
                .expect("missing block count") as _;

            let mut qkv = MergeCollector::<NUM_QKV>::new(blk);
            let mut gate_up = MergeCollector::<NUM_GATE_UP>::new(blk);

            for (name, tensor) in tensors {
                let Some(captures) = BLK_TENSOR_REGEX.captures(&name) else {
                    self.tensors.insert(name, tensor);
                    continue;
                };
                let i = &captures[1];
                if let Some((name, tensor)) = match &captures[2] {
                    NAME_Q => qkv.put(tensor, i, 0),
                    NAME_K => qkv.put(tensor, i, 1),
                    NAME_V => qkv.put(tensor, i, 2),
                    NAME_GATE => gate_up.put(tensor, i, 0),
                    NAME_UP => gate_up.put(tensor, i, 1),
                    _ => Some((name, tensor)),
                } {
                    self.tensors.insert(name, tensor);
                }
            }
        } else {
            for (name, tensor) in tensors {
                let Some(captures) = BLK_TENSOR_REGEX.captures(&name) else {
                    self.tensors.insert(name, tensor);
                    continue;
                };
                let i = &captures[1];
                match &captures[2] {
                    NAME_QKV => {
                        let [q, k, v] = split_qkv(tensor);
                        self.tensors.insert(blk_tensor_name(i, NAME_Q), q);
                        self.tensors.insert(blk_tensor_name(i, NAME_K), k);
                        self.tensors.insert(blk_tensor_name(i, NAME_V), v);
                    }
                    NAME_GATE_UP => {
                        let [gate, up] = split_gate_up(tensor);
                        self.tensors.insert(blk_tensor_name(i, NAME_GATE), gate);
                        self.tensors.insert(blk_tensor_name(i, NAME_UP), up);
                    }
                    _ => {
                        self.tensors.insert(name, tensor);
                    }
                }
            }
        }
    }
}

const NUM_QKV: usize = 3;
const NUM_GATE_UP: usize = 2;
const NAME_QKV: &str = "attn_qkv";
const NAME_Q: &str = "attn_q";
const NAME_K: &str = "attn_k";
const NAME_V: &str = "attn_v";
const NAME_GATE_UP: &str = "ffn_gate_up";
const NAME_GATE: &str = "ffn_gate";
const NAME_UP: &str = "ffn_up";

struct MergeCollector<'a, const N: usize> {
    buf: Vec<[Option<Tensor<'a>>; N]>,
}

impl<'a, const N: usize> MergeCollector<'a, N> {
    fn new(blk: usize) -> Self {
        Self {
            buf: (0..blk).map(|_| std::array::from_fn(|_| None)).collect(),
        }
    }

    fn collect(&mut self, i: &str, tensor: Tensor<'a>, k: usize) -> Option<[Tensor<'a>; N]> {
        let i: usize = i.parse().unwrap();
        self.buf[i][k] = Some(tensor);
        if self.buf[i].iter().all(Option::is_some) {
            Some(std::array::from_fn(|k| self.buf[i][k].take().unwrap()))
        } else {
            None
        }
    }
}

impl<'a> MergeCollector<'a, NUM_QKV> {
    fn put(&mut self, tensor: Tensor<'a>, i: &str, k: usize) -> Option<(Cow<'a, str>, Tensor<'a>)> {
        self.collect(i, tensor, k).map(|[q, k, v]| {
            let qr = q.shape[1];
            let kr = k.shape[1];
            let vr = v.shape[1];
            assert_eq!(qr % kr, 0);
            assert!(qr >= kr);
            assert_eq!(kr, vr);
            (blk_tensor_name(i, NAME_QKV), concat1([q, k, v]))
        })
    }
}

impl<'a> MergeCollector<'a, NUM_GATE_UP> {
    fn put(&mut self, tensor: Tensor<'a>, i: &str, k: usize) -> Option<(Cow<'a, str>, Tensor<'a>)> {
        self.collect(i, tensor, k).map(|[gate, up]| {
            assert_eq!(gate.shape[1], up.shape[1]);
            (blk_tensor_name(i, NAME_GATE_UP), concat1([gate, up]))
        })
    }
}

fn split_qkv(tensor: Tensor) -> [Tensor; NUM_QKV] {
    let [c, r, _] = distruct(&tensor);
    let rq = c;
    let rkv = (r - c) / 2;
    split1(tensor, [rq, rkv, rkv])
}

fn split_gate_up(tensor: Tensor) -> [Tensor; NUM_GATE_UP] {
    let r = tensor.shape[1] / 2;
    split1(tensor, [r, r])
}

/// 解构形状，补充分布维度
fn distruct(t: &Tensor) -> [u64; 3] {
    match *t.shape {
        [c, r] => [c, r, 1],
        [c, r, n] => [c, r, n],
        [..] => panic!("invalid tensor shape: {:?}", t.shape),
    }
}

/// 构造形状，去除分布维度
fn construct(c: u64, r: u64, n: u64) -> Vec<u64> {
    if n == 1 {
        vec![c, r]
    } else {
        vec![c, r, n]
    }
}

/// 在最高维分割数据
macro_rules! split0 {
    ($s:expr; $d:expr; [$i: expr]) => {
        $s[$d * $i..][..$d]
    };
}

fn concat1<const N: usize>(tensors: [Tensor; N]) -> Tensor {
    // 提取数据类型和形状
    let ty = tensors[0].ty;
    let [c, mut r, n] = distruct(&tensors[0]);
    for t in &tensors[1..] {
        let [c_, r_, n_] = distruct(t);
        assert_eq!(c, c_);
        assert_eq!(n, n_);
        r += r_;
    }
    // 锁定形状和数据
    let r = r;
    let data = tensors.map(|t| t.data);
    // 生成张量
    Tensor {
        ty,
        shape: construct(c, r, n),
        data: DataPromise::lazy(move || {
            let data: [_; N] = std::array::from_fn(|i| data[i].get());

            let len = data.iter().map(|s| s.len()).sum();
            assert_eq!(len, ty.size().elements_to_bytes(&[c, r, n]));

            let n = n as _;
            let mut ans = MmapMut::map_anon(len).unwrap();
            for i in 0..n {
                let mut dst = &mut split0!(ans; len / n; [i]);
                for data in data {
                    let data = &split0!(data; data.len() / n; [i]);
                    let (dst_, tail) = dst.split_at_mut(data.len());
                    dst_.copy_from_slice(data);
                    dst = tail;
                }
                assert!(dst.is_empty());
            }
            ans
        }),
    }
}

fn split1<const N: usize>(tensor: Tensor, split: [u64; N]) -> [Tensor; N] {
    // 提取数据类型和形状
    let ty = tensor.ty;
    let [c, r, n] = distruct(&tensor);
    assert_eq!(r, split.iter().sum());
    // 计算规模
    let size = ty.size();
    let d = size.elements_to_bytes(&[c, r]);
    // 生成张量
    let mut presum = 0;
    split.map(|r_| {
        let d_ = size.elements_to_bytes(&[c, r_]);
        let data = tensor.data.clone();
        let presum_ = presum;
        presum += d_;
        Tensor {
            ty,
            shape: construct(c, r_, n),
            data: DataPromise::lazy(move || {
                let n = n as _;
                let data = data.get();
                assert_eq!(data.len(), d * n);

                let mut ans = MmapMut::map_anon(d_ * n).unwrap();
                for i in 0..n {
                    let src = &split0!(data; d; [i]);
                    let dst = &mut split0!(ans; d_; [i]);
                    dst.copy_from_slice(&src[presum_..][..d_]);
                }
                ans
            }),
        }
    })
}
