use super::{super::Tensor, Content, DataPromise, Operator};
use ggus::{llm_block_count, DataFuture};
use indexmap::IndexMap;
use memmap2::MmapMut;
use regex::Regex;
use std::{borrow::Cow, sync::LazyLock};

impl Operator {
    pub fn merge_linear(m: impl AsRef<str>) -> Self {
        let m = m.as_ref().trim();
        Self::MergeLinear(match m.to_lowercase().as_str() {
            "yes" | "y" | "true" | "t" => true,
            "no" | "n" | "false" | "f" => false,
            _ => panic!("unsupported transpose type: {m}"),
        })
    }
}

impl Content<'_> {
    pub(super) fn merge_linear(&mut self, ty: bool) {
        self.assert_llama();

        if self.tensors.contains_key("blk.0.attn_qkv.weight") == ty {
            assert_eq!(self.tensors.contains_key("blk.0.ffn_gate_up.weight"), ty);
            return;
        }

        if ty {
            let blk = self
                .meta_kvs
                .get(&*llm_block_count("llama"))
                .map(|kv| kv.value_reader().read_llm_block_count_val().unwrap())
                .expect("missing block count");

            self.merge(blk as _);
        } else {
            self.split();
        }
    }

    fn merge(&mut self, blk: usize) {
        let mut qkv = MergeCollector::<3>::new(blk);
        let mut gate_up = MergeCollector::<2>::new(blk);

        let tensors = std::mem::take(&mut self.tensors);
        for (name, tensor) in tensors {
            let Some(captures) = REGEX.captures(&name) else {
                self.tensors.insert(name, tensor);
                continue;
            };
            match &captures[2] {
                "attn_q" => qkv.put(&mut self.tensors, &captures[1], tensor, 0),
                "attn_k" => qkv.put(&mut self.tensors, &captures[1], tensor, 1),
                "attn_v" => qkv.put(&mut self.tensors, &captures[1], tensor, 2),
                "ffn_gate" => gate_up.put(&mut self.tensors, &captures[1], tensor, 0),
                "ffn_up" => gate_up.put(&mut self.tensors, &captures[1], tensor, 1),
                _ => {
                    self.tensors.insert(name, tensor);
                }
            }
        }
    }

    fn split(&mut self) {
        let tensors = std::mem::take(&mut self.tensors);
        for (name, tensor) in tensors {
            let Some(captures) = REGEX.captures(&name) else {
                self.tensors.insert(name, tensor);
                continue;
            };
            match &captures[2] {
                "attn_qkv" => {
                    let i: usize = captures[1].parse().unwrap();
                    let (q, k, v) = split_qkv(&tensor);
                    self.tensors
                        .insert(format!("blk.{i}.attn_q.weight").into(), q);
                    self.tensors
                        .insert(format!("blk.{i}.attn_k.weight").into(), k);
                    self.tensors
                        .insert(format!("blk.{i}.attn_v.weight").into(), v);
                }
                "ffn_gate_up" => {
                    let i: usize = captures[1].parse().unwrap();
                    let (gate, up) = split_gate_up(&tensor);
                    self.tensors
                        .insert(format!("blk.{i}.ffn_gate.weight").into(), gate);
                    self.tensors
                        .insert(format!("blk.{i}.ffn_up.weight").into(), up);
                }
                _ => {
                    self.tensors.insert(name, tensor);
                }
            }
        }
    }
}

static REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^blk\.(\d+)\.(\w+)\.weight$").unwrap());

struct Blk<'a, const N: usize>([Option<Tensor<'a>>; N]);

struct MergeCollector<'a, const N: usize> {
    buf: Vec<Blk<'a, N>>,
}

impl<'a> MergeCollector<'a, 3> {
    fn new(blk: usize) -> Self {
        Self {
            buf: (0..blk).map(|_| Blk([None, None, None])).collect(),
        }
    }

    fn put(
        &mut self,
        map: &mut IndexMap<Cow<'a, str>, Tensor<'a>>,
        i: impl AsRef<str>,
        tensor: Tensor<'a>,
        n: usize,
    ) {
        let i: usize = i.as_ref().parse().unwrap();
        self.buf[i].0[n] = Some(tensor);

        let qkv = &mut self.buf[i];
        if qkv.0.iter().any(Option::is_none) {
            return;
        }

        let q = qkv.0[0].take().unwrap();
        let k = qkv.0[1].take().unwrap();
        let v = qkv.0[2].take().unwrap();

        let &[c, qr] = &*q.shape else {
            panic!("invalid q shape: {:?}", q.shape);
        };
        let &[kc, kr] = &*k.shape else {
            panic!("invalid k shape: {:?}", k.shape);
        };
        let &[vc, vr] = &*v.shape else {
            panic!("invalid v shape: {:?}", v.shape);
        };
        assert_eq!(qr % kr, 0);
        assert!(qr >= kr);
        assert_eq!(kr, vr);
        assert_eq!(kc, c);
        assert_eq!(vc, c);

        let ty = q.ty;
        assert_eq!(k.ty, ty);
        assert_eq!(v.ty, ty);

        let r = qr + kr + vr;
        let q = q.data;
        let k = k.data;
        let v = v.data;
        map.insert(
            format!("blk.{i}.attn_qkv.weight").into(),
            Tensor {
                ty,
                shape: vec![c, r],
                data: DataPromise::lazy(move || {
                    let q = q.get();
                    let k = k.get();
                    let v = v.get();

                    let len = q.len() + k.len() + v.len();
                    assert_eq!(len, ty.size().elements_to_bytes(&[c, r]));

                    let mut dst = MmapMut::map_anon(len).unwrap();
                    let (q_, tail) = dst.split_at_mut(q.len());
                    let (k_, v_) = tail.split_at_mut(k.len());
                    q_.copy_from_slice(q);
                    k_.copy_from_slice(k);
                    v_.copy_from_slice(v);
                    dst
                }),
            },
        );
    }
}

impl<'a> MergeCollector<'a, 2> {
    fn new(blk: usize) -> Self {
        Self {
            buf: (0..blk).map(|_| Blk([None, None])).collect(),
        }
    }

    fn put(
        &mut self,
        map: &mut IndexMap<Cow<'a, str>, Tensor<'a>>,
        i: impl AsRef<str>,
        tensor: Tensor<'a>,
        n: usize,
    ) {
        let i: usize = i.as_ref().parse().unwrap();
        self.buf[i].0[n] = Some(tensor);

        let gate_up = &mut self.buf[i];
        if gate_up.0.iter().any(Option::is_none) {
            return;
        }

        let gate = gate_up.0[0].take().unwrap();
        let up = gate_up.0[1].take().unwrap();

        let &[c, r] = &*gate.shape else {
            panic!("invalid gate shape: {:?}", gate.shape);
        };
        let &[c_, r_] = &*up.shape else {
            panic!("invalid up shape: {:?}", up.shape);
        };
        assert_eq!(r, r_);
        assert_eq!(c, c_);

        let ty = gate.ty;
        assert_eq!(up.ty, ty);

        let r = r * 2;
        let gate = gate.data;
        let up = up.data;
        map.insert(
            format!("blk.{i}.ffn_gate_up.weight").into(),
            Tensor {
                ty,
                shape: vec![c, r],
                data: DataPromise::lazy(move || {
                    let gate = gate.get();
                    let up = up.get();

                    let len = gate.len() + up.len();
                    assert_eq!(len, ty.size().elements_to_bytes(&[c, r]));

                    let mut dst = MmapMut::map_anon(len).unwrap();
                    let (gate_, up_) = dst.split_at_mut(gate.len());
                    gate_.copy_from_slice(gate);
                    up_.copy_from_slice(up);
                    dst
                }),
            },
        );
    }
}

fn split_qkv<'a>(tensor: &Tensor<'a>) -> (Tensor<'a>, Tensor<'a>, Tensor<'a>) {
    let &[c, r] = &*tensor.shape else {
        panic!("invalid tensor shape: {:?}", tensor.shape);
    };

    let ty = tensor.ty;
    let rq = c;
    let rkv = (r - c) / 2;
    let size = ty.size();
    let dq = size.elements_to_bytes(&[c, rq]);
    let dkv = size.elements_to_bytes(&[c, rkv]);
    let q = {
        let data = tensor.data.clone();
        Tensor {
            ty,
            shape: vec![c, rq],
            data: DataPromise::lazy(move || copy_to_mmap(&data.get()[..dq])),
        }
    };
    let k = {
        let data = tensor.data.clone();
        Tensor {
            ty,
            shape: vec![c, rkv],
            data: DataPromise::lazy(move || copy_to_mmap(&data.get()[dq..][..dkv])),
        }
    };
    let v = {
        let data = tensor.data.clone();
        Tensor {
            ty,
            shape: vec![c, rkv],
            data: DataPromise::lazy(move || copy_to_mmap(&data.get()[dq + dkv..])),
        }
    };
    (q, k, v)
}

fn split_gate_up<'a>(tensor: &Tensor<'a>) -> (Tensor<'a>, Tensor<'a>) {
    let &[c, r] = &*tensor.shape else {
        panic!("invalid tensor shape: {:?}", tensor.shape);
    };

    let ty = tensor.ty;
    let r = r / 2;
    let d = ty.size().elements_to_bytes(&[c, r]);
    let gate = {
        let data = tensor.data.clone();
        Tensor {
            ty,
            shape: vec![c, r],
            data: DataPromise::lazy(move || copy_to_mmap(&data.get()[..d])),
        }
    };
    let up = {
        let data = tensor.data.clone();
        Tensor {
            ty,
            shape: vec![c, r],
            data: DataPromise::lazy(move || copy_to_mmap(&data.get()[d..])),
        }
    };
    (gate, up)
}

fn copy_to_mmap(data: &[u8]) -> MmapMut {
    let mut dst = MmapMut::map_anon(data.len()).unwrap();
    dst.copy_from_slice(data);
    dst
}
