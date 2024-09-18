use super::{
    super::{DataPromise, MetaValue},
    Content, Operator, BLK_TENSOR_REGEX,
};
use ggus::{DataFuture, GGmlTypeSize, GGufMetaDataValueType, GGufMetaMapExt};
use log::{info, warn};
use memmap2::MmapMut;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::{borrow::Cow, slice::from_raw_parts_mut};

impl Operator {
    #[inline]
    pub fn distribute(n: impl AsRef<str>) -> Self {
        Self::Distribute(n.as_ref().parse().unwrap())
    }
}

const META_TY: GGufMetaDataValueType = GGufMetaDataValueType::U32;
const META_KEY: fn(&str) -> Cow<'static, str> = |arch| format!("{}.distribute", arch).into();
const META_VAL: fn(n: usize) -> Cow<'static, [u8]> = |n| (n as u32).to_le_bytes().to_vec().into();

impl Content<'_> {
    pub(super) fn distribute_meta(&self) -> usize {
        self.meta_kvs
            .get(&META_KEY(self.general_architecture().unwrap()))
            .map_or(1, |val| {
                assert_eq!(val.ty, META_TY);
                val.value_reader().read::<u32>().unwrap() as _
            })
    }

    pub(super) fn distribute(&mut self, n: usize) {
        let current = self.distribute_meta();
        if current == n {
            info!("Model already distributed to {n} parts, skip this step.");
            return;
        }

        if self.is_linear_merged() {
            warn!("Distribute linear-merged model is not supported yet, do split->distribute->merge instead.");
            self.merge_linear(false);
            self.distribute(n);
            self.merge_linear(true);
            return;
        }

        self.assert_llama();
        match n {
            0 => unreachable!("Cannot distribute to 0 parts"),
            1 => self.gather(current),
            _ => self.distribute_(current, n),
        }
    }

    fn distribute_(&mut self, current: usize, target: usize) {
        info!("Distributing model to {target} parts.");
        let m = current as u64;
        let n = target as u64;
        let distruct = move |shape: &[u64]| match *shape {
            [c, r] => {
                assert_eq!(m, 1);
                (c, r)
            }
            [c, r, n] => {
                assert_eq!(m, n);
                (c, r)
            }
            [..] => unreachable!(),
        };

        for (k, v) in &mut self.tensors {
            let Some(captures) = BLK_TENSOR_REGEX.captures(k) else {
                continue;
            };
            match &captures[2] {
                "attn_q" | "attn_k" | "attn_v" | "ffn_up" | "ffn_gate" => {
                    let (c, r) = distruct(&v.shape);
                    assert_eq!(r * m % n, 0);
                    v.shape = vec![c, r * m / n, n];
                }
                "attn_output" | "ffn_down" => {
                    let (c, r) = distruct(&v.shape);
                    assert_eq!(c * m % n, 0);
                    v.shape = vec![c * m / n, r, n];

                    let size = v.ty.size();
                    let data = v.data.clone();
                    v.data = DataPromise::lazy(move || rearrange(data.get(), size, c, r, m, n));
                }
                "attn_qkv" | "ffn_gate_up" => unreachable!(),
                _ => {}
            }
        }

        use indexmap::map::Entry::{Occupied, Vacant};
        match self
            .meta_kvs
            .entry(META_KEY(self.general_architecture().unwrap()))
        {
            Occupied(mut entry) => {
                entry.get_mut().value = META_VAL(n as _);
            }
            Vacant(entry) => {
                entry.insert(MetaValue {
                    ty: META_TY,
                    value: META_VAL(n as _),
                });
            }
        }
    }

    fn gather(&mut self, current: usize) {
        info!("Gathering model to one part.");
        let n = current as u64;
        let distruct = move |shape: &[u64]| {
            let &[c, r, n_] = shape else { unreachable!() };
            assert_eq!(n_, n);
            (c, r)
        };

        for (k, v) in &mut self.tensors {
            let Some(captures) = BLK_TENSOR_REGEX.captures(k) else {
                continue;
            };
            match &captures[2] {
                "attn_q" | "attn_k" | "attn_v" | "ffn_up" | "ffn_gate" => {
                    let (c, r) = distruct(&v.shape);
                    v.shape = vec![c, r * n];
                }
                "attn_output" | "ffn_down" => {
                    let (c, r) = distruct(&v.shape);
                    v.shape = vec![c * n, r];

                    let size = v.ty.size();
                    let data = v.data.clone();
                    v.data = DataPromise::lazy(move || rearrange(data.get(), size, c, r, n, 1));
                }
                "attn_qkv" | "ffn_gate_up" => unreachable!(),
                _ => {}
            }
        }

        self.meta_kvs
            .shift_remove(&META_KEY(self.general_architecture().unwrap()));
    }
}

fn rearrange(data: &[u8], size: GGmlTypeSize, c: u64, r: u64, m: u64, n: u64) -> MmapMut {
    let len = size.elements_to_bytes(&[c, r, m]);
    debug_assert_eq!(len, data.len());

    let r = r as usize;
    let d = gcd(c, c * m / n); // 元素/段
    let n_src = (c / d) as usize; // 当前段/卡
    let n_dst = (c * m / n / d) as usize; // 目标段/卡
    let n = (c * m / d) as usize; // 段
    let d = size.elements_to_bytes(&[d]); // 字节/段

    let mut ans = MmapMut::map_anon(data.len()).unwrap();
    let dst = ans.as_mut_ptr() as usize;
    (0..n * r).into_par_iter().for_each(|i| {
        let j = i % r;
        let i = i / r;
        let c_src = i / n_src; // 当前卡号
        let d_src = i % n_src; // 当前段号
        let c_dst = i / n_dst; // 目标卡号
        let d_dst = i % n_dst; // 目标段号
        let i = (c_src * r + j) * n_src + d_src;
        let j = (c_dst * r + j) * n_dst + d_dst;

        let src = &data[d * i..][..d];
        let dst = unsafe { from_raw_parts_mut((dst + d * j) as *mut u8, d) };
        dst.copy_from_slice(src);
    });
    ans
}

#[inline]
const fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let rem = a % b;
        a = b;
        b = rem;
    }
    a
}
