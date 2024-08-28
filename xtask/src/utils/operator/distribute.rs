use super::{
    super::{DataPromise, MetaValue},
    Content, Operator, BLK_TENSOR_REGEX,
};
use ggus::{DataFuture, GGmlTypeSize, GGufMetaDataValueType};
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
        self.meta_kvs.get(&META_KEY(self.arch())).map_or(1, |val| {
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
                "attn_qkv" => todo!(),
                "ffn_gate_up" => todo!(),
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
                    v.data = DataPromise::lazy(move || {
                        rearrange(data.get(), size, (c, r, m), (c * m / n, r, n))
                    });
                }
                _ => {}
            }
        }

        use indexmap::map::Entry::{Occupied, Vacant};
        match self.meta_kvs.entry(META_KEY(self.arch())) {
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
                "attn_qkv" => todo!(),
                "ffn_gate_up" => todo!(),
                "attn_q" | "attn_k" | "attn_v" | "ffn_up" | "ffn_gate" => {
                    let (c, r) = distruct(&v.shape);
                    v.shape = vec![c, r * n];
                }
                "attn_output" | "ffn_down" => {
                    let (c, r) = distruct(&v.shape);
                    v.shape = vec![c * n, r];

                    let size = v.ty.size();
                    let data = v.data.clone();
                    v.data = DataPromise::lazy(move || {
                        rearrange(data.get(), size, (c, r, n), (c * n, r, 1))
                    });
                }
                _ => {}
            }
        }

        self.meta_kvs.shift_remove(&META_KEY(self.arch()));
    }
}

fn rearrange(
    data: &[u8],
    size: GGmlTypeSize,
    shape: (u64, u64, u64),
    target: (u64, u64, u64),
) -> MmapMut {
    let (c, r, m) = shape;
    let (c_, r_, n) = target;
    let len = size.elements_to_bytes(&[c, r, m]);

    debug_assert_eq!(r, r_);
    debug_assert_eq!(c * m, c_ * n);
    debug_assert_eq!(len, data.len());

    let r = r as usize;
    let d = gcd(c, c_);
    let md = (c / d) as usize;
    let nd = (c_ / d) as usize;
    let d = size.elements_to_bytes(&[d]);
    let m = m as usize;
    let n = n as usize;
    // d md r m -> d nd r n

    let mut ans = MmapMut::map_anon(data.len()).unwrap();
    let dst = ans.as_mut_ptr() as usize;
    (0..len / d).into_par_iter().for_each(|i| {
        let j = i / (md * m);
        let i = i % (md * m);
        let a = i / m * (md * r) + j * md + i % m;
        let b = i / n * (nd * r) + j * nd + i % n;

        let src = &data[d * a..][..d];
        let dst = unsafe { from_raw_parts_mut((dst + d * b) as *mut u8, d) };
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
