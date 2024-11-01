mod half;
mod iq1m;
mod iq1s;
mod iq2s;
mod iq2xs;
mod iq2xxs;
mod iq3s;
mod iq3xxs;
mod iq4nl;
mod iq4xs;
mod q2_k;
mod q3_k;
mod q4_0;
mod q4_0_4_4;
mod q4_0_4_8;
mod q4_0_8_8;
mod q4_1;
mod q4_k;
mod q5_0;
mod q5_1;
mod q5_k;
mod q6_k;
mod q8_0;
mod q8_1;
mod q8_k;

use std::{cmp::max_by, convert::identity, iter::repeat_with, usize};

pub use ::half::{bf16, f16};
pub use iq1m::IQ1M;
pub use iq1s::IQ1S;
pub use iq2s::IQ2S;
pub use iq2xs::IQ2XS;
pub use iq2xxs::IQ2XXS;
pub use iq3s::IQ3S;
pub use iq3xxs::IQ3XXS;
pub use iq4nl::IQ4NL;
pub use iq4xs::IQ4XS;
pub use q2_k::Q2K;
pub use q3_k::Q3K;
pub use q4_0::Q4_0;
pub use q4_0_4_4::Q4_0_4_4;
pub use q4_0_4_8::Q4_0_4_8;
pub use q4_0_8_8::Q4_0_8_8;
pub use q4_1::Q4_1;
pub use q4_k::Q4K;
pub use q5_0::Q5_0;
pub use q5_1::Q5_1;
pub use q5_k::Q5K;
pub use q6_k::Q6K;
pub use q8_0::Q8_0;
pub use q8_1::Q8_1;
pub use q8_k::Q8K;

#[derive(Clone, Copy, PartialEq, PartialOrd, Debug)]
#[repr(C, align(4))]
struct DeltaMin {
    delta: f16,
    min: f16,
}

impl DeltaMin {
    const ZERO: Self = Self {
        delta: f16::ZERO,
        min: f16::ZERO,
    };

    #[inline]
    fn new(delta: f32, min: f32) -> Self {
        Self {
            delta: f16::from_f32(delta),
            min: f16::from_f32(min),
        }
    }

    #[inline]
    fn no_delta(min: f32) -> Self {
        Self {
            delta: f16::ZERO,
            min: f16::from_f32(min),
        }
    }

    #[inline]
    fn to_f32(self) -> (f32, f32) {
        (self.delta.to_f32(), self.min.to_f32())
    }
}

#[inline]
fn max_abs(data: &[f32]) -> f32 {
    data.iter().fold(0., |acc, x| acc.max(x.abs()))
}

#[inline]
fn max_by_abs(data: &[f32]) -> f32 {
    data.iter()
        .fold(0., |acc, &x| if x.abs() > acc.abs() { x } else { acc })
}

#[inline]
fn min_max(data: &[f32]) -> (f32, f32) {
    data.iter().fold((f32::MAX, f32::MIN), |(min, max), &x| {
        (min.min(x), max.max(x))
    })
}

fn make_qx_quants(x: &[f32], nmax: isize) -> (f32, Vec<usize>) {
    assert!(nmax > 0);
    let max = max_by_abs(x);
    if max.abs() < GROUP_MAX_EPS {
        return (0., std::iter::repeat(0).take(x.len()).collect());
    }
    (-9..=9)
        .into_iter()
        .map(|i| -(nmax as f32) / max + (i as f32) * 0.1)
        .map(|iscale| {
            let suml = x.iter().fold((0., 0.), |(sumlx, suml2), a| {
                let l = (*a * iscale).round() as isize;
                let l = (-nmax).max((nmax - 1).min(l)) as f32;
                let w = a * a;
                (sumlx + w * a * l, suml2 + w * l * l)
            });
            (iscale, suml)
        })
        .max_by(|(_, suml_1), (_, suml_2)| {
            let best = |(sumlx, suml2): &(f32, f32)| {
                if *suml2 != 0.0 {
                    sumlx * sumlx / suml2
                } else {
                    0.0
                }
            };
            best(suml_1).partial_cmp(&best(suml_2)).unwrap()
        })
        .map(|(iscale, (sumlx, suml2))| {
            let L = x.iter().map(|a| {
                let l = (*a * iscale).round() as isize;
                let l = (-nmax).max((nmax - 1).min(l));
                (l + nmax) as usize
            });
            (sumlx / suml2, L.collect())
        })
        .unwrap()
}

#[test]
fn test_make_qx_quants() {
    let a: [f32; 16] = [
        0.6134935558875086,
        0.27100422321951445,
        0.662907299814267,
        0.3012722972026105,
        0.8809210890237902,
        0.9113154578272312,
        0.9586460741733003,
        0.17865102136670108,
        0.5596914646668039,
        0.09094331112669951,
        0.01917780062861074,
        0.5313069088633986,
        0.1885782128334208,
        0.4678985378766791,
        0.060239429412906054,
        0.7827442050642704,
    ];
    let (delta, L) = make_qx_quants(&a, 16);
    println!(" delta: {}", delta);
    let a_after = L
        .iter()
        .map(|&x| (x as isize - 16) as f32 * delta)
        .collect::<Vec<_>>();
    println!(" a_after: {:?}", &a_after);
    println!(
        " error is :{:?}",
        a.iter()
            .zip(a_after.iter())
            .map(|(x, y)| { (x - y) * (x - y) })
            .collect::<Vec<_>>()
    )
}

const _1: usize = 1;
const _32: usize = 32;
const _256: usize = 256;

const GROUP_MAX_EPS: f32 = 1e-15;
