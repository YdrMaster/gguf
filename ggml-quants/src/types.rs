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

const _1: usize = 1;
const _32: usize = 32;
const _256: usize = 256;
