use super::{max_by_abs, _256};
use crate::{DataBlock, Quantize};
use half::f16;
use std::iter::zip;

#[repr(C)]
pub struct Q8K {
    delta: f16,
    quants: [i8; _256],
    sums: [i16; _256 / 16],
}

impl DataBlock for Q8K {
    const COUNT: usize = _256;
    const ZEROS: Self = Self {
        delta: f16::ZERO,
        quants: [0; _256],
        sums: [0; _256 / 16],
    };
}

impl Quantize<f32, _256> for Q8K {
    fn quantize(data: &[f32; _256]) -> Self {
        const { assert!(Self::COUNT == _256) }

        let max = max_by_abs(data);
        if max == 0. {
            return Self::ZEROS;
        }

        let delta = max / -127.;
        let recip = delta.recip();

        let mut quants = [0; _256];
        let mut sums = [0; _256 / 16];
        for (i, (y, &x)) in zip(&mut quants, data).enumerate() {
            *y = ((x * recip).round() as i8).min(127);
            sums[i / 16] += *y as i16;
        }

        Self {
            delta: f16::from_f32(delta),
            quants,
            sums,
        }
    }

    #[inline]
    fn dequantize(&self) -> [f32; _256] {
        let delta = self.delta.to_f32();
        self.quants.map(|x| x as f32 * delta)
    }
}
