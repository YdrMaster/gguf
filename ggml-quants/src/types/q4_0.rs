use super::{f16, max_by_abs, _32};
use crate::{DataBlock, Quantize};
use std::array::from_fn;

#[repr(C)]
pub struct Q4_0 {
    delta: f16,
    quants: [u8; _32 / 2],
}

impl DataBlock for Q4_0 {
    const COUNT: usize = _32;
    const ZEROS: Self = Self {
        delta: f16::ZERO,
        quants: [0; _32 / 2],
    };
}

impl Quantize<f32, _32> for Q4_0 {
    fn quantize(data: &[f32; _32]) -> Self {
        const { assert!(Self::COUNT == _32) }

        let max = max_by_abs(data);
        if max == 0. {
            return Self::ZEROS;
        }

        let delta = max / -8.;
        let recip = if max == 0. { 0. } else { delta.recip() };
        let f = |x| ((x * recip + 8.5) as u8).min(15);

        let (l, h) = data.split_at(_32 / 2);
        Self {
            delta: f16::from_f32(delta),
            quants: from_fn(|i| (f(h[i]) << 4) | f(l[i])),
        }
    }

    fn dequantize(&self) -> [f32; _32] {
        let delta = self.delta.to_f32();
        let f = |x| (x - 8) as f32 * delta;

        let mut ans = [0.; _32];
        let (l, h) = ans.split_at_mut(_32 / 2);
        for (i, &x) in self.quants.iter().enumerate() {
            l[i] = f(x & 0xf);
            h[i] = f(x >> 4);
        }
        ans
    }
}
