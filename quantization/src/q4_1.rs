use super::{QuantBlock, _32};
use half::f16;
use std::array::from_fn;

#[repr(C)]
pub struct Q4_1 {
    delta: f16,
    min: f16,
    quants: [u8; _32 / 2],
}

impl QuantBlock<f32, _32> for Q4_1 {
    fn quantize(data: &[f32; _32]) -> Self {
        let (min, max) = data.iter().fold((f32::MAX, f32::MIN), |(min, max), &x| {
            (min.min(x), max.max(x))
        });

        const D: f32 = ((1 << 4) - 1) as _;
        let delta = (max - min) / D;
        let recip = if delta == 0. { 0. } else { delta.recip() };
        let f = |x| (((x - min) * recip + 0.5) as u8).min((1 << 4) - 1);

        let (l, h) = data.split_at(_32 / 2);
        Self {
            delta: f16::from_f32(delta),
            min: f16::from_f32(min),
            quants: from_fn(|i| (f(h[i]) << 4) | f(l[i])),
        }
    }

    fn dequantize(&self) -> [f32; _32] {
        let delta = self.delta.to_f32();
        let min = self.min.to_f32();
        let f = |x| x as f32 * delta + min;

        let mut ans = [0.; _32];
        let (l, h) = ans.split_at_mut(_32 / 2);
        for (i, &x) in self.quants.iter().enumerate() {
            l[i] = f(x & 0xf);
            h[i] = f(x >> 4);
        }
        ans
    }
}
