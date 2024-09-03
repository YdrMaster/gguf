use super::{QuantBlock, _32};
use half::f16;
use std::iter::zip;

#[repr(C)]
pub struct Q5_0 {
    delta: f16,
    qh: [u8; 4],
    ql: [u8; _32 / 2],
}

impl QuantBlock<f32, _32> for Q5_0 {
    fn quantize(data: &[f32; _32]) -> Self {
        let max = data
            .iter()
            .fold(0.0f32, |max, &x| if x.abs() > max.abs() { x } else { max });

        const D: f32 = (1 << (5 - 1)) as _;
        let delta = max / -D;
        let recip = if max == 0. { 0. } else { delta.recip() };
        let f = |x: f32| ((x * recip + 16.5) as u8).min((1 << 5) - 1);

        let (l, h) = data.split_at(_32 / 2);
        let mut qh = 0;
        let mut ql = [0u8; _32 / 2];
        for (i, (&l, &h)) in zip(l, h).enumerate() {
            let l = f(l);
            let h = f(h);
            qh |= ((l as u32 >> 4) & 1) << i;
            qh |= ((h as u32 >> 4) & 1) << (i + _32 / 2);
            ql[i] = ((h & 0xf) << 4) | (l & 0xf);
        }

        Self {
            delta: f16::from_f32(delta),
            qh: qh.to_le_bytes(),
            ql,
        }
    }

    fn dequantize(&self) -> [f32; _32] {
        let delta = self.delta.to_f32();
        let qh = u32::from_le_bytes(self.qh);
        let f = |l, h| ((l | (h as u8 & 0x10)) - 16) as f32 * delta;

        let mut ans = [0.; _32];
        let (l, h) = ans.split_at_mut(_32 / 2);
        #[rustfmt::skip]
        for (i, x) in self.ql.iter().enumerate() {
            l[i] = f(x & 0xf, (qh >>  i               ) << 4);
            h[i] = f(x >>  4,  qh >> (i + _32 / 2 - 4)      );
        };
        ans
    }
}
