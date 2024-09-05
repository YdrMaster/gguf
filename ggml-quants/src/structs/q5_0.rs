use super::{f16, max_by_abs, _32};
use crate::{DataBlock, Quantize};
use std::iter::zip;

#[repr(C)]
pub struct Q5_0 {
    delta: f16,
    qh: [u8; _32 / 8],
    ql: [u8; _32 / 2],
}

impl DataBlock for Q5_0 {
    const COUNT: usize = _32;
    const ZEROS: Self = Self {
        delta: f16::ZERO,
        qh: [0; _32 / 8],
        ql: [0; _32 / 2],
    };
}

impl Quantize<f32, _32> for Q5_0 {
    fn quantize(data: &[f32; _32]) -> Self {
        const { assert!(Self::COUNT == _32) }

        let max = max_by_abs(data);
        if max == 0. {
            return Self::ZEROS;
        }

        let delta = max / -16.;
        let recip = delta.recip();
        let f = |x: f32| ((x * recip + 16.5) as u8).min(31);

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
