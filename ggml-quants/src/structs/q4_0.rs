use super::{f16, max_by_abs, _32};
use crate::{DataBlock, Quantize};
use std::array::from_fn;

#[repr(C)]
pub struct Q4_0 {
    delta: f16,
    quants: [u8; _32 / 2],
}

impl_data_block! {
    Q4_0 = crate::types::Q4_0;
    Self {
        delta: f16::ZERO,
        quants: [0; _32 / 2],
    }
}

impl Quantize<f32, _32> for Q4_0 {
    fn quantize(data: &[f32; _32]) -> Self {
        #[allow(clippy::assertions_on_constants)]
        const {
            assert!(Self::COUNT == _32)
        }

        let max = max_by_abs(data);
        if max == 0. {
            return Self::ZEROS;
        }

        let delta = max / -8.;
        let recip = delta.recip();
        let f = |x: f32| (x * recip + 8.5).min(15.) as u8;

        let (l, h) = data.split_at(_32 / 2);
        Self {
            delta: f16::from_f32(delta),
            quants: from_fn(|i| (f(h[i]) << 4) | f(l[i])),
        }
    }

    fn dequantize(&self) -> [f32; _32] {
        let delta = self.delta.to_f32();
        let f = |x| (x as i32 - 8) as f32 * delta;

        let mut ans = [0.; _32];
        let (l, h) = ans.split_at_mut(_32 / 2);
        for (i, &x) in self.quants.iter().enumerate() {
            l[i] = f(x & 0xf);
            h[i] = f(x >> 4);
        }
        ans
    }
}

#[test]
fn test_q4_0() {
    crate::test_utils::test::<32_, Q4_0>(8e-2, 0.);
}
