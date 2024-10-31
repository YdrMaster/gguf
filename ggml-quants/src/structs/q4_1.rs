use super::{min_max, DeltaMin, _32};
use crate::{DataBlock, Quantize};
use std::array::from_fn;

#[repr(C)]
pub struct Q4_1 {
    delta_min: DeltaMin,
    quants: [u8; _32 / 2],
}

impl DataBlock for Q4_1 {
    #[cfg(feature = "types")]
    const ID: digit_layout::DigitLayout = crate::types::Q4_1;
    const COUNT: usize = _32;
    const ZEROS: Self = Self {
        delta_min: DeltaMin::ZERO,
        quants: [0; _32 / 2],
    };
}

impl Quantize<f32, _32> for Q4_1 {
    fn quantize(data: &[f32; _32]) -> Self {
        #[allow(clippy::assertions_on_constants)]
        const {
            assert!(Self::COUNT == _32)
        }

        let (min, max) = min_max(data);
        if min == max {
            return Self {
                delta_min: DeltaMin::no_delta(min),
                quants: [0; _32 / 2],
            };
        }

        let delta = (max - min) / ((1 << 4) - 1) as f32;
        let recip = delta.recip();
        let f = |x| (((x - min) * recip + 0.5) as u8).min(15);

        let (l, h) = data.split_at(_32 / 2);
        Self {
            delta_min: DeltaMin::new(delta, min),
            quants: from_fn(|i| (f(h[i]) << 4) | f(l[i])),
        }
    }

    fn dequantize(&self) -> [f32; _32] {
        let (delta, min) = self.delta_min.to_f32();
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

#[test]
fn test_q4_1() {
    use crate::test_utils::{Diff, ErrorCollector};
    use rand::Rng;
    use std::iter::zip;

    let mut data = [0.0f32; _32];
    rand::thread_rng().fill(&mut data);

    let q4_1 = Q4_1::quantize(&data);
    let deq = <Q4_1 as Quantize<f32, _32>>::dequantize(&q4_1);

    let mut ec = ErrorCollector::new(4e-2, 0.);
    for (a, b) in zip(data, deq) {
        ec.push(Diff::new(a, b))
    }
    println!("{ec}");

    for &i in ec.outliers() {
        println!("{} vs {}", data[i], deq[i]);
    }

    assert!(ec.outliers().is_empty());
}
