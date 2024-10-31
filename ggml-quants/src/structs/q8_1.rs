use super::{max_abs, _32};
use crate::{DataBlock, Quantize};
use half::f16;
use std::iter::zip;

// TODO: 比 [Q8_0](crate::Q8_0) 多算了一个 sum，不知道有什么用
#[repr(C, align(4))]
pub struct Q8_1 {
    delta: f16,
    sum: f16,
    quants: [i8; _32],
}

impl DataBlock for Q8_1 {
    #[cfg(feature = "types")]
    const ID: digit_layout::DigitLayout = crate::types::Q8_1;
    const COUNT: usize = _32;
    const ZEROS: Self = Self {
        delta: f16::ZERO,
        sum: f16::ZERO,
        quants: [0; _32],
    };
}

impl Quantize<f32, _32> for Q8_1 {
    fn quantize(data: &[f32; _32]) -> Self {
        #[allow(clippy::assertions_on_constants)]
        const {
            assert!(Self::COUNT == _32)
        }

        let amax = max_abs(data);
        if amax == 0. {
            return Self::ZEROS;
        }

        let delta = amax / ((1 << 7) - 1) as f32;
        let recip: f32 = delta.recip();

        let mut sum = 0;
        let mut quants = [0; _32];
        for (y, x) in zip(&mut quants, data) {
            *y = (x * recip).round() as i8;
            sum += *y as i16;
        }

        Self {
            delta: f16::from_f32(delta),
            sum: f16::from_f32(sum as f32 * delta),
            quants,
        }
    }

    #[inline]
    fn dequantize(&self) -> [f32; _32] {
        let delta = self.delta.to_f32();
        self.quants.map(|x| x as f32 * delta)
    }
}

#[test]
fn test_q8_1() {
    use crate::test_utils::{Diff, ErrorCollector};
    use rand::Rng;
    use std::iter::zip;

    let mut data = [0.0f32; _32];
    rand::thread_rng().fill(&mut data);

    let q8_1 = Q8_1::quantize(&data);
    let deq = <Q8_1 as Quantize<f32, _32>>::dequantize(&q8_1);

    let mut ec = ErrorCollector::new(4e-3, 0.);
    for (a, b) in zip(data, deq) {
        ec.push(Diff::new(a, b))
    }
    println!("{ec}");

    for &i in ec.outliers() {
        println!("{} vs {}", data[i], deq[i]);
    }

    assert!(ec.outliers().is_empty());
}
