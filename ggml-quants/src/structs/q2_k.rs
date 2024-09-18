use super::{DeltaMin, _256};
use crate::{DataBlock, Quantize};
use itertools::Itertools;

#[repr(C)]
pub struct Q2K {
    scales: [u8; _256 / 16],
    qs: [u8; _256 / 4],
    delta_min: DeltaMin,
}

impl DataBlock for Q2K {
    #[cfg(feature = "types")]
    const ID: digit_layout::DigitLayout = crate::types::Q2K;
    const COUNT: usize = _256;
    const ZEROS: Self = Self {
        scales: [0; _256 / 16],
        qs: [0; _256 / 4],
        delta_min: DeltaMin::ZERO,
    };
}

impl Quantize<f32, _256> for Q2K {
    fn quantize(_data: &[f32; _256]) -> Self {
        todo!()
    }
    fn dequantize(&self) -> [f32; _256] {
        let mut ans = [0.; _256];
        let (delta, min) = self.delta_min.to_f32();
        let de_qs = self
            .qs
            .iter()
            .flat_map(|&q| [q & 0b11, q >> 2 & 0b11, q >> 4 & 0b11, q >> 6 & 0b11]);
        let dl_and_ml = self
            .scales
            .iter()
            .map(|sc| (delta * ((sc & 0xF) as f32), min * ((sc >> 4) as f32)));

        let de_qs_chunks = de_qs.chunks(16);
        let result = de_qs_chunks
            .into_iter()
            .zip(dl_and_ml)
            .map(|(qs, (delta_l, min_l))| qs.map(move |n| delta_l * n as f32 - min_l))
            .flatten();

        let mut count = 0;
        for (i, q) in result.enumerate() {
            ans[i] = q;
            count += 1;
        }
        assert!(count == _256);
        ans
    }
}
