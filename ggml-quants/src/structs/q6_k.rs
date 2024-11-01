use super::_256;
use crate::{
    structs::{make_qx_quants, max_by_abs, GROUP_MAX_EPS},
    DataBlock, Quantize,
};
use half::f16;

#[repr(C)]
pub struct Q6K {
    ql: [u8; _256 / 2],
    qh: [u8; _256 / 4],
    scales: [u8; _256 / 16],
    delta: f16,
}

impl DataBlock for Q6K {
    #[cfg(feature = "types")]
    const ID: digit_layout::DigitLayout = crate::types::Q6K;
    const COUNT: usize = _256;
    const ZEROS: Self = Self {
        ql: [0; _256 / 2],
        qh: [0; _256 / 4],
        scales: [0; _256 / 16],
        delta: f16::ZERO,
    };
}

impl Quantize<f32, _256> for Q6K {
    fn quantize(data: &[f32; _256]) -> Self {
        let (scales, L_vec): (Vec<_>, Vec<_>) = data
            .chunks(16)
            .into_iter()
            .map(|x| make_qx_quants(x, 32))
            .unzip();
        let max_abs_scale = max_by_abs(&scales);
        
        if max_abs_scale.abs() < GROUP_MAX_EPS {
            return Self::ZEROS;
        }


        todo!()
    }
    fn dequantize(&self) -> [f32; _256] {
        // let (low, high) = self.qh.split_at(32);
        // let qh = [low, high]
        let qh = self
            .qh
            .chunks(32)
            .into_iter()
            .map(|qh| {
                let qh1 = qh.iter().map(|a| ((a >> 0) & 0b11) << 4);
                let qh2 = qh.iter().map(|a| ((a >> 2) & 0b11) << 4);
                let qh3 = qh.iter().map(|a| ((a >> 4) & 0b11) << 4);
                let qh4 = qh.iter().map(|a| ((a >> 6) & 0b11) << 4);
                qh1.chain(qh2).chain(qh3).chain(qh4)
            })
            .flatten();

        // let (low, high) = self.ql.split_at(64);
        // let ql = [low, high];
        let ql = self
            .ql
            .chunks(64)
            .into_iter()
            .map(|ql| {
                let (l, h) = ql.split_at(32);
                let ql1 = l.iter().map(|a| a & 0b1111);
                let ql2 = h.iter().map(|a| a & 0b1111);
                let ql3 = l.iter().map(|a: &u8| (a >> 4) & 0b1111);
                let ql4 = h.iter().map(|a| (a >> 4) & 0b1111);
                ql1.chain(ql2).chain(ql3).chain(ql4)
            })
            .flatten();
        let y = qh
            .zip(ql)
            .zip(
                self.scales
                    .iter()
                    .flat_map(|x| std::iter::repeat(x).take(16)),
            )
            .map(|((qh, ql), scales)| {
                let q = (qh | ql) - 32;
                self.delta.to_f32() * q as f32 * *scales as f32
            });
        let mut ans: [f32; _256] = [0.; _256];
        let mut count = 0;
        for (i, q) in y.enumerate() {
            ans[i] = q;
            count += 1;
        }
        assert!(count == _256);
        ans
    }
}

#[test]
fn test_Q6K() {
    let a: f32 = 0.7;
    println!("{}", a as isize);
}
