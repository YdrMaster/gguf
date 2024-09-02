use super::QuantBlock;
use half::f16;
use std::iter::zip;

#[repr(C)]
pub struct Q8_0 {
    delta: f16,
    quants: [i8; BLOCK_SIZE],
}
const BLOCK_SIZE: usize = 32;

impl QuantBlock<f32, BLOCK_SIZE> for Q8_0 {
    fn quantize(data: &[f32; BLOCK_SIZE]) -> Self {
        let amax = data.iter().fold(0., |acc, x| x.abs().max(acc));

        let delta = amax / i8::MAX as f32;
        let recip = if delta == 0. { 0. } else { delta.recip() };

        let delta = f16::from_f32(delta);
        let mut quants = [0; BLOCK_SIZE];
        for (y, &x) in zip(&mut quants, data) {
            *y = (x * recip).round() as _;
        }
        Self { delta, quants }
    }
    fn dequantize(&self) -> [f32; BLOCK_SIZE] {
        let mut ans = [0.; BLOCK_SIZE];
        let delta = self.delta.to_f32();
        for (y, &x) in zip(&mut ans, &self.quants) {
            *y = x as f32 * delta;
        }
        ans
    }
}
