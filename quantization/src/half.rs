use super::QuantBlock;

impl QuantBlock<f32, 1> for half::f16 {
    fn quantize(&[data]: &[f32; 1]) -> Self {
        half::f16::from_f32(data)
    }
    fn dequantize(&self) -> [f32; 1] {
        [self.to_f32()]
    }
}

impl QuantBlock<f32, 1> for half::bf16 {
    fn quantize(&[data]: &[f32; 1]) -> Self {
        half::bf16::from_f32(data)
    }
    fn dequantize(&self) -> [f32; 1] {
        [self.to_f32()]
    }
}
