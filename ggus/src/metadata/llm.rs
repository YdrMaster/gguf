use super::{GGufMetaDataValueType, GGufMetaKVPairs};

pub struct LlmMeta<'a>(GGufMetaKVPairs<'a>, &'a str);

impl<'a> GGufMetaKVPairs<'a> {
    #[inline]
    pub fn llm(self, arch: Option<&'a str>) -> LlmMeta<'a> {
        let arch = arch.or_else(|| self.architecture()).unwrap();
        LlmMeta(self, arch)
    }
}

impl<'a> LlmMeta<'a> {
    #[inline]
    pub fn all(self) -> GGufMetaKVPairs<'a> {
        self.0
    }
}

macro_rules! get {
    ($self:expr, $key:literal @ $ty:ident) => {
        $self.0.get_typed(format!("{}.{}", $self.1, $key), GGufMetaDataValueType::$ty)
    };

    ($self:expr, $key:literal(u32)) => {
        get!($self, $key @ U32).map(|mut reader| reader.read::<u32>().unwrap() as _)
    };

    ($self:expr, $key:literal(u64)) => {
        get!($self, $key @ U64).map(|mut reader| reader.read::<u64>().unwrap() as _)
    };

    ($self:expr, $key:literal(f32)) => {
        get!($self, $key @ F32).map(|mut reader| reader.read().unwrap())
    };

    ($self:expr, $key:literal(bool)) => {
        get!($self, $key @ Bool).map(|mut reader| reader.read_bool().unwrap())
    };

    ($self:expr, $key:literal(str)) => {
        get!($self, $key @ String).map(|mut reader| reader.read_str().unwrap())
    };
}

// llm
impl<'a> LlmMeta<'a> {
    #[inline]
    pub fn context_length(&self) -> Option<usize> {
        get!(self, "context_length"(u64))
    }

    #[inline]
    pub fn embedding_length(&self) -> Option<usize> {
        get!(self, "embedding_length"(u64))
    }

    #[inline]
    pub fn block_count(&self) -> Option<usize> {
        get!(self, "block_count"(u64))
    }

    #[inline]
    pub fn feed_forward_length(&self) -> Option<usize> {
        get!(self, "feed_forward_length"(u64))
    }

    #[inline]
    pub fn use_parallel_residual(&self) -> Option<bool> {
        get!(self, "use_parallel_residual"(bool))
    }

    #[inline]
    pub fn tensor_data_layout(&self) -> Option<&'a str> {
        get!(self, "tensor_data_layout"(str))
    }

    #[inline]
    pub fn expert_count(&self) -> Option<usize> {
        get!(self, "expert_count"(u32))
    }

    #[inline]
    pub fn expert_used_count(&self) -> Option<usize> {
        get!(self, "expert_used_count"(u32))
    }
}

// attention
impl<'a> LlmMeta<'a> {
    #[inline]
    pub fn attention_head_count(&self) -> Option<usize> {
        get!(self, "attention.head_count"(u64))
    }

    #[inline]
    pub fn attention_head_count_kv(&self) -> Option<usize> {
        get!(self, "attention.head_count_kv"(u64))
    }

    #[inline]
    pub fn attention_max_alibi_bias(&self) -> Option<f32> {
        get!(self, "attention.max_alibi_bias"(f32))
    }

    #[inline]
    pub fn attention_clamp_kqv(&self) -> Option<f32> {
        get!(self, "attention.clamp_kqv"(f32))
    }

    #[inline]
    pub fn attention_layer_norm_epsilon(&self) -> Option<f32> {
        get!(self, "attention.layer_norm_epsilon"(f32))
    }

    #[inline]
    pub fn attention_layer_norm_rms_epsilon(&self) -> Option<f32> {
        get!(self, "attention.layer_norm_rms_epsilon"(f32))
    }

    #[inline]
    pub fn attention_key_length(&self) -> Option<usize> {
        get!(self, "attention.key_length"(u32))
    }

    #[inline]
    pub fn attention_value_length(&self) -> Option<usize> {
        get!(self, "attention.value_length"(u32))
    }
}

// rope
impl<'a> LlmMeta<'a> {
    #[inline]
    pub fn rope_dimension_count(&self) -> Option<usize> {
        get!(self, "rope.dimension_count"(u64))
    }

    #[inline]
    pub fn rope_freq_base(&self) -> Option<f32> {
        get!(self, "rope.freq_base"(f32))
    }

    #[inline]
    pub fn rope_scaling_type(&self) -> Option<&'a str> {
        get!(self, "rope.scaling.type"(str))
    }

    #[inline]
    pub fn rope_scaling_factor(&self) -> Option<f32> {
        get!(self, "rope.scaling.factor"(f32))
    }

    #[inline]
    pub fn rope_scaling_original_context_length(&self) -> Option<usize> {
        get!(self, "rope.scaling.original_context_length"(u32))
    }

    #[inline]
    pub fn rope_scaling_finetuned(&self) -> Option<bool> {
        get!(self, "rope.scaling.finetuned"(bool))
    }

    #[deprecated = "It is recommended that models use the newer keys if possible, as they are more flexible and allow for more complex scaling schemes."]
    #[inline]
    pub fn rope_scale_linear(&self) -> Option<f32> {
        get!(self, "rope.scale_linear"(f32))
    }
}

// ssm
impl<'a> LlmMeta<'a> {
    #[inline]
    pub fn ssm_conv_kernel(&self) -> Option<usize> {
        get!(self, "ssm.conv_kernel"(u32))
    }

    #[inline]
    pub fn ssm_inner_size(&self) -> Option<usize> {
        get!(self, "ssm.inner_size"(u32))
    }

    #[inline]
    pub fn ssm_state_size(&self) -> Option<usize> {
        get!(self, "ssm.state_size"(u32))
    }

    #[inline]
    pub fn ssm_time_step_rank(&self) -> Option<usize> {
        get!(self, "ssm.time_step_rank"(u32))
    }
}
