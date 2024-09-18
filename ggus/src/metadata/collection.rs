use super::{GGufFileType, GGufMetaDataValueType as Ty, GGufMetaValueArray};
use crate::{GGufReadError, GGufReader};

pub trait GGufMetaMap {
    fn get(&self, key: &str) -> Option<(Ty, &[u8])>;
}

#[derive(Debug)]
pub enum GGufMetaError {
    NotExist,
    TypeMismatch(Ty),
    ArrTypeMismatch(Ty),
    OutOfRange,
    Read(GGufReadError),
}

pub trait GGufMetaMapExt: GGufMetaMap {
    fn get_str(&self, key: &str) -> Result<&str, GGufMetaError> {
        let (ty, val) = self.get(key).ok_or(GGufMetaError::NotExist)?;
        match ty {
            Ty::String => GGufReader::new(val).read_str().map_err(GGufMetaError::Read),
            _ => Err(GGufMetaError::TypeMismatch(ty)),
        }
    }

    fn get_usize(&self, key: &str) -> Result<usize, GGufMetaError> {
        let (ty, val) = self.get(key).ok_or(GGufMetaError::NotExist)?;

        macro_rules! read {
            ($ty:ty) => {
                GGufReader::new(val)
                    .read::<$ty>()
                    .map_err(GGufMetaError::Read)?
            };
        }
        macro_rules! convert {
            ($val:expr) => {
                $val.try_into().map_err(|_| GGufMetaError::OutOfRange)?
            };
        }

        #[rustfmt::skip]
        let ans = match ty {
            Ty::U8   =>          read!(u8 ).into(),
            Ty::U16  =>          read!(u16).into(),
            Ty::U32  => convert!(read!(u32)      ),
            Ty::U64  => convert!(read!(u64)      ),
            Ty::I8   => convert!(read!(i8 )      ),
            Ty::I16  => convert!(read!(i16)      ),
            Ty::I32  => convert!(read!(i32)      ),
            Ty::I64  => convert!(read!(i64)      ),
            Ty::Bool => if read!(bool) { 1 } else { 0 },
            _        => return Err(GGufMetaError::TypeMismatch(ty)),
        };

        Ok(ans)
    }

    fn get_f32(&self, key: &str) -> Result<f32, GGufMetaError> {
        let (ty, val) = self.get(key).ok_or(GGufMetaError::NotExist)?;
        if ty == Ty::F32 {
            GGufReader::new(val).read().map_err(GGufMetaError::Read)
        } else {
            Err(GGufMetaError::TypeMismatch(ty))
        }
    }

    fn get_u32(&self, key: &str) -> Result<u32, GGufMetaError> {
        let (ty, val) = self.get(key).ok_or(GGufMetaError::NotExist)?;
        if ty == Ty::U32 {
            GGufReader::new(val).read().map_err(GGufMetaError::Read)
        } else {
            Err(GGufMetaError::TypeMismatch(ty))
        }
    }

    fn get_bool(&self, key: &str) -> Result<bool, GGufMetaError> {
        let (ty, val) = self.get(key).ok_or(GGufMetaError::NotExist)?;
        if ty == Ty::Bool {
            GGufReader::new(val)
                .read_bool()
                .map_err(GGufMetaError::Read)
        } else {
            Err(GGufMetaError::TypeMismatch(ty))
        }
    }

    fn get_str_arr(&self, key: &str) -> Result<GGufMetaValueArray<str>, GGufMetaError> {
        let (ty, val) = self.get(key).ok_or(GGufMetaError::NotExist)?;
        let mut reader = GGufReader::new(val);
        let (ty, len) = match ty {
            Ty::Array => reader.read_arr_header().map_err(GGufMetaError::Read)?,
            ty => return Err(GGufMetaError::TypeMismatch(ty)),
        };
        if ty == Ty::String {
            Ok(GGufMetaValueArray::new(reader, len))
        } else {
            Err(GGufMetaError::ArrTypeMismatch(ty))
        }
    }

    fn get_i32_arr(&self, key: &str) -> Result<GGufMetaValueArray<i32>, GGufMetaError> {
        let (ty, val) = self.get(key).ok_or(GGufMetaError::NotExist)?;
        let mut reader = GGufReader::new(val);
        let (ty, len) = match ty {
            Ty::Array => reader.read_arr_header().map_err(GGufMetaError::Read)?,
            ty => return Err(GGufMetaError::TypeMismatch(ty)),
        };
        if ty == Ty::I32 {
            Ok(GGufMetaValueArray::new(reader, len))
        } else {
            Err(GGufMetaError::ArrTypeMismatch(ty))
        }
    }

    fn get_f32_arr(&self, key: &str) -> Result<GGufMetaValueArray<f32>, GGufMetaError> {
        let (ty, val) = self.get(key).ok_or(GGufMetaError::NotExist)?;
        let mut reader = GGufReader::new(val);
        let (ty, len) = match ty {
            Ty::Array => reader.read_arr_header().map_err(GGufMetaError::Read)?,
            ty => return Err(GGufMetaError::TypeMismatch(ty)),
        };
        if ty == Ty::F32 {
            Ok(GGufMetaValueArray::new(reader, len))
        } else {
            Err(GGufMetaError::ArrTypeMismatch(ty))
        }
    }

    #[inline]
    fn general_architecture(&self) -> Result<&str, GGufMetaError> {
        self.get_str("general.architecture")
    }

    #[inline]
    fn general_quantization_version(&self) -> Result<usize, GGufMetaError> {
        self.get_usize("general.quantization_version")
    }

    #[inline]
    fn general_alignment(&self) -> Result<usize, GGufMetaError> {
        self.get_usize("general.alignment")
    }

    #[inline]
    fn general_name(&self) -> Result<&str, GGufMetaError> {
        self.get_str("general.name")
    }

    #[inline]
    fn general_author(&self) -> Result<&str, GGufMetaError> {
        self.get_str("general.author")
    }

    #[inline]
    fn general_version(&self) -> Result<&str, GGufMetaError> {
        self.get_str("general.version")
    }

    #[inline]
    fn general_organization(&self) -> Result<&str, GGufMetaError> {
        self.get_str("general.organization")
    }

    #[inline]
    fn general_basename(&self) -> Result<&str, GGufMetaError> {
        self.get_str("general.basename")
    }

    #[inline]
    fn general_finetune(&self) -> Result<&str, GGufMetaError> {
        self.get_str("general.finetune")
    }

    #[inline]
    fn general_description(&self) -> Result<&str, GGufMetaError> {
        self.get_str("general.description")
    }

    #[inline]
    fn general_quantized_by(&self) -> Result<&str, GGufMetaError> {
        self.get_str("general.quantized_by")
    }

    #[inline]
    fn general_size_label(&self) -> Result<&str, GGufMetaError> {
        self.get_str("general.size_label")
    }

    #[inline]
    fn general_license(&self) -> Result<&str, GGufMetaError> {
        self.get_str("general.license")
    }

    #[inline]
    fn general_license_name(&self) -> Result<&str, GGufMetaError> {
        self.get_str("general.license.name")
    }

    #[inline]
    fn general_license_link(&self) -> Result<&str, GGufMetaError> {
        self.get_str("general.license.link")
    }

    #[inline]
    fn general_url(&self) -> Result<&str, GGufMetaError> {
        self.get_str("general.url")
    }

    #[inline]
    fn general_doi(&self) -> Result<&str, GGufMetaError> {
        self.get_str("general.doi")
    }

    #[inline]
    fn general_uuid(&self) -> Result<&str, GGufMetaError> {
        self.get_str("general.uuid")
    }

    #[inline]
    fn general_repo_url(&self) -> Result<&str, GGufMetaError> {
        self.get_str("general.repo_url")
    }

    #[inline]
    fn general_tags(&self) -> Result<GGufMetaValueArray<str>, GGufMetaError> {
        self.get_str_arr("general.tags")
    }

    #[inline]
    fn general_languages(&self) -> Result<GGufMetaValueArray<str>, GGufMetaError> {
        self.get_str_arr("general.languages")
    }

    #[inline]
    fn general_datasets(&self) -> Result<GGufMetaValueArray<str>, GGufMetaError> {
        self.get_str_arr("general.datasets")
    }

    #[inline]
    fn general_filetype(&self) -> Result<GGufFileType, GGufMetaError> {
        (self.get_usize("general.filetype")? as u32)
            .try_into()
            .map_err(|_| GGufMetaError::OutOfRange)
    }

    #[inline]
    fn general_source_url(&self) -> Result<&str, GGufMetaError> {
        self.get_str("general.source.url")
    }

    #[inline]
    fn general_source_doi(&self) -> Result<&str, GGufMetaError> {
        self.get_str("general.source.doi")
    }

    #[inline]
    fn general_source_uuid(&self) -> Result<&str, GGufMetaError> {
        self.get_str("general.source.uuid")
    }

    #[inline]
    fn general_source_repo_url(&self) -> Result<&str, GGufMetaError> {
        self.get_str("general.source.repo_url")
    }

    #[inline]
    fn general_base_model_count(&self) -> Result<usize, GGufMetaError> {
        self.get_usize("general.base_model.count")
    }

    #[inline]
    fn general_base_model_name(&self, id: usize) -> Result<&str, GGufMetaError> {
        self.get_str(&format!("general.base_model.{id}.name"))
    }

    #[inline]
    fn general_base_model_author(&self, id: usize) -> Result<&str, GGufMetaError> {
        self.get_str(&format!("general.base_model.{id}.author"))
    }

    #[inline]
    fn general_base_model_version(&self, id: usize) -> Result<&str, GGufMetaError> {
        self.get_str(&format!("general.base_model.{id}.version"))
    }

    #[inline]
    fn general_base_model_organization(&self, id: usize) -> Result<&str, GGufMetaError> {
        self.get_str(&format!("general.base_model.{id}.organization"))
    }

    #[inline]
    fn general_base_model_url(&self, id: usize) -> Result<&str, GGufMetaError> {
        self.get_str(&format!("general.base_model.{id}.url"))
    }

    #[inline]
    fn general_base_model_doi(&self, id: usize) -> Result<&str, GGufMetaError> {
        self.get_str(&format!("general.base_model.{id}.doi"))
    }

    #[inline]
    fn general_base_model_uuid(&self, id: usize) -> Result<&str, GGufMetaError> {
        self.get_str(&format!("general.base_model.{id}.uuid"))
    }

    #[inline]
    fn general_base_model_repo_url(&self, id: usize) -> Result<&str, GGufMetaError> {
        self.get_str(&format!("general.base_model.{id}.repo_url"))
    }

    #[inline]
    fn llm_context_length(&self) -> Result<usize, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_usize(&format!("{llm}.context_length"))
    }

    #[inline]
    fn llm_embedding_length(&self) -> Result<usize, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_usize(&format!("{llm}.embedding_length"))
    }

    #[inline]
    fn llm_block_count(&self) -> Result<usize, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_usize(&format!("{llm}.block_count"))
    }

    #[inline]
    fn llm_feed_forward_length(&self) -> Result<usize, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_usize(&format!("{llm}.feed_forward_length"))
    }

    #[inline]
    fn llm_use_parallel_residual(&self) -> Result<bool, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_bool(&format!("{llm}.use_parallel_residual"))
    }

    #[inline]
    fn llm_tensor_data_layout(&self) -> Result<&str, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_str(&format!("{llm}.tensor_data_layout"))
    }

    #[inline]
    fn llm_expert_count(&self) -> Result<usize, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_usize(&format!("{llm}.expert_count"))
    }

    #[inline]
    fn llm_expert_used_count(&self) -> Result<usize, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_usize(&format!("{llm}.expert_used_count"))
    }

    #[inline]
    fn llm_attention_head_count(&self) -> Result<usize, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_usize(&format!("{llm}.attention.head_count"))
    }

    #[inline]
    fn llm_attention_head_count_kv(&self) -> Result<usize, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_usize(&format!("{llm}.attention.head_count_kv"))
    }

    #[inline]
    fn llm_attention_max_alibi_bias(&self) -> Result<f32, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_f32(&format!("{llm}.attention.max_alibi_bias"))
    }

    #[inline]
    fn llm_attention_clamp_kqv(&self) -> Result<f32, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_f32(&format!("{llm}.attention.clamp_kqv"))
    }

    #[inline]
    fn llm_attention_layer_norm_epsilon(&self) -> Result<f32, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_f32(&format!("{llm}.attention.layer_norm_epsilon"))
    }

    #[inline]
    fn llm_attention_layer_norm_rms_epsilon(&self) -> Result<f32, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_f32(&format!("{llm}.attention.layer_norm_rms_epsilon"))
    }

    #[inline]
    fn llm_attention_key_length(&self) -> Result<usize, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_usize(&format!("{llm}.attention.key_length"))
    }

    #[inline]
    fn llm_attention_value_length(&self) -> Result<usize, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_usize(&format!("{llm}.attention.value_length"))
    }

    #[inline]
    fn llm_rope_dimension_count(&self) -> Result<usize, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_usize(&format!("{llm}.rope.dimension_count"))
    }

    #[inline]
    fn llm_rope_freq_base(&self) -> Result<f32, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_f32(&format!("{llm}.rope.freq_base"))
    }

    #[inline]
    fn llm_rope_scaling_type(&self) -> Result<&str, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_str(&format!("{llm}.rope.scaling.type"))
    }

    #[inline]
    fn llm_rope_scaling_factor(&self) -> Result<f32, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_f32(&format!("{llm}.rope.scaling.type"))
    }

    #[inline]
    fn llm_rope_scaling_original_context_length(&self) -> Result<usize, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_usize(&format!("{llm}.rope.scaling.original_context_length"))
    }

    #[inline]
    fn llm_rope_scaling_finetuned(&self) -> Result<bool, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_bool(&format!("{llm}.rope.scaling.finetuned"))
    }

    #[inline]
    fn llm_rope_scale_linear(&self) -> Result<f32, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_f32(&format!("{llm}.rope.scale_linear"))
    }

    #[inline]
    fn llm_ssm_conv_kernel(&self) -> Result<usize, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_usize(&format!("{llm}.ssm.conv_kernel"))
    }

    #[inline]
    fn llm_ssm_inner_size(&self) -> Result<usize, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_usize(&format!("{llm}.ssm.inner_size"))
    }

    #[inline]
    fn llm_ssm_state_size(&self) -> Result<usize, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_usize(&format!("{llm}.ssm.state_size"))
    }

    #[inline]
    fn llm_ssm_time_step_rank(&self) -> Result<usize, GGufMetaError> {
        let llm = self.general_architecture().unwrap();
        self.get_usize(&format!("{llm}.ssm.time_step_rank"))
    }

    #[inline]
    fn tokenizer_ggml_model(&self) -> Result<&str, GGufMetaError> {
        self.get_str("tokenizer.ggml.model")
    }

    #[inline]
    fn tokenizer_ggml_tokens(&self) -> Result<GGufMetaValueArray<str>, GGufMetaError> {
        self.get_str_arr("tokenizer.ggml.tokens")
    }

    #[inline]
    fn tokenizer_ggml_scores(&self) -> Result<GGufMetaValueArray<f32>, GGufMetaError> {
        self.get_f32_arr("tokenizer.ggml.scores")
    }

    #[inline]
    fn tokenizer_ggml_token_type(&self) -> Result<GGufMetaValueArray<i32>, GGufMetaError> {
        self.get_i32_arr("tokenizer.ggml.token_type")
    }

    #[inline]
    fn tokenizer_ggml_merges(&self) -> Result<GGufMetaValueArray<str>, GGufMetaError> {
        self.get_str_arr("tokenizer.ggml.merges")
    }

    #[inline]
    fn tokenizer_ggml_added_tokens(&self) -> Result<GGufMetaValueArray<str>, GGufMetaError> {
        self.get_str_arr("tokenizer.ggml.added_tokens")
    }

    #[inline]
    fn tokenizer_ggml_bos_token_id(&self) -> Result<u32, GGufMetaError> {
        self.get_u32("tokenizer.ggml.bos_token_id")
    }

    #[inline]
    fn tokenizer_ggml_eos_token_id(&self) -> Result<u32, GGufMetaError> {
        self.get_u32("tokenizer.ggml.eos_token_id")
    }

    #[inline]
    fn tokenizer_ggml_unknown_token_id(&self) -> Result<u32, GGufMetaError> {
        self.get_u32("tokenizer.ggml.unknown_token_id")
    }

    #[inline]
    fn tokenizer_ggml_separator_token_id(&self) -> Result<u32, GGufMetaError> {
        self.get_u32("tokenizer.ggml.separator_token_id")
    }

    #[inline]
    fn tokenizer_ggml_padding_token_id(&self) -> Result<u32, GGufMetaError> {
        self.get_u32("tokenizer.ggml.padding_token_id")
    }
}

impl<T: GGufMetaMap> GGufMetaMapExt for T {}
