use super::{
    super::{DataPromise, MetaValue, Tensor},
    Content, BLK_TENSOR_REGEX,
};
use ggus::{
    ggml_quants::{bf16, f16},
    DataFuture, GGmlType, GGufMetaMapExt,
};
use memmap2::MmapMut;
use std::{alloc::Layout, ops::MulAssign};

impl Content<'_> {
    pub(super) fn convert_to_llama(&mut self, extra: Option<String>) {
        match self.general_architecture() {
            Ok("llama") => {}
            Ok("minicpm") => from_minicpm(self, extra),
            Ok(arch) => todo!("Unsupported architecture: {arch:?}"),
            Err(e) => panic!("Faild to read general architecture: {e:?}"),
        }
    }
}

fn from_minicpm(content: &mut Content, extra: Option<String>) {
    const ERR_MSG: &str =
        "convert from minicpm requires extra args \"embd_scale=<num> res_scale=<num>\"";

    let nblk = content.llm_block_count().expect("Missing llm_block_count");
    let mut embd_scale = None;
    let mut res_scale = None;
    for s in extra.expect(ERR_MSG).split_whitespace() {
        let (k, v) = s.split_once('=').expect(ERR_MSG);
        match k {
            "embd_scale" => embd_scale = Some(v.parse::<f64>().expect(ERR_MSG)),
            "res_scale" => res_scale = Some(v.parse::<f64>().expect(ERR_MSG)),
            _ => {}
        }
    }
    let embd_scale = embd_scale.expect(ERR_MSG);
    let res_scale = res_scale.expect(ERR_MSG) / (nblk as f64).sqrt();

    for (name, tensor) in content.tensors.iter_mut() {
        if name == "token_embd.weight" {
            scale_tensor(tensor, embd_scale);
        } else if let Some(captures) = BLK_TENSOR_REGEX.captures(name) {
            match &captures[2] {
                "attn_output" | "ffn_down" => scale_tensor(tensor, res_scale),
                _ => {}
            }
        }
    }

    set_arch(content, "minicpm", "llama")
}

fn set_arch(content: &mut Content, old: &str, new: &str) {
    let old = format!("{old}.");
    for (k, v) in std::mem::take(&mut content.meta_kvs) {
        if k == "general.architecture" {
            content.meta_kvs.insert(k, MetaValue::string(new));
        } else {
            let k = match k.strip_prefix(&old) {
                Some(body) => format!("{new}.{body}").into(),
                None => k,
            };
            content.meta_kvs.insert(k, v);
        }
    }
}

fn scale_tensor(tensor: &mut Tensor, scale: f64) {
    let data = tensor.data.clone();
    tensor.data = match tensor.ty {
        GGmlType::F64 => DataPromise::lazy(move || scale_data(data.get(), scale)),
        GGmlType::F32 => DataPromise::lazy(move || scale_data(data.get(), scale as f32)),
        GGmlType::F16 => DataPromise::lazy(move || scale_data(data.get(), f16::from_f64(scale))),
        GGmlType::BF16 => DataPromise::lazy(move || scale_data(data.get(), bf16::from_f64(scale))),
        ty => todo!("unsupported tensor type: {ty:?}"),
    };
}

fn scale_data<T: MulAssign + Clone>(data: &[u8], scale: T) -> MmapMut {
    assert_eq!(data.len() % size_of::<T>(), 0);
    let len = data.len() / size_of::<T>();

    let mut ans = MmapMut::map_anon(Layout::array::<T>(len).unwrap().size()).unwrap();
    ans.copy_from_slice(data);

    let (&mut [], data, &mut []) = (unsafe { ans.align_to_mut::<T>() }) else {
        panic!("data not aligned")
    };
    for x in data {
        *x *= scale.clone();
    }

    ans
}
