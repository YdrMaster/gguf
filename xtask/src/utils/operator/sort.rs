use super::{Content, Operator};
use itertools::Itertools;
use std::{collections::HashMap, sync::LazyLock};

impl Operator {
    #[inline]
    pub fn sort_tensors() -> Self {
        Self::SortTensors
    }
}

impl Content<'_> {
    pub(super) fn sort_tensors(&mut self) {
        self.assert_llama();
        let tensors = std::mem::take(&mut self.tensors);
        self.tensors = tensors
            .into_iter()
            .sorted_unstable_by_key(|(k, _)| rank(k).unwrap_or(usize::MAX))
            .collect();
    }
}

fn rank(name: &str) -> Option<usize> {
    let (head, tail): (&str, usize);
    if let Some(name) = name.strip_suffix(".weight") {
        head = name;
        tail = 0;
    } else {
        head = name.strip_suffix(".bias")?;
        tail = 1;
    };

    static ORDER_MAP: LazyLock<HashMap<&str, usize>> = LazyLock::new(|| {
        [
            "token_embd",
            "output_norm",
            "output",
            "attn_norm",
            "attn_norm_2",
            "attn_qkv",
            "attn_q",
            "attn_k",
            "attn_v",
            "attn_output",
            "ffn_norm",
            "ffn_gate_up",
            "ffn_up",
            "ffn_gate",
            "ffn_down",
            "ffn_up_exp",
            "ffn_up_exps",
            "ffn_gate_inp",
            "ffn_gate_exp",
            "ffn_gate_exps",
            "ffn_down_exp",
            "ffn_down_exps",
        ]
        .iter()
        .enumerate()
        .map(|(i, s)| (*s, i))
        .collect()
    });

    let head = match head.strip_prefix("blk.") {
        Some(body) => {
            let (blk, name) = body.split_once('.')?;
            blk.parse::<usize>().unwrap() * ORDER_MAP.len() + *ORDER_MAP.get(name)?
        }
        None => *ORDER_MAP.get(head)?,
    };
    Some(head * 2 + tail)
}
