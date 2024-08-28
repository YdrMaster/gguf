use super::{super::MetaValue, Content, Operator};
use ggus::GGufMetaDataValueType;
use log::{info, warn};
use std::borrow::Cow;

impl Operator {
    #[inline]
    pub fn distribute(n: impl AsRef<str>) -> Self {
        Self::Distribute(n.as_ref().parse().unwrap())
    }
}

const META_TY: GGufMetaDataValueType = GGufMetaDataValueType::U32;
const META_KEY: fn(&str) -> Cow<'static, str> = |arch| format!("{}.distribute", arch).into();
const META_VAL: fn(n: usize) -> Cow<'static, [u8]> = |n| (n as u32).to_le_bytes().to_vec().into();

impl Content<'_> {
    pub(super) fn distribute_meta(&self) -> usize {
        self.meta_kvs.get(&META_KEY(self.arch())).map_or(1, |val| {
            assert_eq!(val.ty, META_TY);
            val.value_reader().read::<u32>().unwrap() as _
        })
    }

    pub(super) fn distribute(&mut self, n: usize) {
        let current = self.distribute_meta();
        if current == n {
            info!("Model already distributed to {n} parts, skip this step.");
            return;
        }

        if self.is_linear_merged() {
            warn!("Distribute linear-merged model is not supported yet, do split->distribute->merge instead.");
            self.merge_linear(false);
            self.distribute(n);
            self.merge_linear(true);
            return;
        }

        self.assert_llama();
        match n {
            0 => unreachable!("Cannot distribute to 0 parts"),
            1 => self.gather(),
            _ => self.distribute_(n),
        }
    }

    fn distribute_(&mut self, n: usize) {
        use indexmap::map::Entry::{Occupied, Vacant};
        match self.meta_kvs.entry(META_KEY(self.arch())) {
            Occupied(mut entry) => {
                entry.get_mut().value = META_VAL(n);
            }
            Vacant(entry) => {
                entry.insert(MetaValue {
                    ty: META_TY,
                    value: META_VAL(n),
                });
            }
        }

        todo!()
    }

    fn gather(&mut self) {
        self.meta_kvs.shift_remove(&META_KEY(self.arch()));

        todo!()
    }
}
